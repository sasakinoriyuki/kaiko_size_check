"""Install a local update into the user's selected app folder, with rollback."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import tempfile
from datetime import datetime


PROTECTED_FOLDERS = {'.venv', 'data', 'output', 'backups', '_kaiko_backup', 'runtime', 'logs', 'updates'}


def safe_relative(name):
    """Validate Windows filenames even when tests run on another platform."""
    if not isinstance(name, str) or not name or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError('更新先に不正なパスが含まれています。')
    parts = name.replace('\\', '/').split('/')
    device_names = {'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'}
    device_names |= {prefix+suffix for prefix in ('com', 'lpt') for suffix in '123456789¹²³'}
    if any(not part or part in {'.', '..'} or part != part.rstrip(' .')
           or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
           or part.split('.')[0].casefold() in device_names for part in parts):
        raise ValueError('更新先に不正なWindowsファイル名が含まれています。')
    if any(part.casefold() in PROTECTED_FOLDERS for part in parts):
        raise ValueError('計測記録・Python環境・バックアップは更新対象にできません。')
    return Path(*parts)


def is_link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def check_path(root, relative):
    candidate = root/relative
    for current in [candidate, *candidate.parents]:
        if current == root:
            break
        if is_link(current):
            raise ValueError('更新先にリンクされたファイルやフォルダが含まれています。')
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('更新先に不正なパスが含まれています。')
    return resolved


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.kaiko-update-', dir=destination.parent)
    os.close(handle)
    try:
        shutil.copyfile(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def install_update(source, target, *, layout="legacy", allow_new=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('更新ZIP側ではなく、これまで使っていたapp.pyを選んでください。')
    if layout not in ('legacy', 'one_click'):
        raise ValueError('不明なインストール形式です。')
    app_name = '_internal/app.py' if layout == 'one_click' else 'app.py'
    manifest_name = '_internal/update_manifest.json' if layout == 'one_click' else 'update_manifest.json'
    if layout == 'legacy':
        if not (target/app_name).is_file() or not (target/'.venv/Scripts/python.exe').is_file():
            raise ValueError('今までのapp.pyと.venvがあるフォルダを選んでください。')
    elif not allow_new and not (target/app_name).is_file():
        raise ValueError('インストール済みのアプリがありません。')
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(check_path(source, Path(manifest_name)).read_text(encoding='utf-8-sig'))
    plans = []
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError('更新内容が空です。')
    seen = set()
    for name, expected in files.items():
        relative = safe_relative(name)
        if layout == 'one_click' and relative.as_posix().casefold() == manifest_name.casefold():
            raise ValueError('マニフェストは自分自身を登録できません。')
        identity = '/'.join(relative.parts).casefold()
        if identity in seen or not isinstance(expected, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected):
            raise ValueError('更新内容に重複した名前または不正なハッシュがあります。')
        seen.add(identity)
        expected = expected.lower()
        src, dst = check_path(source, relative), check_path(target, relative)
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise ValueError(f'更新ファイルが不足・破損しています。ZIPを再展開してください：{relative.name}')
        if dst.exists() and not dst.is_file():
            raise ValueError(f'更新先がファイルではありません：{relative.name}')
        for parent in dst.parents:
            if parent == target:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError('更新先のフォルダを確認してください。')
        if layout == 'one_click' and relative.suffix.casefold() == '.bat' and dst.exists():
            # cmd.exe may still be reading this file. Its forwarding shim is
            # immutable; update the PowerShell/Python helpers instead.
            if hashlib.sha256(dst.read_bytes()).hexdigest() != expected:
                raise ValueError('起動バッチの互換性を確認してください。')
            continue
        plans.append((relative, src, dst, dst.exists(), expected))
    if not plans or not any(p[0].as_posix() == app_name for p in plans):
        raise ValueError('更新内容が空です。')
    if layout == 'one_click':
        relative = Path(manifest_name)
        src, dst = check_path(source, relative), check_path(target, relative)
        if dst.exists() and not dst.is_file():
            raise ValueError('更新先マニフェストを確認してください。')
        plans.append((relative, src, dst, dst.exists(), hashlib.sha256(src.read_bytes()).hexdigest()))
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('self_model_%Y%m%d_%H%M%S_'),
                                   dir=_backup_root(target)))
    # Back up every replaced file before changing any application files.
    record = {'status': 'prepared', 'target': str(target), 'files': []}
    original_hashes = {}
    for relative, _, dst, exists, expected in plans:
        if exists:
            saved = backup/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, saved)
            if hashlib.sha256(saved.read_bytes()).hexdigest() != hashlib.sha256(dst.read_bytes()).hexdigest():
                raise OSError('バックアップの検査に失敗しました。アプリのファイルは変更していません。')
            original_hashes[relative] = hashlib.sha256(saved.read_bytes()).hexdigest()
        record['files'].append({'path': relative.as_posix(), 'existed': exists, 'update_sha256': expected,
                                'original_sha256': original_hashes.get(relative)})
    details = backup/'_kaiko_backup'
    details.mkdir()
    (details/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    touched = []
    try:
        for relative, src, dst, exists, expected in plans:
            touched.append((relative, dst, exists))
            atomic_copy(src, dst)
            if hashlib.sha256(dst.read_bytes()).hexdigest() != expected:
                raise OSError(f'コピー後の検査に失敗しました：{relative.name}')
        record['status'] = 'complete'
        (details/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as original_error:
        failed = []
        for relative, dst, exists in reversed(touched):
            try:
                if exists:
                    if hashlib.sha256((backup/relative).read_bytes()).hexdigest() != original_hashes[relative]:
                        raise OSError('元のバックアップのハッシュが一致しません。')
                    atomic_copy(backup/relative, dst)
                    if hashlib.sha256(dst.read_bytes()).hexdigest() != original_hashes[relative]:
                        raise OSError('復元後のハッシュが元のファイルと一致しません。')
                else:
                    dst.unlink(missing_ok=True)
            except Exception as restore_error:
                failed.append({'path': relative.as_posix(), 'error': str(restore_error)})
        record.update(status='rollback_incomplete' if failed else 'rolled_back', restore_errors=failed)
        try:
            (details/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass  # Restoration must continue even when the disk cannot hold a log.
        if failed:
            raise OSError('更新に失敗し、一部のファイルを元に戻せませんでした。'
                          f'バックアップは残しています：{backup}\n'
                          'アプリを閉じて、上記バックアップ内の元ファイルを同じ相対位置へ戻してください。'
                          ' _kaiko_backup は記録用なのでコピー不要です。\n'
                          f'復元できなかったファイル：{", ".join(f["path"] for f in failed)}') from original_error
        raise
    return backup


def _backup_root(target):
    path = target/'backups'
    if is_link(path):
        raise ValueError('バックアップ先がリンクになっています。')
    path.mkdir(exist_ok=True)
    return path


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    root.title('蚕CHANNEL 自作モデルへ更新')
    root.geometry('650x220')
    panel = ttk.Frame(root, padding=20); panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='今のアプリを閉じてから、普段使っている app.py を選んでください。', wraplength=600).pack(anchor='w')
    ttk.Label(panel, text='元のプログラムはバックアップします。計測記録・Python環境は更新対象に含めません。', wraplength=600).pack(anchor='w', pady=10)
    status = tk.StringVar(value='選択したフォルダに更新し、自作モデルの画面を起動します。')

    def choose():
        name = filedialog.askopenfilename(title='今まで使っていた app.py を選ぶ', filetypes=[('app.py', 'app.py')])
        if not name:
            return
        button.configure(state='disabled')
        root.update_idletasks()
        try:
            target = Path(name).parent
            backup = install_update(Path(__file__).parent, target)
        except Exception as exc:
            button.configure(state='normal')
            messagebox.showerror('更新できませんでした', str(exc), parent=root)
            return
        try:
            subprocess.Popen([str(target/'.venv/Scripts/python.exe'), str(target/'start_scratch_app.py')],
                             cwd=target, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
        except OSError as exc:
            button.configure(state='normal')
            status.set(f'更新は完了しました。バックアップ：{backup.name}')
            messagebox.showerror('更新済み・起動できませんでした',
                                 '更新したファイルは保存されています。元のフォルダのstart_scratch_windows.batから起動してください。'
                                 f'\nバックアップ：{backup}\n{exc}', parent=root)
            return
        status.set(f'更新しました。元のファイルの保存先：{backup.name}')
        root.after(1000, root.destroy)

    button = ttk.Button(panel, text='今までの app.py を選んで更新・起動', command=choose)
    button.pack(anchor='w', pady=8)
    ttk.Label(panel, textvariable=status, wraplength=600).pack(anchor='w')
    root.mainloop()


if __name__ == '__main__':
    main()


def restore_update(backup, target):
    """Restore a completed update after dependency/launch verification failed."""
    backup, target = Path(backup).resolve(), Path(target).resolve()
    if not backup.is_relative_to((target/'backups').resolve()):
        raise ValueError('バックアップの場所を確認してください。')
    record_path = backup/'_kaiko_backup/record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if Path(record['target']).resolve() != target:
        raise ValueError('バックアップの対象が違います。')
    errors = []
    for item in reversed(record['files']):
        try:
            relative = safe_relative(item['path'])
            dst = check_path(target, relative)
            if item['existed']:
                src = check_path(backup, relative)
                if hashlib.sha256(src.read_bytes()).hexdigest() != item['original_sha256']:
                    raise ValueError('バックアップのハッシュが違います。')
                atomic_copy(src, dst)
            else:
                dst.unlink(missing_ok=True)
        except Exception as exc:
            errors.append({'path': item['path'], 'error': str(exc)})
    record.update(status='rollback_incomplete' if errors else 'rolled_back', restore_errors=errors)
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:
        raise OSError('元に戻せないファイルがあります。バックアップは保持しています。')
