"""Install a local update into the user's selected app folder, with rollback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.kaiko-update-', dir=destination.parent)
    os.close(handle)
    try:
        shutil.copyfile(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def install_update(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('更新ZIP側ではなく、これまで使っていたapp.pyを選んでください。')
    if not (target/'app.py').is_file() or not (target/'.venv/Scripts/python.exe').is_file():
        raise ValueError('今までのapp.pyと.venvがあるフォルダを選んでください。')
    manifest = json.loads((source/'update_manifest.json').read_text(encoding='utf-8'))
    plans = []
    for name, expected in manifest['files'].items():
        relative = Path(name)
        src, dst = (source/relative).resolve(), (target/relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts
                or relative.parts[0] in {'.venv', 'data', 'output', 'backups'}
                or not src.is_relative_to(source) or not dst.is_relative_to(target)):
            raise ValueError('更新先に不正なパスが含まれています。')
        unresolved = target/relative
        for current in [unresolved, *unresolved.parents]:
            if current == target:
                break
            if current.is_symlink():
                raise ValueError('更新先にリンクされたファイルやフォルダが含まれています。')
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise ValueError(f'更新ファイルが不足・破損しています。ZIPを再展開してください：{relative.name}')
        if dst.exists() and not dst.is_file():
            raise ValueError(f'更新先がファイルではありません：{relative.name}')
        for parent in dst.parents:
            if parent == target:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError('更新先のフォルダを確認してください。')
        plans.append((relative, src, dst, dst.exists()))
    if not plans or not any(str(p[0]) == 'app.py' for p in plans):
        raise ValueError('更新内容が空です。')
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('self_model_%Y%m%d_%H%M%S_'),
                                   dir=_backup_root(target)))
    # Back up every replaced file before changing any application files.
    for relative, _, dst, exists in plans:
        if exists:
            saved = backup/relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, saved)
    touched = []
    try:
        for relative, src, dst, exists in plans:
            touched.append((relative, dst, exists))
            atomic_copy(src, dst)
    except Exception:
        for relative, dst, exists in reversed(touched):
            if exists:
                atomic_copy(backup/relative, dst)
            else:
                dst.unlink(missing_ok=True)
        raise
    return backup


def _backup_root(target):
    path = target/'backups'
    if path.is_symlink():
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
            subprocess.Popen([str(target/'.venv/Scripts/python.exe'), str(target/'start_scratch_app.py')],
                             cwd=target, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
        except Exception as exc:
            button.configure(state='normal')
            messagebox.showerror('更新できませんでした', str(exc), parent=root)
            return
        status.set(f'更新しました。元のファイルの保存先：{backup.name}')
        root.after(1000, root.destroy)

    button = ttk.Button(panel, text='今までの app.py を選んで更新・起動', command=choose)
    button.pack(anchor='w', pady=8)
    ttk.Label(panel, textvariable=status, wraplength=600).pack(anchor='w')
    root.mainloop()


if __name__ == '__main__':
    main()
