"""One entrypoint for per-user setup, signed updates, and application launch."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))

from instance_lock import FileLock
from secure_updates import read_package, sequence, verify_envelope, download, download_package, unpack_release, MAX_ARCHIVE
from update_and_start import atomic_copy, check_path, install_update, is_link, restore_update


class SetupError(RuntimeError): pass


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    handle,name = tempfile.mkstemp(prefix='.kaiko-state-',dir=path.parent)
    try:
        with os.fdopen(handle,'w',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2)
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_state(root):
    path = Path(root)/'_internal/data/install_state.json'
    if not path.is_file(): return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict): raise SetupError('起動設定が壊れています。')
    if data.get('active_environment') is not None:
        environment_path(root,data['active_environment'])
    return data


def environment_path(root, value):
    if not isinstance(value,str) or not re.fullmatch(r'runtime/venvs/[a-f0-9]{20}',value):
        raise SetupError('Python環境の保存先が不正です。')
    return check_path(Path(root).resolve(),Path(value))


def command(args, log, timeout=None):
    log('準備: '+str(args[1:5]))
    if timeout is not None:
        result = subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,check=False)
        if result.stdout: log(result.stdout.rstrip())
        if result.stderr: log(result.stderr.rstrip())
        if result.returncode: raise SetupError('準備のコマンドに失敗しました。ログを確認してください。')
        return result.stdout
    with subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
        for line in process.stdout: log(line.rstrip())
        if process.wait(): raise SetupError('必要なライブラリの準備に失敗しました。ネット接続とログを確認してください。')
    return ''


def python_in(environment, windowless=False):
    if os.name=='nt': return Path(environment)/'Scripts'/('pythonw.exe' if windowless else 'python.exe')
    return Path(environment)/'bin/python'


def dependencies(root):
    text = (Path(root)/'_internal/requirements.txt').read_text(encoding='utf-8')
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'): continue
        match = re.fullmatch(r'([A-Za-z0-9_-]+)==([A-Za-z0-9.+_-]+)',line.strip())
        if not match: raise SetupError('依存ライブラリの形式が不正です。')
        result[match[1]] = match[2]
    runtime_path = Path(root)/'_internal/runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {'python':[3,12],'torch_version':'2.8.0'}
    if runtime.get('python') != [3,12] or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+',runtime.get('torch_version','')):
        raise SetupError('この起動ファイルに対応していないPython環境です。')
    result['torch'] = runtime['torch_version']
    return text,result


def versions_ok(environment, expected):
    interpreter = python_in(environment)
    if not interpreter.is_file(): return False
    code = "import importlib.metadata as m,json,sys; assert sys.version_info[:2]==(3,12); print(json.dumps({k:m.version(k).split('+')[0] for k in json.loads(sys.argv[1])}))"
    try:
        r = subprocess.run([str(interpreter),'-I','-X','utf8','-c',code,json.dumps(list(expected))],capture_output=True,text=True,encoding='utf-8',timeout=15)
        return r.returncode==0 and json.loads(r.stdout.strip())==expected
    except (OSError,ValueError,subprocess.TimeoutExpired): return False


def ensure_environment(package, root, log):
    requirements,expected = dependencies(package)
    base = getattr(sys,'_base_executable',sys.executable)
    key = hashlib.sha256((requirements+'\ntorch='+expected['torch']+'\n'+os.path.normcase(str(Path(base).resolve()))).encode()).hexdigest()[:20]
    environment = environment_path(root,'runtime/venvs/'+key)
    if versions_ok(environment,expected): return environment
    if environment.exists():
        retired = Path(root)/'runtime/retired'; retired.mkdir(parents=True,exist_ok=True)
        environment.rename(retired/(key+'_'+uuid.uuid4().hex[:8]))
    log('初回の準備です。必要なライブラリを自動で入れています。数分かかる場合があります。')
    environment.parent.mkdir(parents=True,exist_ok=True)
    command([base,'-I','-X','utf8','-m','venv',str(environment)],log)
    interpreter = str(python_in(environment))
    pip = [interpreter,'-I','-X','utf8','-m','pip','--isolated','--disable-pip-version-check']
    if read_package(package).get('storage_mode') == 'portable':
        pip += ['--cache-dir',str(Path(root)/'runtime/pip-cache')]
    command(pip+['install','--index-url','https://pypi.org/simple','--only-binary=:all:','--progress-bar','off','-r',str(Path(package)/'_internal/requirements.txt')],log)
    command(pip+['install','--index-url','https://download.pytorch.org/whl/cpu','--only-binary=:all:','--progress-bar','off','torch=='+expected['torch']+'+cpu'],log)
    if not versions_ok(environment,expected): raise SetupError('ライブラリの確認に失敗しました。')
    return environment


def preflight(package, environment, log, install_root=None):
    names = ['kaiko_scratch.pt','kaiko_texture.pt','kaiko_texture_v6.pt']
    if any(not (Path(package)/'_internal/models'/name).is_file() for name in names):
        raise SetupError('必要な自作モデルが不足しています。')
    code = "import sys;sys.path.insert(0,sys.argv[1]);import cv2,numpy as np,scipy,skimage,PIL,tkinter,torch;torch.set_num_threads(3);from silkworm_scratch import load_model,predict;from pathlib import Path;models=[load_model(Path(sys.argv[1])/'models'/name)[0] for name in ['kaiko_scratch.pt','kaiko_texture.pt','kaiko_texture_v6.pt']];outputs=[predict(m,np.zeros((32,48,3),np.uint8)) for m in models];assert all(p.shape==(32,48) and np.isfinite(p).all() for p in outputs);print('ライブラリと自作モデルを確認しました。')"
    command([str(python_in(environment)),'-I','-X','utf8','-c',code,str(Path(package)/'_internal')],log,timeout=60)
    if (Path(package)/'_internal/instance_detector.py').is_file():
        # Large, pinned model assets live in the installed folder and survive
        # program updates. Neither model files nor caches use AppData.
        destination=Path(install_root or package)/'_internal/models/silkworm_dsri.pt'
        asset_code="import sys;sys.path.insert(0,sys.argv[1]);from instance_detector import ensure_instance_model,load_detector;import torch;torch.set_num_threads(3);p=ensure_instance_model(sys.argv[2]);m=load_detector(p);print('蚕専用モデルを確認しました。')"
        log('無料の蚕専用モデルを準備しています（初回 約7MB）。')
        command([str(python_in(environment)),'-I','-X','utf8','-c',asset_code,str(Path(package)/'_internal'),str(destination)],log,timeout=120)


def legacy_sources(profile, install_root):
    candidates = []
    profile = Path(profile)
    for place in ['Downloads','Desktop','Documents']:
        base = profile/place
        if not base.is_dir(): continue
        dirs = [p for p in base.iterdir() if p.is_dir() and p.name.casefold().startswith(('kaiko','蚕')) and not is_link(p)]
        frontier = [(p,0) for p in dirs]
        visited = 0
        while frontier and visited<300:
            folder,depth = frontier.pop(); visited+=1
            if folder.resolve().is_relative_to(Path(install_root).resolve()): continue
            app,data = folder/'app.py',folder/'data'
            if not app.is_file() and (folder/'_internal/app.py').is_file():
                app = folder/'_internal/app.py'
            try:
                if app.is_file() and data.is_dir() and not is_link(data) and '蚕CHANNEL 体長チェック' in app.read_text(encoding='utf-8-sig'):
                    record = data/'measurements.csv'
                    stamp = record.stat().st_mtime if record.is_file() else data.stat().st_mtime
                    candidates.append((stamp,data))
            except (OSError,UnicodeError): pass
            if depth<2:
                try: frontier.extend((p,depth+1) for p in folder.iterdir() if p.is_dir() and not is_link(p) and p.name not in {'.venv','data','models','backups'})
                except OSError: pass
    return [path for _,path in sorted(candidates,key=lambda x:x[0],reverse=True)]


def migrate_data(source, target):
    source,target = Path(source),Path(target)
    copied = 0
    for directory,dirs,files in os.walk(source,followlinks=False):
        parent = Path(directory)
        dirs[:] = [d for d in dirs if not is_link(parent/d)]
        for name in files:
            src = parent/name
            if is_link(src) or name in {'.app.lock','.setup.lock','install_state.json','app_state.json'} or name.startswith('.launch_'): continue
            dst = target/src.relative_to(source)
            if not dst.exists():
                atomic_copy(src,dst); copied+=1
    return copied


def legacy_school_sources(profile, install_root, flavor='school'):
    """Read the previous school data once; never install into LocalAppData."""
    local = Path(os.environ.get('LOCALAPPDATA',str(Path(profile)/'AppData/Local')))
    name = 'SizeCheckSchool' if flavor == 'school' else 'SizeCheck'
    folder = local/'KaikoChannel'/name/'_internal'
    data = folder/'data'
    try:
        if (not folder.resolve().is_relative_to(Path(install_root).resolve())
                and not is_link(folder) and not is_link(data)
                and data.is_dir() and (folder/'app.py').is_file()
                and '蚕CHANNEL 体長チェック' in (folder/'app.py').read_text(encoding='utf-8-sig')):
            return [data]
    except (OSError,UnicodeError): pass
    return []


def choose_online(package, current_sequence, root, log, fetch=download):
    internal = Path(package)/'_internal'
    config = json.loads((internal/'update_channel.json').read_text(encoding='utf-8'))
    if not config.get('feed_url'): return None
    try:
        key = json.loads((internal/'update_public_key.json').read_text(encoding='utf-8'))
        info = verify_envelope(fetch(config['feed_url'],65536,timeout=4),key)
        if config.get('channel') and info.get('channel') != config['channel']:
            raise SetupError('更新先のブランチが違います。')
        if sequence(info)<=current_sequence: return None
        log('新しい版が見つかりました。更新ファイルを確認しています。')
        raw = download_package(info,fetch=fetch)
        staging_parent = Path(root)/'updates'; staging_parent.mkdir(parents=True,exist_ok=True)
        staging = staging_parent/('package_'+uuid.uuid4().hex)
        try:
            unpack_release(raw,info,staging)
            if package_flavor(read_package(staging)) != package_flavor(read_package(package)):
                raise SetupError('学校用と通常版の更新ファイルが異なります。')
        except Exception:
            shutil.rmtree(staging,ignore_errors=True)
            raise
        return staging
    except Exception as exc:
        log('最新版を確認できないため、手元の版で起動します。('+type(exc).__name__+')')
        return None


def package_flavor(manifest):
    flavor = manifest.get('installation_flavor','desktop')
    if flavor not in ('desktop','school'):
        raise SetupError('対応していない起動版です。')
    return flavor


def select_package(source, installed):
    source_info = read_package(source)
    flavor = package_flavor(source_info)
    try: installed_info = read_package(installed)
    except (OSError,ValueError,KeyError): installed_info = None
    if installed_info and package_flavor(installed_info) != flavor:
        raise SetupError('学校用と通常版の保存先は分けてください。')
    if installed_info and sequence(installed_info)>=sequence(source_info): return Path(installed),installed_info
    return Path(source),source_info


def activate(package, root, environment, previous_state):
    package,root = Path(package).resolve(),Path(root).resolve()
    backup = None
    manifest = read_package(package)
    new_state = {**previous_state,'installed_sequence':sequence(manifest),
                 'active_environment':environment.relative_to(root).as_posix()}
    try:
        if package!=root:
            backup = install_update(package,root,layout='one_click',allow_new=True)
        atomic_json(root/'_internal/data/install_state.json',new_state)
    except Exception:
        if backup: restore_update(backup,root)
        raise
    return backup,new_state


def focus_existing(root):
    if os.name!='nt': return
    try:
        import ctypes
        from ctypes import wintypes
        state = json.loads((Path(root)/'_internal/data/app_state.json').read_text())
        pid = state['pid']
        if type(pid) is not int or not 1<=pid<=0xffffffff: return
        user = ctypes.windll.user32
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
        user.EnumWindows.argtypes=[callback_type,wintypes.LPARAM]
        user.EnumWindows.restype=wintypes.BOOL
        user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
        user.GetWindowThreadProcessId.restype=wintypes.DWORD
        user.IsWindowVisible.argtypes=[wintypes.HWND];user.IsWindowVisible.restype=wintypes.BOOL
        user.ShowWindow.argtypes=[wintypes.HWND,ctypes.c_int]
        user.SetForegroundWindow.argtypes=[wintypes.HWND]
        def callback(hwnd,_):
            process = wintypes.DWORD(); user.GetWindowThreadProcessId(hwnd,ctypes.byref(process))
            if process.value==pid and user.IsWindowVisible(hwnd):
                user.ShowWindow(hwnd,9); user.SetForegroundWindow(hwnd)
            return True
        user.EnumWindows(callback_type(callback),0)
    except Exception: pass


def start_gui(root, environment, log, timeout=35):
    root,environment = Path(root),Path(environment)
    token = uuid.uuid4().hex
    ready = root/'_internal/data'/('.launch_'+token+'.json')
    env = {**os.environ,'KAIKO_ONE_CLICK':'1','KAIKO_VENV_PATH':str(environment)}
    if read_package(root).get('storage_mode') == 'portable':
        env['KAIKO_STORAGE_ROOT'] = str(root.resolve())
    process = subprocess.Popen([str(python_in(environment,windowless=True)),'-I','-X','utf8',str(root/'_internal/run_app.py'),'--token',token],cwd=root,env=env,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if ready.is_file():
            result = json.loads(ready.read_text(encoding='utf-8')); ready.unlink(missing_ok=True)
            if result.get('status')=='ready': return process
            raise SetupError('アプリの起動確認に失敗しました。')
        if process.poll() is not None: raise SetupError('アプリを起動できませんでした。')
        time.sleep(.1)
    # Do not kill a slow GUI or overwrite files underneath it.
    raise SetupError('起動に時間がかかっています。画面とlogs内の記録を確認してください。')


def run(source, root, log, profile=None, allow_online=True):
    source,root = Path(source).resolve(),Path(root).absolute()
    root.mkdir(parents=True,exist_ok=True)
    if is_link(root) or is_link(root/'_internal') or is_link(root/'runtime'):
        raise SetupError('保存先にリンクされたフォルダがあります。')
    root = root.resolve()
    check_path(root,Path('_internal/data'))
    setup_lock = FileLock(root/'_internal/data/.setup.lock')
    if not setup_lock.acquire(): log('別の画面で起動の準備が進んでいます。'); return 0
    staging = None
    batch_lock = FileLock(root/'_internal/data/.batch.lock')
    try:
        package,manifest = select_package(source,root)
        # A stale downloaded launcher hands over to the installed newer one.
        if package==root and source!=root and sequence(manifest)>sequence(read_package(source)):
            setup_lock.release()
            return subprocess.call([sys.executable,'-I','-X','utf8',str(root/'_internal/bootstrap.py'),'--package-root',str(root),'--install-root',str(root),'--no-delegate'],env=os.environ.copy())
        previous = load_state(root)
        if allow_online:
            staging = choose_online(package,max(sequence(manifest),previous.get('blocked_release_sequence',0)),root,log)
            if staging: package,manifest = staging,read_package(staging)
        if package!=root and not batch_lock.acquire():
            if not (root/'_internal/update_manifest.json').is_file():
                raise SetupError('自動収集を停止してから、セットアップを再実行してください。')
            package,manifest=root,read_package(root)
            log('自動収集・解析が動いているため更新を次回へ見送ります。準備済みの版を起動します。')
        app_lock = FileLock(root/'_internal/data/.app.lock')
        if not app_lock.acquire():
            if package==root:
                focus_existing(root); log('すでに起動しています。'); return 0
            log('更新のため、開いている蚕チェックの画面を閉じてください。閉じると自動で続きます。')
            while not app_lock.acquire(): time.sleep(.5)
        try:
            environment = ensure_environment(package,root,log)
            if package!=root or previous.get('active_environment')!=environment.relative_to(root).as_posix():
                preflight(package,environment,log,install_root=root)
            backup,state = activate(package,root,environment,previous)
            if not state.get('legacy_import_checked'):
                sources = legacy_sources(profile or Path.home(),root)
                portable = manifest.get('storage_mode') == 'portable'
                if portable:
                    sources = legacy_school_sources(profile or Path.home(),root,package_flavor(manifest)) + sources
                if sources:
                    try:
                        count = migrate_data(sources[0],root/('data' if portable else '_internal/data'))
                        log(f'以前の記録を{count}ファイルコピーしました。元のフォルダも残しています。')
                    except OSError as exc: log('以前の記録のコピーは完了していません。元の記録は残っています。('+str(exc)+')')
                state['legacy_import_checked']=True
                atomic_json(root/'_internal/data/install_state.json',state)
        except Exception:
            if previous.get('active_environment') and package!=root:
                log('更新を完了できなかったため、今までの版を起動します。')
                current = environment_path(root,previous['active_environment'])
                valid = read_package(root)
                if sequence(valid)!=previous.get('installed_sequence'): raise SetupError('元の版の確認に失敗しました。')
                if versions_ok(current,dependencies(root)[1]):
                    app_lock.release(); start_gui(root,current,log); return 0
            raise
        finally: app_lock.release()
        log('セットアップ完了。蚕チェックを起動します。')
        try:
            start_gui(root,environment,log)
        except SetupError:
            # Roll back only after the failed GUI has exited and released its lock.
            lock = FileLock(root/'_internal/data/.app.lock')
            if backup and previous.get('active_environment') and lock.acquire():
                try:
                    restore_update(backup,root)
                    atomic_json(root/'_internal/data/install_state.json',{**previous,'blocked_release_sequence':sequence(manifest)})
                finally: lock.release()
                log('新しい版を起動できなかったため、以前の版へ戻しました。')
                start_gui(root,environment_path(root,previous['active_environment']),log)
            else: raise
        return 0
    finally:
        batch_lock.release()
        setup_lock.release()
        if staging: shutil.rmtree(staging,ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root',required=True); parser.add_argument('--install-root',required=True)
    parser.add_argument('--log-file'); parser.add_argument('--no-delegate',action='store_true')
    args = parser.parse_args()
    if args.log_file:
        logpath=Path(args.log_file)
    else:
        logpath=Path(args.install_root)/'logs'/('setup_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt')
    logpath.parent.mkdir(parents=True,exist_ok=True)
    def log(message):
        print(message,flush=True)
        with logpath.open('a',encoding='utf-8') as f: f.write(message+'\n')
    try: return run(Path(args.package_root),Path(args.install_root),log)
    except Exception as exc:
        log('準備を完了できませんでした。'+str(exc)); log('記録の保存先: '+str(logpath)); return 1


if __name__=='__main__': sys.exit(main())
