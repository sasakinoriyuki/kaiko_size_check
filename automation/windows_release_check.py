"""Run the existing signed BAT on a free, disposable Windows CI runner.

No user photos, measurements, credentials or uploaded artifacts are used.
The school X: mapping and junction are created only on this disposable runner.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from secure_updates import download,verify_envelope


def close_gui(pid,expected_version):
    from ctypes import wintypes as w
    user=ctypes.WinDLL('user32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    callback_type=ctypes.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM)
    user.GetWindowThreadProcessId.argtypes=[w.HWND,ctypes.POINTER(w.DWORD)]
    user.GetWindowTextW.argtypes=[w.HWND,w.LPWSTR,ctypes.c_int]
    user.PostMessageW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM]
    user.EnumWindows.argtypes=[callback_type,w.LPARAM]
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];kernel.WaitForSingleObject.restype=w.DWORD
    kernel.CloseHandle.argtypes=[w.HANDLE]
    matches=[]
    @callback_type
    def visit(hwnd,_):
        process=w.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(process))
        if process.value==pid:
            title=ctypes.create_unicode_buffer(512);user.GetWindowTextW(hwnd,title,512)
            if expected_version in title.value:matches.append(hwnd)
        return True
    user.EnumWindows(visit,0)
    assert matches,'The launched process has no release-labelled GUI window'
    handle=kernel.OpenProcess(0x00100000,False,pid)
    assert handle,'Cannot monitor the launched GUI'
    try:
        for hwnd in matches:user.PostMessageW(hwnd,0x0010,0,0)
        assert kernel.WaitForSingleObject(handle,45000)==0,'GUI did not close cleanly'
    finally:kernel.CloseHandle(handle)


def run(channel):
    assert os.name=='nt','Windows CI only'
    assert sys.version_info[:2]==(3,12)
    base=Path(os.environ['RUNNER_TEMP'])/'kaiko verification'
    base.mkdir(exist_ok=True)
    root=base/(channel+' 蚕 & 日本語');root.mkdir()
    (root/'data').mkdir();marker=root/'data/keep.txt';marker.write_bytes(b'synthetic retention fixture')
    env=os.environ.copy()
    env['LOCALAPPDATA']=str(base/'empty-profile/AppData/Local')
    env['PYTHONUTF8']='1';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    prefix='https://raw.githubusercontent.com/sasakinoriyuki/kaiko_size_check/'+channel+'/'
    info=verify_envelope(download(prefix+'.updates/channel.json',65536),
                         json.loads((SOURCE/'update_public_key.json').read_text()))
    assert info['channel']==channel
    bat=root/'蚕チェック.bat';raw=download(prefix+quote(bat.name),65536);bat.write_bytes(raw)
    mapped=False;school_python=None
    try:
        if channel=='school':
            assert not Path('X:/').exists(),'X: already exists; no existing drive is replaced'
            school_drive=base/'school-drive';school_drive.mkdir()
            subprocess.run(['subst','X:',str(school_drive)],check=True);mapped=True
            school_python=Path('X:/python 3.12')
            command='mklink /J "'+str(school_python)+'" "'+sys.base_prefix+'"'
            subprocess.run('"'+os.environ['COMSPEC']+'" /d /s /c "'+command+'"',check=True,capture_output=True)
            assert (school_python/'python.exe').is_file()
        environments=[];processes=[]
        for iteration in range(2):
            log=root/('ci-launch-'+str(iteration+1)+'.log')
            with log.open('wb') as stream:
                completed=subprocess.run('"'+os.environ['COMSPEC']+'" /d /s /c ""'+str(bat)+'""',
                    cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            if completed.returncode:
                print(log.read_text(encoding='utf-8',errors='replace')[-5000:])
                raise AssertionError('BAT returned '+str(completed.returncode))
            manifest=json.loads((root/'_internal/update_manifest.json').read_text(encoding='utf-8'))
            assert manifest['version']==info['version']
            state=json.loads((root/'_internal/data/install_state.json').read_text(encoding='utf-8'))
            environments.append(state['active_environment'])
            app_state=json.loads((root/'_internal/data/app_state.json').read_text(encoding='utf-8'))
            close_gui(int(app_state['pid']),info['version']);processes.append(app_state['pid'])
            assert marker.read_bytes()==b'synthetic retention fixture'
            assert bat.read_bytes()==raw
        assert environments[0]==environments[1]
        assert len(list(root.glob('*.bat')))==1
        assert not (Path(env['LOCALAPPDATA'])/'KaikoChannel').exists()
        if channel=='school':
            logs='\n'.join(p.read_text(encoding='utf-8-sig',errors='replace') for p in (root/'logs').glob('setup_*.txt'))
            assert str(school_python/'python.exe') in logs
        interpreter=root/environments[-1]/'Scripts/python.exe'
        modules=['test_measurement','test_instance_geometry','test_instance_counting',
                 'test_visible_measurement','test_app','test_ai_measurement',
                 'test_instance_detector','test_archive_batch','test_archive_review']
        check_code=("import sys,unittest;sys.path.insert(0,sys.argv[1]);"
                    "suite=unittest.defaultTestLoader.loadTestsFromNames("+repr(modules)+
                    ");result=unittest.TextTestRunner(verbosity=1).run(suite);"
                    "raise SystemExit(0 if result.wasSuccessful() else 1)")
        subprocess.run([str(interpreter),'-I','-X','utf8','-c',check_code,str(SOURCE)],
                       cwd=root,env=env,check=True,timeout=180)
        subprocess.run([str(interpreter),'-I','-X','utf8',str(SOURCE/'automation/runtime_gui_check.py'),
                        '--installed-root',str(root),'--expected-version',info['version'],'--rounds','3'],
                       cwd=root,env=env,check=True,timeout=240)
        report={'channel':channel,'release':info['version'],'signed_bat_fresh_and_repeat_start':True,
                'same_runtime_reused':True,'old_data_preserved':True,'one_unchanged_bat':True,
                'portable_storage':True,'school_fixed_python_executed':channel=='school',
                'windows_ci':True,'core_regression_modules_passed':modules,
                'user_photos_or_records_uploaded':False}
        print(json.dumps(report),flush=True)
        return report
    finally:
        if school_python is not None and school_python.exists():
            # Remove only this disposable junction, never its Python target.
            os.rmdir(school_python)
        if mapped:subprocess.run(['subst','X:','/D'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--channel',choices=['school','parsonal'],required=True)
    run(p.parse_args().channel)
