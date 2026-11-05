"""Developer packaging only. End users open the single included batch file."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from secure_updates import unpack_release

CODE = ['app.py','measurement.py','ai_measurement.py','scratch_measurement.py','silkworm_scratch.py','silkworm_texture.py','start_scratch_app.py','check_environment.py','prepare_training_images.py','collect_youtube_samples.py','bootstrap.py','bootstrap_windows.ps1','instance_lock.py','secure_updates.py','run_app.py','update_and_start.py','requirements.txt','runtime.json','update_channel.json','update_public_key.json','ONE_CLICK.md','LICENSE']
MODELS = ['kaiko_scratch','kaiko_texture','kaiko_texture_v6']


def build(destination, models, *, examples=None, release_sequence, version, public_update=False, school=False):
    if type(release_sequence) is not int or release_sequence<1: raise ValueError('Positive release sequence required')
    source = Path(__file__).resolve().parent
    destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage = Path(folder); (stage/'_internal/models').mkdir(parents=True)
        # This immutable cmd forwarding shim must stay identical across Windows
        # and Unix Git checkouts; never rewrite a running batch during an update.
        bat = (source/'蚕チェック.bat').read_text(encoding='ascii').replace('\r\n','\n').replace('\n','\r\n')
        (stage/'蚕チェック.bat').write_bytes(bat.encode('ascii'))
        (stage/'はじめに.txt').write_text('ZIPをすべて展開して、蚕チェック.batを開きます。\n初回の準備と次回の起動は同じファイルです。\n自動更新は配信開始後に利用できます。認識モデルの自動体長・頭数は未完成です。\n',encoding='utf-8-sig')
        for name in CODE: shutil.copyfile(source/name,stage/'_internal'/name)
        if school:
            # Keep the immutable BAT identical; select the school launcher underneath it.
            launcher = (source/'bootstrap_school_windows.ps1').read_text(encoding='utf-8-sig')
            (stage/'_internal/bootstrap_windows.ps1').write_text(launcher,encoding='utf-8-sig')
            (stage/'_internal/ONE_CLICK.md').write_text((source/'SCHOOL.md').read_text(encoding='utf-8'),encoding='utf-8')
            (stage/'_internal/update_channel.json').write_text(json.dumps({'schema':1,'feed_url':''}),encoding='utf-8')
            (stage/'はじめに.txt').write_text('蚕CHANNEL 体長チェック（学校用）\n\nZIPをすべて展開して、蚕チェック.batを開きます。\n使用するPython: X:\\python 3.12\\python.exe\n初回はネット接続が必要です。専用のライブラリを自分のユーザーフォルダへ準備します。\n共有Pythonの変更や別のPythonのインストールは行いません。\n次回も同じBAT、またはデスクトップの「蚕CHANNEL 体長チェック（学校用）」から起動します。\n学校版の自動更新配信は未有効です。認識モデルの自動体長・頭数は未完成です。\n',encoding='utf-8-sig')
        for name in MODELS:
            for suffix in ('.pt','.json'): shutil.copyfile(Path(models)/(name+suffix),stage/'_internal/models'/(name+suffix))
        if examples and not public_update:
            (stage/'_internal/examples').mkdir()
            for name in ('startup.png','dark_instar1.png','prior_instar5.png','instar5.png'):
                p=Path(examples)/name
                if p.is_file(): shutil.copyfile(p,stage/'_internal/examples'/name)
        manifest={'schema_version':2,'layout':'one_click','release_sequence':release_sequence,'version':version,
                  'installation_flavor':'school' if school else 'desktop',
                  'auto_update_channel':'not-configured' if school else 'signed-feed','files':{p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}}
        (stage/'_internal/update_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file(): z.write(p,p.relative_to(stage).as_posix())
    if public_update:
        raw=destination.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            unpack_release(raw,{'release_sequence':release_sequence,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},Path(folder)/'verified')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--examples');parser.add_argument('--release-sequence',required=True,type=int)
    parser.add_argument('--version',required=True);parser.add_argument('--public-update',action='store_true')
    parser.add_argument('--school',action='store_true',help='Use the fixed school Python without installing Python')
    a=parser.parse_args();build(a.output,a.models,examples=a.examples,release_sequence=a.release_sequence,version=a.version,public_update=a.public_update,school=a.school)
