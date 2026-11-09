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
CODE += ['instance_detector.py','instance_geometry.py','instance_counting.py','THIRD_PARTY.md','ACCURACY.md']
CODE += [p.relative_to(Path(__file__).parent).as_posix() for p in sorted((Path(__file__).parent/'kaiko_dsri').rglob('*'))
         if p.is_file() and p.suffix in {'.py','.md','.txt'}]
MODELS = ['kaiko_scratch','kaiko_texture','kaiko_texture_v6']


def build(destination, models, *, examples=None, release_sequence, version, public_update=False, school=False, online=False):
    if type(release_sequence) is not int or release_sequence<1: raise ValueError('Positive release sequence required')
    source = Path(__file__).resolve().parent
    destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage = Path(folder); (stage/'_internal/models').mkdir(parents=True)
        # This immutable cmd forwarding shim must stay identical across Windows
        # and Unix Git checkouts; never rewrite a running batch during an update.
        if not online:
            bat = (source/'蚕チェック.bat').read_text(encoding='utf-8').replace('\r\n','\n').replace('\n','\r\n')
            (stage/'蚕チェック.bat').write_bytes(bat.encode('utf-8'))
        (stage/'はじめに.txt').write_text('ZIPをすべて展開して、蚕チェック.batを開きます。\n初回の準備と次回の起動は同じファイルです。\n自動更新は配信開始後に利用できます。認識モデルの自動体長・頭数は未完成です。\n',encoding='utf-8-sig')
        for name in CODE:
            target=stage/'_internal'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/name,target)
        if school:
            # Keep the immutable BAT identical; select the school launcher underneath it.
            launcher = (source/'bootstrap_school_windows.ps1').read_text(encoding='utf-8-sig')
            (stage/'_internal/bootstrap_windows.ps1').write_text(launcher,encoding='utf-8-sig')
            (stage/'_internal/ONE_CLICK.md').write_text((source/'SCHOOL.md').read_text(encoding='utf-8'),encoding='utf-8')
            (stage/'_internal/update_channel.json').write_text(json.dumps({'schema':1,'feed_url':''}),encoding='utf-8')
            (stage/'はじめに.txt').write_text('蚕CHANNEL 体長チェック（学校用・フォルダ内保存）\n\n使いたいフォルダへZIPをすべて展開して、蚕チェック.batを開きます。\n使用するPython: X:\\python 3.12\\python.exe\n初回はネット接続が必要です。アプリ・専用ライブラリ・記録は展開したフォルダの中に保存します。\n検証用ZIPは data\\samples に入ります。「ZIP保存先を開く」から直接開けます。\n次回も、このフォルダの蚕チェック.batから起動します。\n学校版の自動更新配信は未有効です。認識モデルの自動体長・頭数は未完成です。\n',encoding='utf-8-sig')
        for name in MODELS:
            for suffix in ('.pt','.json'): shutil.copyfile(Path(models)/(name+suffix),stage/'_internal/models'/(name+suffix))
        if examples and not public_update:
            (stage/'_internal/examples').mkdir()
            for name in ('startup.png','dark_instar1.png','prior_instar5.png','instar5.png'):
                p=Path(examples)/name
                if p.is_file(): shutil.copyfile(p,stage/'_internal/examples'/name)
        if online:
            channel = 'school' if school else 'parsonal'
            config = {'schema':1,'channel':channel,'feed_url':f'https://raw.githubusercontent.com/sasakinoriyuki/kaiko_size_check/{channel}/.updates/channel.json'}
            (stage/'_internal/update_channel.json').write_text(json.dumps(config),encoding='utf-8')
            (stage/'はじめに.txt').write_text('起動は手元の蚕チェック.batです。プログラムと、その版に必要なPyTorch等を自動で準備・更新します。記録はdata、ライブラリはruntimeへ保存します。\n',encoding='utf-8-sig')
        manifest={'schema_version':2,'layout':'one_click','release_sequence':release_sequence,'version':version,
                  'installation_flavor':'school' if school else 'desktop',
                  'storage_mode':'portable' if school or online else 'per-user',
                  'auto_update_channel':'signed-github' if online else ('not-configured' if school else 'signed-feed'),
                  'files':{p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}}
        if online: manifest['entrypoint'] = 'external_launcher'
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
    parser.add_argument('--online',action='store_true',help='Build the payload for the standalone online BAT')
    a=parser.parse_args();build(a.output,a.models,examples=a.examples,release_sequence=a.release_sequence,version=a.version,public_update=a.public_update,school=a.school,online=a.online)
