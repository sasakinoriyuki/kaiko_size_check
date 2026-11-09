"""Prepare free collection tools inside the existing portable application folder."""
import hashlib
import importlib
import os
from pathlib import Path
import platform
import re
import sys
import zipfile
import io
import urllib.request
from urllib.parse import urlparse

from collect_youtube_samples import find_tool, _run


def asset_checksum(raw):
    # Windows Deno releases use PowerShell Get-FileHash output; Unix releases
    # use sha256sum. Accept one unambiguous SHA256 in either official format.
    found=re.findall(r'\b[0-9a-fA-F]{64}\b',raw.decode('utf-8-sig'))
    if len(found)!=1:raise ValueError('Denoのチェックサムを確認できません。')
    return found[0].lower()


def download_deno_asset(url,limit,timeout=30):
    if not url.startswith('https://github.com/denoland/deno/releases/latest/download/'):
        raise ValueError('Denoの公式配布URLではありません。')
    def checked(target):
        p=urlparse(target)
        if p.scheme!='https' or p.hostname not in {'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'} or p.username or p.password or p.port not in {None,443}:
            raise ValueError('Denoの取得先を確認できません。')
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,request,fp,code,message,headers,newurl):
            checked(newurl)
            return super().redirect_request(request,fp,code,message,headers,newurl)
    with urllib.request.build_opener(Redirect()).open(url,timeout=timeout) as response:
        checked(response.geturl());raw=response.read(limit+1)
    if len(raw)>limit:raise ValueError('Denoの取得サイズが大きすぎます。')
    return raw


def prepare_tools(storage_root,runner,log=lambda message:None,fetch=None):
    root=Path(storage_root).resolve()/'runtime/archive_tools';root.mkdir(parents=True,exist_ok=True)
    temporary=root/'tmp';temporary.mkdir(exist_ok=True)
    os.environ['TMP']=os.environ['TEMP']=os.environ['TMPDIR']=str(temporary)
    os.environ['KAIKO_ARCHIVE_TEMP']=str(temporary)
    for key,folder in [('DENO_DIR','deno_cache'),('PYTHONPYCACHEPREFIX','pycache'),('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','share'),('IMAGEIO_USERDIR','imageio')]:
        location=root/folder;location.mkdir(exist_ok=True);os.environ[key]=str(location)
    os.environ['PYTHONNOUSERSITE']='1'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['PATH']=str(root)+os.pathsep+os.environ.get('PATH','')
    importlib.invalidate_caches()
    try:yt=find_tool('yt-dlp',runner=runner)
    except RuntimeError:yt=None
    try:ffmpeg=find_tool('ffmpeg')
    except RuntimeError:ffmpeg=None
    if yt is None or ffmpeg is None:
        log('無料の取得ツールを、このアプリのruntimeフォルダに準備しています。')
        _run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
              '--target',str(root),'--upgrade','yt-dlp[default]','imageio-ffmpeg'],runner,600)
        importlib.invalidate_caches()
        if yt is None:yt=[sys.executable,'-m','yt_dlp']
        if ffmpeg is None:
            module=importlib.import_module('imageio_ffmpeg')
            ffmpeg=[module.get_ffmpeg_exe()]
    try:deno=find_tool('deno')
    except RuntimeError:
        fetch=fetch or download_deno_asset
        machine=platform.machine().lower()
        arch='aarch64' if machine in {'arm64','aarch64'} else 'x86_64'
        system={'win32':'pc-windows-msvc','linux':'unknown-linux-gnu','darwin':'apple-darwin'}.get(sys.platform)
        if not system:raise RuntimeError('このOSでの取得ツールの自動準備には未対応です。')
        filename=f'deno-{arch}-{system}.zip'
        url='https://github.com/denoland/deno/releases/latest/download/'+filename
        log('動画取得用の無料JavaScript実行環境をruntime内に準備しています。')
        checksum=asset_checksum(fetch(url+'.sha256sum',1024,timeout=30))
        raw=fetch(url,60*1024**2,timeout=180)
        if hashlib.sha256(raw).hexdigest()!=checksum.lower():raise ValueError('Denoの取得内容が一致しません。')
        name='deno.exe' if os.name=='nt' else 'deno'
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entry=archive.getinfo(name)
            if entry.file_size>250*1024**2:raise ValueError('Denoの実行ファイルが大きすぎます。')
            binary=archive.read(entry)
        target=root/name;staging=root/(name+'.tmp');staging.write_bytes(binary);staging.chmod(0o755);os.replace(staging,target)
        deno=[str(target)]
    # Explicitly enable this portable runtime; do not rely on per-user settings.
    yt=[*yt,'--js-runtimes','deno:'+deno[0]]
    return yt,ffmpeg
