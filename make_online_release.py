"""Build signed GitHub payloads and the single-file Windows launcher."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from build_one_click_package import build
from secure_updates import unpack_release, verify_envelope
import tempfile


def render_launcher(channel, public):
    if channel not in ('school','parsonal'): raise ValueError('Unknown branch')
    modulus=int(public['n'],16)
    encoded=base64.b64encode(modulus.to_bytes((modulus.bit_length()+7)//8,'big')).decode()
    root=Path(__file__).resolve().parent
    script=(root/'online_starter.ps1').read_text(encoding='utf-8-sig')
    script=script.replace('__CHANNEL__',channel).replace('__MODULUS__',encoded)
    command="$s=Get-Content -LiteralPath $env:KAIKO_LAUNCHER_FILE -Raw -Encoding UTF8; $m=('# KAIKO_'+'STARTER'); & ([ScriptBlock]::Create($s.Substring($s.IndexOf($m)+$m.Length)))"
    header='\n'.join(['@echo off','setlocal DisableDelayedExpansion','chcp 65001 >nul','set "KAIKO_LAUNCHER_FILE=%~f0"',
        '"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "'+command+'"',
        'set "KAIKO_EXIT_CODE=%ERRORLEVEL%"','if not "%KAIKO_EXIT_CODE%"=="0" pause','exit /b %KAIKO_EXIT_CODE%','# KAIKO_STARTER',''])
    text=(header+script).replace('\r\n','\n').replace('\n','\r\n')
    assert text.count('# KAIKO_STARTER')==1
    return text.encode('utf-8')


def release(output, models, key, channel, sequence, version):
    root=Path(__file__).resolve().parent
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    archive=output/'payload.zip'
    build(archive,models,release_sequence=sequence,version=version,public_update=True,school=channel=='school',online=True)
    raw=archive.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    artifact=f'package_{sequence}_{digest[:12]}.b64'
    info={'schema':1,'release_sequence':sequence,'version':version,'minimum_bootstrap':2,'channel':channel,'encoding':'base64',
          'url':f'https://raw.githubusercontent.com/sasakinoriyuki/kaiko_size_check/{channel}/.updates/{artifact}',
          'bytes':len(raw),'sha256':digest}
    payload=json.dumps(info,sort_keys=True,separators=(',',':')).encode()
    signature=subprocess.run(['openssl','dgst','-sha256','-sign',str(key)],input=payload,capture_output=True,check=True).stdout
    envelope=json.dumps({'payload':base64.b64encode(payload).decode(),'signature':base64.b64encode(signature).decode()},separators=(',',':')).encode()
    public=json.loads((root/'update_public_key.json').read_text())
    verify_envelope(envelope,public)
    with tempfile.TemporaryDirectory() as folder:unpack_release(raw,info,Path(folder)/'verified')
    updates=output/'.updates';updates.mkdir(exist_ok=True)
    (updates/artifact).write_bytes(base64.b64encode(raw)+b'\n')
    (updates/'channel.json').write_bytes(envelope)
    bat=render_launcher(channel,public)
    (output/'蚕チェック.bat').write_bytes(bat)
    with zipfile.ZipFile(output/f'kaiko_{channel}_launcher.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('蚕チェック.bat',bat)
    (output/'release_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--private-key',required=True);parser.add_argument('--channel',choices=['school','parsonal'],required=True)
    parser.add_argument('--sequence',type=int,required=True);parser.add_argument('--version',required=True)
    a=parser.parse_args();print(json.dumps(release(a.output,a.models,a.private_key,a.channel,a.sequence,a.version)))
