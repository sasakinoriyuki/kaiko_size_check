"""Publisher-only: sign an image-free update; never bundle the private key."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from secure_updates import unpack_release,verify_envelope


def sign(archive, key, url, output):
    archive=Path(archive);raw=archive.read_bytes()
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('_internal/update_manifest.json'))
        public=json.loads(z.read('_internal/update_public_key.json'))
    info={'schema':1,'release_sequence':manifest['release_sequence'],'version':manifest['version'],
          'minimum_bootstrap':1,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    with tempfile.TemporaryDirectory() as folder: unpack_release(raw,info,Path(folder)/'check')
    payload=json.dumps(info,sort_keys=True,separators=(',',':')).encode()
    signature=subprocess.run(['openssl','dgst','-sha256','-sign',str(key)],input=payload,capture_output=True,check=True).stdout
    envelope=json.dumps({'payload':base64.b64encode(payload).decode(),'signature':base64.b64encode(signature).decode()},separators=(',',':')).encode()
    verify_envelope(envelope,public)
    Path(output).write_bytes(envelope)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--private-key',required=True)
    parser.add_argument('--url',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();sign(a.archive,a.private_key,a.url,a.output)
