"""Bounded HTTPS downloads, pinned signatures, and data-preserving update ZIPs."""
import base64
import hashlib
import hmac
import io
import json
from pathlib import Path
import re
import stat
import urllib.parse
import urllib.request
import zipfile

from update_and_start import safe_relative

BOOTSTRAP_API = 1
MAX_ARCHIVE = 32*1024*1024
ALLOWED_HOSTS = {'drive.google.com', 'drive.usercontent.google.com'}


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('重複したJSONキーがあります。')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def sequence(manifest):
    value = manifest.get('release_sequence')
    if type(value) is not int or value < 1:
        raise ValueError('更新番号が不正です。')
    return value


def checked_url(url):
    p = urllib.parse.urlsplit(url)
    allowed = p.hostname in ALLOWED_HOSTS or bool(p.hostname and p.hostname.endswith('.googleusercontent.com'))
    if p.scheme != 'https' or not allowed or p.username or p.password or p.port not in (None,443):
        raise ValueError('許可されていない更新先です。')
    return url


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def download(url, limit, timeout=8):
    checked_url(url)
    opener = urllib.request.build_opener(CheckedRedirect())
    request = urllib.request.Request(url, headers={'User-Agent':'KaikoSizeCheck/1', 'Cache-Control':'no-cache'})
    with opener.open(request, timeout=timeout) as response:
        checked_url(response.geturl())
        data = response.read(limit+1)
    if len(data) > limit:
        raise ValueError('更新ファイルが大きすぎます。')
    return data


def verify_envelope(raw, key):
    if len(raw)>65536:
        raise ValueError('更新案内が大きすぎます。')
    envelope = strict_json(raw)
    payload = base64.b64decode(envelope['payload'], validate=True)
    signature = base64.b64decode(envelope['signature'], validate=True)
    if key.get('algorithm') != 'rsa-pkcs1v15-sha256':
        raise ValueError('署名方式が違います。')
    n, e = int(key['n'],16), key['e']
    if not 2048 <= n.bit_length() <= 4096 or e != 65537 or n%2 != 1:
        raise ValueError('公開鍵を確認してください。')
    size = (n.bit_length()+7)//8
    number = int.from_bytes(signature,'big')
    if len(signature)!=size or number>=n:
        raise ValueError('更新署名の長さが違います。')
    digest_info = bytes.fromhex('3031300d060960864801650304020105000420')+hashlib.sha256(payload).digest()
    expected = b'\x00\x01'+b'\xff'*(size-len(digest_info)-3)+b'\x00'+digest_info
    actual = pow(number,e,n).to_bytes(size,'big')
    if not hmac.compare_digest(actual,expected):
        raise ValueError('更新の署名が一致しません。')
    info = strict_json(payload)
    if info.get('schema') != 1 or type(info.get('minimum_bootstrap')) is not int or info['minimum_bootstrap']>BOOTSTRAP_API:
        raise ValueError('この起動ファイルに対応していない更新です。')
    sequence(info)
    checked_url(info['url'])
    if not re.fullmatch('[a-f0-9]{64}',info['sha256']):
        raise ValueError('更新ハッシュが不正です。')
    if type(info.get('bytes')) is not int or not 1 <= info['bytes'] <= MAX_ARCHIVE:
        raise ValueError('更新サイズが不正です。')
    return info


def read_package(root):
    root = Path(root).resolve()
    manifest = strict_json((root/'_internal/update_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('layout') != 'one_click' or manifest.get('schema_version') != 2:
        raise ValueError('統一起動版のパッケージではありません。')
    sequence(manifest)
    files = manifest.get('files')
    if not isinstance(files,dict) or not 3 <= len(files) <= 200 or '_internal/app.py' not in files:
        raise ValueError('更新内容を確認してください。')
    seen = set()
    bats = []
    from update_and_start import check_path
    for name,digest in files.items():
        relative = safe_relative(name)
        identity = relative.as_posix().casefold()
        if identity in seen or identity == '_internal/update_manifest.json':
            raise ValueError('重複した更新ファイルがあります。')
        seen.add(identity)
        if relative.suffix.casefold()=='.bat': bats.append(relative.as_posix())
        path = check_path(root,relative)
        if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'不足・破損したファイルがあります：{relative.name}')
    if bats != ['蚕チェック.bat']:
        raise ValueError('起動バッチは蚕チェック.batの1本だけ必要です。')
    return manifest


def unpack_release(raw, info, destination):
    if len(raw)!=info['bytes'] or hashlib.sha256(raw).hexdigest()!=info['sha256']:
        raise ValueError('ダウンロードした更新のハッシュが違います。')
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    seen = set(); total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if not 3 <= len(archive.infolist()) <= 200:
            raise ValueError('更新のファイル数が不正です。')
        for item in archive.infolist():
            if item.is_dir(): raise ValueError('予期しないディレクトリエントリです。')
            relative = safe_relative(item.filename)
            identity = relative.as_posix().casefold()
            if identity in seen or stat.S_IFMT(item.external_attr>>16)==stat.S_IFLNK:
                raise ValueError('更新に重複またはリンクが含まれています。')
            if (relative.suffix.casefold() not in {'.py','.ps1','.bat','.json','.txt','.md','.pt'} and relative.name != 'LICENSE') or 'examples' in [part.casefold() for part in relative.parts]:
                raise ValueError('自動更新に画像等を含めることはできません。')
            total += item.file_size
            if total>64*1024*1024 or item.file_size>32*1024*1024:
                raise ValueError('展開後の更新が大きすぎます。')
            seen.add(identity)
            path = destination/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.read(item))
    manifest = read_package(destination)
    expected = {Path(name).as_posix().casefold() for name in manifest['files']} | {'_internal/update_manifest.json'}
    if seen != expected or sequence(manifest)!=sequence(info):
        raise ValueError('更新の内容と案内が一致しません。')
    return manifest
