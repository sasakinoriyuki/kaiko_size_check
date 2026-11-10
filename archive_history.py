"""Collection receipts contain opaque keys, never images or size estimates."""
import hashlib
import json
from pathlib import Path
import re


def completion_key(video_id,count):
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',video_id) or not 1<=count<=20:
        raise ValueError('取得済み記録の動画IDまたは枚数が不正です。')
    return hashlib.sha256(f'kaiko-archive-completed-v1:{video_id}:{count}'.encode()).hexdigest()


def load_completion_keys(path=None):
    path=Path(path) if path else Path(__file__).resolve().parent/'archive_completion_history.json'
    if not path.exists():return set()
    if path.stat().st_size>1024**2:raise ValueError('取得済み履歴が大きすぎます。')
    info=json.loads(path.read_text(encoding='utf-8'))
    if info.get('schema')!=1 or not isinstance(info.get('keys'),list) or any(not isinstance(k,str) or not re.fullmatch('[a-f0-9]{64}',k) for k in info['keys']):
        raise ValueError('取得済み履歴の形式が不正です。')
    return set(info['keys'])
