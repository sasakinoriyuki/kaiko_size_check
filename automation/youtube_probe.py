"""Probe public archive access; never read browser cookies or download media."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SEEDS = ['0rijis5y8c0', 'RzZ0gH6DMF8', 'pbcVIyIBjig']


def run_json(arguments, runner=subprocess.run):
    result = runner([sys.executable, '-m', 'yt_dlp', '--ignore-config',
                    '--no-warnings', '--socket-timeout', '15', '--retries', '1',
                    '--js-runtimes', 'node', *arguments],
                    capture_output=True, timeout=150, check=False)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace')
        detail = re.sub(r'https?://\S+', '[URL]', detail)
        raise RuntimeError(detail[-1200:].strip())
    return json.loads(result.stdout)


def probe(runner=subprocess.run):
    report = {'schema': 1, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'status': 'unavailable', 'channel_id': None, 'attempts': [],
              'cookie_access': False, 'media_downloaded': False}
    for video_id in SEEDS:
        try:
            info = run_json(['--no-playlist', '--skip-download', '--dump-single-json',
                             'https://www.youtube.com/watch?v='+video_id], runner)
            channel = info.get('channel_id', '')
            if not re.fullmatch(r'UC[A-Za-z0-9_-]{22}', channel):
                raise ValueError('No verified channel ID returned for seed')
            report['channel_id'] = channel
            report['channel_name'] = str(info.get('channel', ''))[:200]
            report['attempts'].append({'video_id': video_id, 'metadata_ok': True})
            listing = run_json(['--flat-playlist', '--playlist-end', '10',
                                '--dump-single-json',
                                'https://www.youtube.com/channel/'+channel+'/streams'], runner)
            report['first_archive_ids'] = [e['id'] for e in listing.get('entries', [])
                                          if isinstance(e, dict) and
                                          re.fullmatch(r'[A-Za-z0-9_-]{11}', str(e.get('id', '')))]
            report['status'] = 'metadata_and_archive_listing_available'
            break
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
            report['attempts'].append({'video_id': video_id, 'error': str(exc)[:1400]})
    return report


if __name__ == '__main__':
    target = Path(sys.argv[1] if len(sys.argv)>1 else '.automation/youtube_probe.json')
    report = probe();target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
