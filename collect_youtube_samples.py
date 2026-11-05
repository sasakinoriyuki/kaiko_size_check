"""Collect images from short sections of a completed YouTube archive on the user's PC.

Uses existing yt-dlp/ffmpeg commands. Never downloads a whole channel or a model.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform == 'win32'


def _command_options():
    return {'shell': False, **({'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if IS_WINDOWS else {})}


def _local_candidates(name):
    candidates = []
    located = shutil.which(name)
    if located:
        candidates.append([located])
    for directory in [BASE, BASE / "tools", Path(sys.executable).parent]:
        for suffix in [".exe", ""]:
            candidate = directory / (name + suffix)
            if candidate.is_file():
                candidates.append([str(candidate)])
    if IS_WINDOWS:
        local = os.environ.get('LOCALAPPDATA')
        if local:
            winget = Path(local)/'Microsoft/WinGet'
            link = winget/'Links'/(name+'.exe')
            if link.is_file():
                candidates.append([str(link)])
            if name == 'ffmpeg':
                for pattern in ['Gyan.FFmpeg*/ffmpeg*/bin/ffmpeg.exe', 'Gyan.FFmpeg*/bin/ffmpeg.exe']:
                    candidates.extend([[str(p)] for p in (winget/'Packages').glob(pattern) if p.is_file()])
        if name == 'deno':
            path = Path.home()/'.deno/bin/deno.exe'
            if path.is_file():
                candidates.append([str(path)])
    return list({tuple(c): c for c in candidates}.values())


def find_tool(name, runner=None):
    runner = runner or subprocess.run
    candidates = _local_candidates(name)
    if name == 'yt-dlp':
        if importlib.util.find_spec('yt_dlp') is not None:
            candidates.insert(0, [sys.executable, '-m', 'yt_dlp'])
        launcher = shutil.which('py') if IS_WINDOWS else None
        if launcher:
            candidates.insert(0, [launcher, '-m', 'yt_dlp'])
        working = []
        for command in candidates:
            try:
                result = runner([*command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=5, check=False, **_command_options())
                if result.returncode:
                    continue
                version = re.search(r'\b(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\.(\d+))?', result.stdout.decode('utf-8', errors='replace'))
                key = tuple(int(v or 0) for v in version.groups()) if version else (0, 0, 0, 0)
                working.append((key, command))
            except (OSError, subprocess.TimeoutExpired):
                continue
        if working:
            return max(working, key=lambda c: c[0])[1]
    elif candidates:
        return candidates[0]
    raise RuntimeError(f"{name} が見つかりません。導入済みのコマンドをPATHに追加するか、このアプリのフォルダに置いてください。")


def _run(command, runner, timeout):
    tool = 'yt-dlp' if 'yt_dlp' in command or 'yt-dlp' in Path(command[0]).name else 'ffmpeg'
    try:
        result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, **_command_options())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{tool} の処理が時間切れになりました。通信状態を確認して再実行してください。") from exc
    except OSError as exc:
        raise RuntimeError(f'{tool} を起動できませんでした。コマンドの導入先を確認してください。') from exc
    if result.returncode:
        detail = (result.stderr or b'').decode('utf-8', errors='replace')
        detail = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', detail)
        detail = re.sub(r'https?://[^\s<>\"\x27]+', '[URL非表示]', detail)
        detail = re.sub(r'(?im)\b(cookie|authorization)\s*[:=][^\r\n]*', r'\1=[非表示]', detail)
        detail = re.sub(r'(?i)\b(token|signature|cookie|authorization|api[_-]?key)\s*[:=]\s*\S+', r'\1=[非表示]', detail)
        detail = ''.join(c for c in detail if c.isprintable() or c in '\n\t').strip()[-1800:]
        raise RuntimeError(f'{tool} の処理が失敗しました（終了コード {result.returncode}）。\n'+(detail or '詳細が返されませんでした。コマンドの導入と動画の公開状態を確認してください。'))
    return result.stdout


def youtube_samples(url, count=5, cancelled=None, runner=subprocess.run, yt_command=None, ffmpeg_command=None):
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or parsed.hostname not in {
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"} or parsed.username or parsed.password:
        raise ValueError("YouTubeの動画URLを1本入力してください。")
    if not 1 <= count <= 20:
        raise ValueError("抽出枚数は1〜20です。")
    yt = yt_command or find_tool("yt-dlp", runner=runner)
    ffmpeg = ffmpeg_command or find_tool("ffmpeg")
    common = ["--ignore-config", "--no-playlist", "--no-warnings", "--no-progress",
              "--socket-timeout", "20", "--retries", "2", "--fragment-retries", "2"]
    if IS_WINDOWS:
        deno = _local_candidates('deno')
        if deno:
            common += ['--js-runtimes', 'deno:'+deno[0][0]]
    payload = _run([*yt, *common, "--skip-download", "--dump-single-json", url], runner, 180)
    try:
        info = json.loads(payload)
        duration = float(info.get("duration", 0) or 0)
    except (ValueError, TypeError) as exc:
        raise ValueError("動画の情報を読み込めませんでした。") from exc
    if (info.get("_type") in {"playlist", "multi_video"} or info.get("is_live")
            or info.get("live_status") in {"is_live", "is_upcoming"}
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("再生時間の分かる、完了済みのアーカイブ動画を1本選んでください。")
    video_id = str(info.get("id", "video"))
    # This is metadata only; the external title/ID never becomes a path or command.
    times = [duration*.5] if count == 1 else np.linspace(duration*.1, duration*.9, count)
    with TemporaryDirectory(prefix="kaiko_youtube_") as folder:
        directory = Path(folder)
        cached = {}
        for index, target in enumerate(times, 1):
            if cancelled is not None and cancelled.is_set():
                raise ValueError("画像の準備を中止しました。")
            start = max(0., min(float(target)-1., max(0., duration-3.)))
            end = min(duration, start+3.)
            section = (round(start, 3), round(end, 3))
            clip = cached.get(section)
            if clip is None:
                prefix = f"clip_{index:03d}"
                template = str(directory / (prefix + ".%(ext)s"))
                command = [*yt, *common, "--ffmpeg-location", ffmpeg[0],
                           "-f", "bv/b", "--download-sections",
                           f"*{section[0]:.3f}-{section[1]:.3f}", "-o", template, url]
                _run(command, runner, 600)
                clips = [p for p in directory.glob(prefix+".*") if p.suffix.lower() in {".mp4", ".webm", ".mkv"}]
                if len(clips) != 1:
                    raise RuntimeError("短い区間の録画ファイルを取得できませんでした。")
                clip = cached[section] = clips[0]
            if cancelled is not None and cancelled.is_set():
                raise ValueError("画像の準備を中止しました。")
            png = _run([*ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(clip),
                        "-frames:v", "1", "-f", "image2pipe",
                        "-vcodec", "png", "pipe:1"], runner, 120)
            if not png:
                raise RuntimeError("取得した区間から画像を復元できませんでした。別の動画か保存済みの録画を選んでください。")
            image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("取得した区間から画像を読み込めませんでした。")
            yield image, {"source_file": "youtube_archive", "source_kind": "youtube_archive",
                          "video_id": video_id, "video_url": url, "video_title": str(info.get("title", "")),
                          "requested_video_seconds": float(target),
                          "requested_section_start": section[0], "requested_section_end": section[1],
                          "frame_selection": "first_decodable_frame_in_section",
                          "timing_note": "Approximate source position after section cutting; not an exact original frame index."}
