"""Offline diagnostics; never opens a camera, downloads files, or starts inference."""
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


BASE = Path(__file__).resolve().parent
DEPENDENCIES = (('tkinter', '画面'), ('cv2', '画像処理'), ('numpy', '数値処理'),
                ('scipy', '中心線の処理'), ('skimage', '輪郭の処理'), ('PIL', '画像表示'))

# Keep dependency imports in bounded child processes. Missing DLLs or a failed
# native import must not prevent the stdlib-only parent from saving a report.
IMPORT_PROBE = r'''
import importlib, json, sys
try:
    module = importlib.import_module(sys.argv[1])
    version = getattr(module, '__version__', getattr(module, 'TkVersion', '不明'))
    print(json.dumps({'ok': True, 'version': str(version)}))
except Exception as exc:
    print(json.dumps({'ok': False, 'error': type(exc).__name__}))
'''

TORCH_PROBE = r'''
import json, sys
try:
    import torch
    result = {'ok': True, 'version': str(torch.__version__),
              'cuda_build': torch.version.cuda, 'cuda_available': False,
              'gpu_names': [], 'model': {'ok': False, 'error': 'not_checked'}}
    try:
        result['cuda_available'] = bool(torch.cuda.is_available())
        if result['cuda_available']:
            result['gpu_names'] = [str(torch.cuda.get_device_name(i)) for i in range(min(torch.cuda.device_count(), 4))]
    except Exception as exc:
        result['cuda_error'] = type(exc).__name__
    if sys.argv[1] != '-':
        try:
            checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
            metadata = checkpoint.get('metadata', {})
            architecture = metadata.get('architecture')
            state = checkpoint.get('state_dict')
            if architecture not in ('kaiko_small_unet_v1', 'kaiko_context_unet_v2', 'kaiko_texture_mlp_v1') or not isinstance(state, dict) or not state:
                raise ValueError('Unexpected checkpoint format')
            if not all(isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError('Unexpected state values')
            sys.path.insert(0, sys.argv[2])
            from silkworm_scratch import create_model
            network = create_model(architecture)
            network.load_state_dict(state, strict=True)
            result['model'] = {'ok': True, 'architecture': architecture,
                               'model_version': metadata.get('model_version'),
                               'parameters': sum(value.numel() for value in state.values())}
        except Exception as exc:
            result['model'] = {'ok': False, 'error': type(exc).__name__}
    print(json.dumps(result))
except Exception as exc:
    print(json.dumps({'ok': False, 'error': type(exc).__name__}))
'''


def run_json_probe(code, arguments, timeout, runner=subprocess.run):
    try:
        result = runner([sys.executable, '-c', code, *arguments], capture_output=True,
                        text=True, encoding='utf-8', errors='replace', timeout=timeout,
                        shell=False, check=False)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timeout', 'timeout_seconds': timeout}
    except OSError as exc:
        return {'ok': False, 'error': type(exc).__name__}
    if result.returncode:
        return {'ok': False, 'error': 'process_exit', 'exit_code': result.returncode}
    try:
        value = json.loads(result.stdout.strip())
        if not isinstance(value, dict) or not isinstance(value.get('ok'), bool):
            raise ValueError('Invalid probe response')
        return value
    except (ValueError, TypeError):
        return {'ok': False, 'error': 'invalid_response'}


def nvidia_command():
    found = shutil.which('nvidia-smi')
    if found:
        return found
    if os.name == 'nt':
        candidates = [Path(os.environ.get('SystemRoot', 'C:/Windows'))/'System32/nvidia-smi.exe',
                      Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'NVIDIA Corporation/NVSMI/nvidia-smi.exe']
        for path in candidates:
            if path.is_file():
                return str(path)
    return None


def probe_nvidia(command, runner=subprocess.run):
    if not command:
        return {'ok': False, 'error': 'not_found'}
    try:
        result = runner([command, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace',
                        timeout=5, shell=False, check=False)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timeout', 'timeout_seconds': 5}
    except OSError as exc:
        return {'ok': False, 'error': type(exc).__name__}
    if result.returncode:
        return {'ok': False, 'error': 'process_exit', 'exit_code': result.returncode}
    rows = []
    for row in csv.reader(io.StringIO(result.stdout[:4096])):
        if len(row) == 3:
            rows.append({'name': row[0].strip()[:100], 'driver': row[1].strip()[:40], 'memory_mib': row[2].strip()[:24]})
    return {'ok': True, 'gpus': rows[:4]} if rows else {'ok': False, 'error': 'invalid_response'}


def collect_environment(base=BASE, runner=subprocess.run, command_finder=nvidia_command):
    base = Path(base)
    model = base/'models/kaiko_scratch.pt'
    model_info = {'exists': model.is_file()}
    if model_info['exists']:
        try:
            model_info['sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
        except OSError as exc:
            model_info['read_error'] = type(exc).__name__
    expected_env = Path(os.environ.get('KAIKO_VENV_PATH', str(base/'.venv')))
    in_venv = Path(sys.prefix).resolve() == expected_env.resolve()
    with ThreadPoolExecutor(max_workers=3) as pool:
        dependencies = {name: pool.submit(run_json_probe, IMPORT_PROBE, [name], 10, runner)
                        for name, _ in DEPENDENCIES}
        torch = pool.submit(run_json_probe, TORCH_PROBE, [str(model) if model_info.get('sha256') else '-', str(base.resolve())], 30, runner)
        found = {name: future.result() for name, future in dependencies.items()}
        torch_info = torch.result()
    report = {'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
              'python': platform.python_version(), 'system': platform.system(),
              'release': platform.release(), 'machine': platform.machine(),
              'processor': platform.processor(), 'using_app_venv': in_venv,
              'dependencies': found, 'torch': torch_info, 'model': model_info,
              'nvidia': probe_nvidia(command_finder(), runner),
              'camera_opened': False, 'network_used': False, 'inference_tested': False}
    return report


def clean(value, limit=150):
    return str(value).replace('\n', ' ').replace('\r', ' ')[:limit]


def failure_text(result):
    if result.get('error') == 'timeout':
        return f"時間切れ（{result['timeout_seconds']}秒）。未導入とは限りません。"
    if result.get('error') == 'process_exit':
        return f"確認プロセスが終了しました（コード{result.get('exit_code')}）。"
    return clean(result.get('error', '確認不可'))


def format_report(report):
    lines = ['蚕CHANNEL 起動環境の確認', f"作成日時：{clean(report['created_at'])}", '',
             f"OS：{clean(report['system'])} {clean(report['release'])} / {clean(report['machine'])}",
             f"Python：{clean(report['python'])}",
             f"CPU情報：{clean(report['processor']) or '取得できませんでした'}",
             'アプリの.venvから実行：' + ('はい' if report['using_app_venv'] else 'いいえ（この結果は別のPython環境の可能性があります）'), '', '■ ライブラリ']
    for name, label in DEPENDENCIES:
        result = report['dependencies'][name]
        status = f"OK / {clean(result.get('version', '不明'))}" if result['ok'] else '確認不可 / '+failure_text(result)
        lines.append(f'{label}（{name}）：{status}')
    torch = report['torch']
    lines.extend(['', '■ 自作モデルとPyTorch'])
    if not torch['ok']:
        lines.append('PyTorch：確認不可 / '+failure_text(torch))
    else:
        lines.append('PyTorch：'+clean(torch.get('version')))
        if torch.get('cuda_build') is None:
            lines.append('PyTorchの種類：CPU版（NVIDIA GPUの有無とは別です）')
        else:
            lines.append('PyTorchのCUDAビルド：'+clean(torch['cuda_build']))
        if torch.get('cuda_available'):
            lines.append('CUDA認識：利用可能と認識。解析動作・速度はこの確認では検証していません。')
            lines.extend('CUDA GPU：'+clean(name) for name in torch.get('gpu_names', []))
        else:
            lines.append('CUDA認識：利用不可。自作モデルはCPUを使う設定になります。')
        if torch.get('cuda_error'):
            lines.append('CUDA確認時の情報：'+clean(torch['cuda_error']))
    model = report['model']
    if not model['exists']:
        lines.append('モデル：models/kaiko_scratch.pt がありません。更新ZIPのmodelsも必要です。')
    elif model.get('read_error'):
        lines.append('モデル：ファイル読込不可 / '+clean(model['read_error']))
    else:
        lines.append('モデルSHA-256：'+model['sha256'])
        if torch['ok']:
            check = torch.get('model', {})
            if check.get('ok'):
                lines.append(f"重みの読込：OK / {clean(check.get('architecture'))} / {check.get('parameters')} パラメーター")
                if check.get('model_version'):
                    lines.append('モデル版：'+clean(check['model_version']))
            else:
                lines.append('重みの読込：確認不可 / '+failure_text(check))
        else:
            lines.append('重みの読込：PyTorchの確認ができないため未確認')
    lines.extend(['', '■ NVIDIAのドライバー情報'])
    driver = report['nvidia']
    if driver['ok']:
        for gpu in driver['gpus']:
            lines.append(f"{clean(gpu['name'])} / ドライバー {clean(gpu['driver'])} / VRAM {clean(gpu['memory_mib'])} MiB")
    elif driver.get('error') == 'not_found':
        lines.append('nvidia-smi未検出。NVIDIA GPUの有無は、この結果だけでは分かりません。')
    else:
        lines.append('nvidia-smi：確認不可 / '+failure_text(driver))
    lines.extend(['', '■ 次に確認すること'])
    if not report['using_app_venv']:
        lines.append('今までのapp.pyと.venvがあるフォルダへ更新し、その場所のBATから起動してください。')
    if any(not check['ok'] for check in report['dependencies'].values()) or not torch['ok']:
        lines.append('依存ライブラリを確認できませんでした。通常の導入は setup_scratch_windows.bat を使います。')
    if model.get('exists') and torch['ok'] and not torch.get('model', {}).get('ok'):
        lines.append('モデルを読み込めませんでした。更新ZIPを再展開し、modelsも更新してください。')
    lines.extend(['アプリが動く場合、この確認結果を送る作業は必須ではありません。',
                  '起動で止まった場合は、このテキストをチャットに添付すると原因を確認できます。', '',
                  'この確認はオフラインです。カメラを開かず、画像・映像を取得せず、モデルの推論も実行していません。',
                  'GPUでの解析精度・速度や、OBSとの同時動作が確認できたという意味ではありません。'])
    return '\n'.join(lines)+'\n'


def save_report(text, base=BASE):
    directory = Path(base)/'data/diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime('environment_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6]+'.txt'
    path = directory/name
    path.write_text(text, encoding='utf-8-sig')
    return path


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    print('環境を確認しています。通常は数秒〜30秒程度かかります。カメラやネット接続は使いません。', flush=True)
    text = format_report(collect_environment())
    print(text, flush=True)
    try:
        path = save_report(text)
    except OSError:
        print('結果のファイル保存ができませんでした。画面の結果をコピーして送れます。')
        return 1
    print('保存先：data/diagnostics/'+path.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
