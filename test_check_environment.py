import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from check_environment import (DEPENDENCIES, IMPORT_PROBE, TORCH_PROBE,
                               collect_environment, format_report, probe_nvidia,
                               run_json_probe, save_report)


class EnvironmentTests(unittest.TestCase):
    def test_json_probe_uses_argument_list_no_shell_and_timeout(self):
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout='{"ok":true,"version":"1.2"}', stderr='')
        result = run_json_probe(IMPORT_PROBE, ['numpy'], 10, run)
        self.assertTrue(result['ok'])
        command, options = calls[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[-1], 'numpy')
        self.assertFalse(options['shell'])
        self.assertEqual(options['timeout'], 10)
        self.assertEqual(options['encoding'], 'utf-8')

    def test_timeout_and_native_import_failure_produce_safe_results(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs['timeout'], output='private path')
        self.assertEqual(run_json_probe('code', [], 10, timeout)['error'], 'timeout')
        def crashed(command, **kwargs):
            return SimpleNamespace(returncode=-11, stdout='private full path', stderr='private full path')
        self.assertEqual(run_json_probe('code', [], 10, crashed),
                         {'ok': False, 'error': 'process_exit', 'exit_code': -11})
        def bad_json(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout='not JSON', stderr='')
        self.assertEqual(run_json_probe('code', [], 10, bad_json)['error'], 'invalid_response')

    def test_nvidia_query_is_bounded_and_does_not_run_arbitrary_shell(self):
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout='NVIDIA GeForce RTX 4050 Laptop GPU, 580.00, 6144\n', stderr='')
        result = probe_nvidia('C:/Program Files/NVIDIA/nvidia-smi.exe', run)
        self.assertEqual(result['gpus'][0]['name'], 'NVIDIA GeForce RTX 4050 Laptop GPU')
        self.assertEqual(calls[0][0], ['C:/Program Files/NVIDIA/nvidia-smi.exe',
                                     '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'])
        self.assertFalse(calls[0][1]['shell'])
        self.assertEqual(calls[0][1]['timeout'], 5)
        self.assertEqual(probe_nvidia(None, run)['error'], 'not_found')
        self.assertEqual(len(calls), 1)

    def test_cpu_torch_and_present_nvidia_hardware_are_distinguished(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[0] == 'fake-nvidia-smi':
                return SimpleNamespace(returncode=0, stdout='NVIDIA GeForce RTX 5060, 580.00, 8192\n', stderr='')
            if command[2] == TORCH_PROBE:
                value = {'ok': True, 'version': '2.8.0+cpu', 'cuda_build': None, 'cuda_available': False,
                         'gpu_names': [], 'model': {'ok': True, 'architecture': 'kaiko_context_unet_v2', 'parameters': 56929}}
            else:
                value = {'ok': True, 'version': 'test'}
            return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'models').mkdir()
            raw = b'fake model file'; (base/'models/kaiko_scratch.pt').write_bytes(raw)
            report = collect_environment(base, run, lambda: 'fake-nvidia-smi')
            self.assertEqual(report['model']['sha256'], hashlib.sha256(raw).hexdigest())
            text = format_report(report)
            self.assertIn('CPU版（NVIDIA GPUの有無とは別です）', text)
            self.assertIn('NVIDIA GeForce RTX 5060', text)
            self.assertIn('GPUでの解析精度・速度', text)
            self.assertNotIn(folder, text)
            self.assertFalse(report['camera_opened'] or report['network_used'] or report['inference_tested'])
            self.assertEqual(len(calls), len(DEPENDENCIES)+2)

    def test_missing_dependencies_and_model_still_save_readable_report(self):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout='{"ok":false,"error":"ModuleNotFoundError"}', stderr='')
        with tempfile.TemporaryDirectory() as folder:
            report = collect_environment(folder, run, lambda: None)
            text = format_report(report)
            self.assertIn('models/kaiko_scratch.pt がありません', text)
            self.assertIn('GPUの有無は、この結果だけでは分かりません', text)
            self.assertIn('setup_scratch_windows.bat', text)
            saved = save_report(text, folder)
            self.assertEqual(saved.read_text(encoding='utf-8-sig'), text)
            self.assertTrue(saved.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(saved.parent, Path(folder)/'data/diagnostics')
            self.assertNotEqual(saved, save_report(text, folder))

    def test_gpu_detection_does_not_claim_inference_was_verified(self):
        def run(command, **kwargs):
            value = {'ok': True, 'version': '2.8.0+cu128', 'cuda_build': '12.8',
                     'cuda_available': True, 'gpu_names': ['NVIDIA RTX'],
                     'model': {'ok': False, 'error': 'ValueError'}} if command[2] == TORCH_PROBE else {'ok': True, 'version': 'test'}
            return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'models').mkdir(); (base/'models/kaiko_scratch.pt').write_bytes(b'fake')
            report = collect_environment(base, run, lambda: None)
            text = format_report(report)
            self.assertIn('CUDA認識：利用可能と認識', text)
            self.assertIn('解析動作・速度はこの確認では検証していません', text)
            self.assertIn('重みの読込：確認不可', text)


if __name__ == '__main__':
    unittest.main()
