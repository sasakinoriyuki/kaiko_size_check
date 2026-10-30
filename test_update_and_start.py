import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import update_and_start as update


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.source = self.base/'source'; self.target = self.base/'target'
        self.source.mkdir(); self.target.mkdir()
        (self.target/'.venv/Scripts').mkdir(parents=True); (self.target/'.venv/Scripts/python.exe').write_bytes(b'python')
        (self.target/'app.py').write_bytes(b'old app')
        (self.target/'data').mkdir(); (self.target/'data/measurements.csv').write_bytes(b'user measurements')
        (self.source/'app.py').write_bytes(b'new app'); (self.source/'new.py').write_bytes(b'new file')
        self.manifest = {'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}}
        self.write_manifest()

    def write_manifest(self):
        (self.source/'update_manifest.json').write_text(json.dumps(self.manifest))

    def test_installs_with_backup_without_changing_user_data_or_environment(self):
        backup = update.install_update(self.source, self.target)
        self.assertEqual((backup/'app.py').read_bytes(), b'old app')
        self.assertEqual((self.target/'app.py').read_bytes(), b'new app')
        self.assertEqual((self.target/'data/measurements.csv').read_bytes(), b'user measurements')
        self.assertEqual((self.target/'.venv/Scripts/python.exe').read_bytes(), b'python')

    def test_corrupt_source_is_rejected_before_overwriting(self):
        (self.source/'new.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')

    def test_rejects_protected_paths_and_outside_target(self):
        for name in ('../outside.py', 'data/measurements.csv', '.venv/python.exe'):
            self.manifest['files'][name] = 'bad'; self.write_manifest()
            with self.assertRaises(ValueError):
                update.install_update(self.source, self.target)
            del self.manifest['files'][name]
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')

    def test_failure_restores_replaced_files_and_removes_new_files(self):
        original = update.atomic_copy
        calls = []
        def fail_once(source, destination):
            calls.append(str(destination))
            if len(calls) == 2:
                raise OSError('test write failure')
            return original(source, destination)
        with patch.object(update, 'atomic_copy', side_effect=fail_once):
            with self.assertRaises(OSError):
                update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')
        self.assertFalse((self.target/'new.py').exists())

    def test_symlink_cannot_redirect_application_update_to_measurements(self):
        (self.target/'app.py').unlink()
        (self.target/'app.py').symlink_to(self.target/'data/measurements.csv')
        with self.assertRaises(ValueError):
            update.install_update(self.source, self.target)
        self.assertEqual((self.target/'data/measurements.csv').read_bytes(), b'user measurements')
