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

    def test_windows_names_and_protected_folders_are_validated_on_every_platform(self):
        invalid = ('DATA/measurements.csv', '.VENV/python.exe', 'Backups/app.py',
                   r'models\..\data\x', r'C:\app.py', r'\\server\share\app.py',
                   'models/file.pt:stream', 'models/CON.txt', 'models/NUL',
                   'models/file.pt.', 'models/file.pt ', './app.py', '', 'models//file.pt')
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(ValueError):
                update.safe_relative(name)
        self.assertEqual(update.safe_relative(r'models\own.pt'), Path('models')/'own.pt')

    def test_case_insensitive_duplicate_files_and_malformed_manifests_are_rejected(self):
        self.manifest['files']['APP.PY'] = self.manifest['files']['app.py']; self.write_manifest()
        with self.assertRaises(ValueError):
            update.install_update(self.source, self.target)
        for files in ([], {}, {'app.py': None}):
            self.manifest['files'] = files; self.write_manifest()
            with self.assertRaises(ValueError):
                update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')

    def test_junction_like_backup_destination_is_rejected(self):
        with patch.object(update, 'is_link', side_effect=lambda p: p.name == 'backups'):
            with self.assertRaises(ValueError):
                update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')

    def test_bad_copy_hash_rolls_back_before_application_is_launched(self):
        original = update.atomic_copy
        calls = []
        def corrupt_once(source, destination):
            calls.append(str(destination))
            if len(calls) == 1:
                destination.write_bytes(b'incorrect copy')
            else:
                original(source, destination)
        with patch.object(update, 'atomic_copy', side_effect=corrupt_once):
            with self.assertRaisesRegex(OSError, 'コピー後'):
                update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')

    def test_one_restore_error_does_not_prevent_other_files_from_being_restored(self):
        (self.target/'new.py').write_bytes(b'old second file')
        (self.source/'third.py').write_bytes(b'new third file')
        self.manifest['files']['third.py'] = hashlib.sha256((self.source/'third.py').read_bytes()).hexdigest()
        self.write_manifest()
        original = update.atomic_copy
        def fail_update_and_one_restore(source, destination):
            if source == self.source/'third.py':
                raise OSError('update failed')
            if 'backups' in source.parts and source.name == 'new.py':
                raise OSError('restore failed')
            original(source, destination)
        with patch.object(update, 'atomic_copy', side_effect=fail_update_and_one_restore):
            with self.assertRaisesRegex(OSError, 'バックアップは残しています'):
                update.install_update(self.source, self.target)
        self.assertEqual((self.target/'app.py').read_bytes(), b'old app')
        self.assertFalse((self.target/'third.py').exists())
        records = list((self.target/'backups').glob('*/_kaiko_backup/record.json'))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertEqual(record['status'], 'rollback_incomplete')
        self.assertEqual(record['restore_errors'][0]['path'], 'new.py')
