import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import bootstrap as boot
import secure_updates as updates
from instance_lock import FileLock
from update_and_start import install_update,restore_update,safe_relative


def package(root, seq=1, marker='original'):
    root=Path(root); (root/'_internal').mkdir(parents=True,exist_ok=True)
    values={'蚕チェック.bat':b'@echo off\r\n','_internal/app.py':marker.encode(),
            '_internal/bootstrap.py':b'# setup','_internal/LICENSE':b'MIT','_internal/requirements.txt':b'numpy==2.5.4\n'}
    for name,data in values.items(): (root/name).write_bytes(data)
    info={'layout':'one_click','schema_version':2,'release_sequence':seq,'version':str(seq),
          'files':{name:hashlib.sha256(data).hexdigest() for name,data in values.items()}}
    (root/'_internal/update_manifest.json').write_text(json.dumps(info));return info


def archive(root):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(root).rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(root).as_posix())
    return buffer.getvalue()


class OneClickInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'new 日本語 & spaces';self.target=self.root/'installed'
        package(self.source,2,'new');package(self.target,1,'old')
        self.environment=self.target/'runtime/venvs'/('a'*20);self.environment.mkdir(parents=True)
        self.previous={'installed_sequence':1,'active_environment':self.environment.relative_to(self.target).as_posix(),'legacy_import_checked':True}
        boot.atomic_json(self.target/'_internal/data/install_state.json',self.previous)

    def test_fresh_install_has_one_entrypoint_and_copies_its_manifest(self):
        fresh=self.root/'fresh'
        backup=install_update(self.source,fresh,layout='one_click',allow_new=True)
        self.assertEqual(updates.sequence(updates.read_package(fresh)),2)
        self.assertTrue(backup.is_dir())
        self.assertEqual([p.name for p in fresh.rglob('*.bat')],['蚕チェック.bat'])

    def test_running_batch_is_not_replaced_when_updating_helpers(self):
        import update_and_start as module
        original = module.atomic_copy
        copied=[]
        def copy(source,destination):
            self.assertNotEqual(destination.suffix.casefold(),'.bat')
            copied.append(destination)
            return original(source,destination)
        with patch.object(module,'atomic_copy',side_effect=copy):
            install_update(self.source,self.target,layout='one_click')
        self.assertTrue(copied)
        updates.read_package(self.target)

    def test_install_and_post_launch_rollback_preserve_records_and_both_environments(self):
        record=self.target/'_internal/data/measurements.csv';record.write_bytes(b'user records')
        (self.environment/'preserve').write_bytes(b'old environment')
        backup=install_update(self.source,self.target,layout='one_click')
        restore_update(backup,self.target)
        self.assertEqual((self.target/'_internal/app.py').read_bytes(),b'old')
        self.assertEqual(updates.sequence(updates.read_package(self.target)),1)
        self.assertEqual(record.read_bytes(),b'user records')
        self.assertEqual((self.environment/'preserve').read_bytes(),b'old environment')

    def test_nested_user_data_and_environment_paths_are_never_update_targets(self):
        for name in ['_internal/data/measurements.csv','runtime/venvs/x/python.exe','_internal/.venv/a','logs/start.txt','_internal/backups/x']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_relative(name)

    def test_activation_state_failure_restores_old_app_and_manifest(self):
        with patch.object(boot,'atomic_json',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):boot.activate(self.source,self.target,self.environment,self.previous)
        self.assertEqual((self.target/'_internal/app.py').read_bytes(),b'old')
        self.assertEqual(updates.sequence(updates.read_package(self.target)),1)

    def test_offline_start_reuses_installed_app_without_reinstalling(self):
        logs=[]
        with patch.object(boot,'choose_online',return_value=None),patch.object(boot,'ensure_environment',return_value=self.environment),patch.object(boot,'start_gui') as start,patch.object(boot,'preflight') as preflight,patch.object(boot,'install_update') as install:
            self.assertEqual(boot.run(self.target,self.target,logs.append),0)
        install.assert_not_called();preflight.assert_not_called();start.assert_called_once()

    def test_failed_new_gui_rolls_back_then_launches_previous_and_blocks_bad_release(self):
        logs=[];newenv=self.target/'runtime/venvs'/('b'*20);newenv.mkdir()
        with patch.object(boot,'choose_online',return_value=None),patch.object(boot,'ensure_environment',return_value=newenv),patch.object(boot,'preflight'),patch.object(boot,'start_gui',side_effect=[boot.SetupError('bad launch'),object()]) as start:
            self.assertEqual(boot.run(self.source,self.target,logs.append),0)
        self.assertEqual(start.call_count,2)
        self.assertEqual(start.call_args.args[1],self.environment)
        state=boot.load_state(self.target)
        self.assertEqual(state['installed_sequence'],1)
        self.assertEqual(state['blocked_release_sequence'],2)
        self.assertEqual(updates.sequence(updates.read_package(self.target)),1)
        self.assertTrue(newenv.is_dir())

    def test_dependency_failure_leaves_working_app_and_state_untouched(self):
        with patch.object(boot,'choose_online',return_value=None),patch.object(boot,'ensure_environment',side_effect=boot.SetupError('network unavailable')),patch.object(boot,'versions_ok',return_value=True),patch.object(boot,'start_gui') as start:
            self.assertEqual(boot.run(self.source,self.target,lambda _:None),0)
        self.assertEqual(boot.load_state(self.target),self.previous)
        self.assertEqual((self.target/'_internal/app.py').read_bytes(),b'old')
        self.assertEqual(start.call_args.args[1],self.environment)

    def test_incomplete_model_package_is_rejected_before_running_preflight(self):
        with patch.object(boot,'command') as command:
            with self.assertRaisesRegex(boot.SetupError,'自作モデル'):
                boot.preflight(self.source,self.environment,lambda _:None)
        command.assert_not_called()

    def test_gui_wrapper_restores_streams_and_releases_lock_after_normal_exit(self):
        import run_app,sys
        appdir=self.root/'gui/_internal';appdir.mkdir(parents=True)
        token='1'*32;before=(sys.stdout,sys.stderr)
        with patch.object(run_app,'APP_DIR',appdir),patch.object(sys,'argv',['run_app.py','--token',token]),patch('start_scratch_app.main',side_effect=lambda on_ready:on_ready()):
            self.assertEqual(run_app.main(),0)
        self.assertEqual((sys.stdout,sys.stderr),before)
        ready=json.loads((appdir/'data'/('.launch_'+token+'.json')).read_text())
        self.assertEqual(ready['status'],'ready')
        lock=FileLock(appdir/'data/.app.lock');self.assertTrue(lock.acquire());lock.release()

    def test_state_cannot_select_an_executable_outside_managed_environments(self):
        boot.atomic_json(self.target/'_internal/data/install_state.json',{'active_environment':'../elsewhere'})
        with self.assertRaises(boot.SetupError):boot.load_state(self.target)

    def test_school_launcher_cannot_install_into_a_desktop_installation(self):
        manifest_path=self.source/'_internal/update_manifest.json'
        manifest=json.loads(manifest_path.read_text());manifest['installation_flavor']='school'
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(boot,'ensure_environment') as prepare:
            with self.assertRaisesRegex(boot.SetupError,'学校用と通常版'):
                boot.run(self.source,self.target,lambda _:None,allow_online=False)
        prepare.assert_not_called()
        self.assertEqual((self.target/'_internal/app.py').read_bytes(),b'old')
        self.assertEqual(boot.load_state(self.target),self.previous)

    def test_portable_school_copies_old_records_without_writing_old_appdata(self):
        portable=self.root/'portable school';package(portable,3)
        manifest_path=portable/'_internal/update_manifest.json'
        manifest=json.loads(manifest_path.read_text());manifest.update(installation_flavor='school',storage_mode='portable')
        manifest_path.write_text(json.dumps(manifest))
        local=self.root/'profile/AppData/Local'
        old=local/'KaikoChannel/SizeCheckSchool/_internal'
        (old/'data/samples').mkdir(parents=True)
        (old/'app.py').write_text('蚕CHANNEL 体長チェック',encoding='utf-8')
        (old/'data/measurements.csv').write_bytes(b'old measurement history')
        (old/'data/samples/existing.zip').write_bytes(b'old private sample')
        (old/'data/.setup.lock').write_bytes(b'old lock')
        snapshots={p:p.read_bytes() for p in local.rglob('*') if p.is_file()}
        environment=portable/'runtime/venvs'/('b'*20);environment.mkdir(parents=True)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(local)}),patch.object(boot,'ensure_environment',return_value=environment),patch.object(boot,'preflight'),patch.object(boot,'start_gui'):
            self.assertEqual(boot.run(portable,portable,lambda _:None,profile=self.root/'profile',allow_online=False),0)
        self.assertEqual((portable/'data/measurements.csv').read_bytes(),b'old measurement history')
        self.assertEqual((portable/'data/samples/existing.zip').read_bytes(),b'old private sample')
        self.assertFalse((portable/'data/.setup.lock').exists())
        self.assertEqual({p:p.read_bytes() for p in local.rglob('*') if p.is_file()},snapshots)
        self.assertTrue(boot.load_state(portable)['legacy_import_checked'])

    def test_school_dependency_cache_is_inside_portable_folder(self):
        manifest_path=self.source/'_internal/update_manifest.json'
        manifest=json.loads(manifest_path.read_text());manifest.update(installation_flavor='school',storage_mode='portable')
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(boot,'versions_ok',side_effect=[False,True]),patch.object(boot,'command') as run:
            boot.ensure_environment(self.source,self.target,lambda _:None)
        pip_commands=[call.args[0] for call in run.call_args_list if 'pip' in call.args[0]]
        self.assertEqual(len(pip_commands),2)
        for command in pip_commands:
            self.assertIn('--cache-dir',command)
            self.assertEqual(command[command.index('--cache-dir')+1],str(self.target/'runtime/pip-cache'))

    def test_new_torch_requirement_prepares_a_separate_environment(self):
        (self.source/'_internal/runtime.json').write_text(json.dumps({'python':[3,12],'torch_version':'2.9.0'}))
        (self.environment/'old_runtime').write_bytes(b'keep working environment')
        with patch.object(boot,'versions_ok',side_effect=[False,True]),patch.object(boot,'command') as run:
            environment=boot.ensure_environment(self.source,self.target,lambda _:None)
        self.assertNotEqual(environment,self.environment)
        self.assertEqual((self.environment/'old_runtime').read_bytes(),b'keep working environment')
        self.assertTrue(any('torch==2.9.0+cpu' in call.args[0] for call in run.call_args_list))

    def test_lock_and_legacy_data_copy_do_not_destroy_originals(self):
        lock=FileLock(self.root/'lock');self.assertTrue(lock.acquire())
        another=FileLock(self.root/'lock');self.assertFalse(another.acquire());lock.release();self.assertTrue(another.acquire());another.release()
        profile=self.root/'profile';old=profile/'Downloads/kaiko_old/kaiko_project';(old/'data').mkdir(parents=True)
        (old/'app.py').write_text('蚕CHANNEL 体長チェック',encoding='utf-8')
        (old/'data/measurements.csv').write_bytes(b'history')
        self.assertEqual(boot.legacy_sources(profile,self.target),[old/'data'])
        dest=self.root/'data-copy';self.assertEqual(boot.migrate_data(old/'data',dest),1)
        self.assertEqual((dest/'measurements.csv').read_bytes(),b'history')
        self.assertEqual((old/'data/measurements.csv').read_bytes(),b'history')
        (dest/'measurements.csv').write_bytes(b'new data');boot.migrate_data(old/'data',dest)
        self.assertEqual((dest/'measurements.csv').read_bytes(),b'new data')


@unittest.skipUnless(shutil.which('openssl'),'OpenSSL is used only to generate test signatures')
class SignedUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.key=cls.root/'TEST-ONLY.pem'
        subprocess.run(['openssl','genrsa','-out',str(cls.key),'2048'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        modulus=subprocess.check_output(['openssl','rsa','-in',str(cls.key),'-noout','-modulus'],stderr=subprocess.DEVNULL).decode().strip().split('=',1)[1]
        cls.public={'algorithm':'rsa-pkcs1v15-sha256','n':modulus,'e':65537}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def envelope(self, data=None, raw=None):
        payload=json.dumps(data,separators=(',',':')).encode() if raw is None else raw
        signature=subprocess.run(['openssl','dgst','-sha256','-sign',str(self.key)],input=payload,capture_output=True,check=True).stdout
        return json.dumps({'payload':base64.b64encode(payload).decode(),'signature':base64.b64encode(signature).decode()}).encode()

    def info(self, raw=b'zip'):
        return {'schema':1,'release_sequence':2,'minimum_bootstrap':1,'url':'https://drive.google.com/uc?id=test&export=download','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

    def test_authentic_signature_and_tamper_rejection(self):
        info=self.info();encoded=self.envelope(info)
        self.assertEqual(updates.verify_envelope(encoded,self.public),info)
        modified=json.loads(encoded);modified['payload']=base64.b64encode(json.dumps({**info,'release_sequence':99}).encode()).decode()
        with self.assertRaises(ValueError):updates.verify_envelope(json.dumps(modified).encode(),self.public)
        modified=json.loads(encoded);modified['signature']=modified['signature'][:-4]
        with self.assertRaises(ValueError):updates.verify_envelope(json.dumps(modified).encode(),self.public)

    def test_signed_duplicate_keys_and_unsupported_bootstrap_are_rejected(self):
        with self.assertRaises(ValueError):updates.verify_envelope(self.envelope(raw=b'{"schema":1,"schema":1}'),self.public)
        with self.assertRaises(ValueError):updates.verify_envelope(self.envelope({**self.info(),'minimum_bootstrap':99}),self.public)

    def test_urls_are_https_and_host_bounded(self):
        self.assertEqual(updates.checked_url('https://doc-test.googleusercontent.com/download'),'https://doc-test.googleusercontent.com/download')
        for value in ['http://drive.google.com/x','https://drive.google.com.evil/x','https://evilgoogleusercontent.com/x','https://user:pass@drive.google.com/x','https://drive.google.com:444/x','file:///tmp/x']:
            with self.subTest(url=value),self.assertRaises(ValueError):updates.checked_url(value)

    def test_package_integrity_and_data_preservation_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src';package(source,2)
            raw=archive(source);info=self.info(raw)
            updates.unpack_release(raw,info,root/'good')
            self.assertEqual(updates.sequence(updates.read_package(root/'good')),2)
            with self.assertRaises(ValueError):updates.unpack_release(raw+b'bad',info,root/'bad')
            for attack in ['../data/x','_internal/data/records.csv','runtime/venvs/a/python.exe']:
                stream=io.BytesIO()
                with zipfile.ZipFile(stream,'w') as z:
                    z.writestr(attack,b'overwrite');z.writestr('x',b'x');z.writestr('y',b'y')
                value=stream.getvalue()
                with self.assertRaises(ValueError):updates.unpack_release(value,self.info(value),root/('reject_'+hashlib.sha256(attack.encode()).hexdigest()[:8]))

    def test_no_download_on_rollback_or_failed_version_and_offline_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src';package(source)
            (source/'_internal/update_channel.json').write_text(json.dumps({'feed_url':'https://drive.google.com/uc?id=channel'}))
            (source/'_internal/update_public_key.json').write_text(json.dumps(self.public))
            calls=[]
            def old_feed(url,limit,**kwargs):calls.append(url);return self.envelope(self.info())
            self.assertIsNone(boot.choose_online(source,2,root,lambda _:None,fetch=old_feed));self.assertEqual(len(calls),1)
            def offline(*args,**kwargs):raise OSError('offline')
            self.assertIsNone(boot.choose_online(source,1,root,lambda _:None,fetch=offline))

    def test_school_rejects_authentic_desktop_update_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'school';package(source)
            manifest_path=source/'_internal/update_manifest.json'
            manifest=json.loads(manifest_path.read_text());manifest['installation_flavor']='school'
            manifest_path.write_text(json.dumps(manifest))
            (source/'_internal/update_channel.json').write_text(json.dumps({'feed_url':'https://drive.google.com/uc?id=channel'}))
            (source/'_internal/update_public_key.json').write_text(json.dumps(self.public))
            incoming=root/'desktop';package(incoming,2,'incoming desktop')
            raw=archive(incoming);feed=self.envelope(self.info(raw));requests=[]
            def fetch(url,limit,**kwargs):
                requests.append(url)
                return feed if len(requests)==1 else raw
            logs=[]
            self.assertIsNone(boot.choose_online(source,1,root,logs.append,fetch=fetch))
            self.assertEqual(len(requests),2)
            self.assertEqual(list((root/'updates').iterdir()),[])
            self.assertEqual((source/'_internal/app.py').read_bytes(),b'original')

    def test_school_with_unconfigured_feed_never_requests_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'school';package(source)
            (source/'_internal/update_channel.json').write_text(json.dumps({'schema':1,'feed_url':''}))
            def fetch(*args,**kwargs):self.fail('An unconfigured school feed must not access the network')
            self.assertIsNone(boot.choose_online(source,1,root,lambda _:None,fetch=fetch))

    def test_signed_base64_github_package_decodes_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src';package(source,2)
            raw=archive(source)
            info={**self.info(raw),'minimum_bootstrap':2,'encoding':'base64','channel':'school',
                  'url':'https://raw.githubusercontent.com/owner/repo/school/.updates/package.b64'}
            self.assertEqual(updates.verify_envelope(self.envelope(info),self.public),info)
            encoded=base64.b64encode(raw)+b'\n'
            decoded=updates.download_package(info,fetch=lambda *args,**kwargs:encoded)
            self.assertEqual(decoded,raw)
            updates.unpack_release(decoded,info,root/'verified')
            with self.assertRaises(ValueError):
                updates.download_package(info,fetch=lambda *args,**kwargs:encoded[:10]+b'!'+encoded[11:])
            with self.assertRaises(ValueError):
                updates.unpack_release(decoded+b'bad',info,root/'corrupted')

    def test_payload_updates_never_rewrite_the_external_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src';target=root/'target';package(source,2);package(target,1)
            manifest_path=source/'_internal/update_manifest.json'
            manifest=json.loads(manifest_path.read_text());manifest['entrypoint']='external_launcher'
            del manifest['files']['蚕チェック.bat'];(source/'蚕チェック.bat').unlink()
            manifest_path.write_text(json.dumps(manifest))
            raw=archive(source);updates.unpack_release(raw,self.info(raw),root/'verified')
            (target/'蚕チェック.bat').write_bytes(b'original standalone starter')
            install_update(source,target,layout='one_click')
            updates.read_package(target)
            self.assertEqual((target/'蚕チェック.bat').read_bytes(),b'original standalone starter')

    def test_github_channel_mismatch_never_downloads_another_branches_program(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src';package(source)
            (source/'_internal/update_channel.json').write_text(json.dumps({'channel':'school','feed_url':'https://raw.githubusercontent.com/owner/repo/school/channel.json'}))
            (source/'_internal/update_public_key.json').write_text(json.dumps(self.public))
            calls=[]
            def fetch(url,limit,**kwargs):
                calls.append(url)
                return self.envelope({**self.info(),'channel':'parsonal'})
            self.assertIsNone(boot.choose_online(source,1,root,lambda _:None,fetch=fetch))
            self.assertEqual(len(calls),1)
