import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from archive_batch import ArchiveBatch,Cancelled,CommandRunner,list_archives,youtube_target,append_row,read_sample_zip
from archive_batch import analysis_fingerprint,LEGACY_TEXTURE_VETO_SHA256
from instance_geometry import measure_proposals
from instance_counting import assign_instance_ids
from archive_history import completion_key,load_completion_keys


VIDEO='0rijis5y8c0'
CHANNEL='UC'+'a'*22


class Engine:
    metadata={'model_version':'unit-fixture'};model_sha256='fixture-sha'
    def __init__(self):self.calls=0
    def analyze(self,image,settings):
        self.calls+=1
        mask=np.zeros((30,140),np.uint8);cv2.line(mask,(15,15),(125,15),1,16)
        cs=measure_proposals([{'mask':mask,'offset':(30,30),'score':.95}],image.shape)
        counts=assign_instance_ids(cs)
        self.last_info={**counts,'measurable_count':counts['measured_count'],'visible_segment_count':0}
        return cs,'fixture'


def extraction(url,count,skip_indices=None,**kwargs):
    for i in range(1,count+1):
        if i in (skip_indices or ()):continue
        yield np.full((240,320,3),80+i,np.uint8),{'sample_index':i,'video_url':url,'requested_video_seconds':10*i}


def entries(count=1):
    return [{'id':VIDEO if i==0 else f'fakevid{i:04}', 'url':'https://www.youtube.com/watch?v='+VIDEO} for i in range(count)]


class BatchTests(unittest.TestCase):
    def setUp(self):
        patcher=patch('archive_batch.load_completion_keys',return_value=set());patcher.start();self.addCleanup(patcher.stop)

    def test_dark_input_after_a_body_records_zero_counts_in_csv_and_zip(self):
        from instance_detector import LocalInstanceSegmenter
        e=LocalInstanceSegmenter.__new__(LocalInstanceSegmenter)
        e.device='test';e.metadata={'model_version':'blank-transition-fixture'};e.model_sha256='fixture'
        mask=np.zeros((30,210),np.uint8);cv2.line(mask,(15,15),(195,15),1,16)
        e.predict=lambda *a,**k:[{'mask':mask,'offset':(30,30),'box':(30,30,240,60),'score':.95} for _ in range(8)]
        with tempfile.TemporaryDirectory() as folder:
            batch=ArchiveBatch(folder,e)
            bright=batch.analyze_image(np.full((240,320,3),150,np.uint8),{},VIDEO,1,2)
            dark=batch.analyze_image(np.zeros((240,320,3),np.uint8),{},VIDEO,2,2)
            self.assertEqual(bright['estimated_count'],1)
            self.assertEqual(dark['estimated_count'],0)
            self.assertEqual(dark['full_length_count'],0)
            _,info=read_sample_zip(Path(folder)/dark['sample_zip'])
            self.assertEqual(info['model_info']['estimated_count'],0)
            self.assertEqual(info['model_info']['instances'],[])
            with Path(folder,'results.csv').open(encoding='utf-8-sig',newline='') as f:
                self.assertEqual([r['estimated_count'] for r in csv.DictReader(f)],['1','0'])

    def test_urls_cannot_select_other_hosts_files_or_commands(self):
        for value in ['file:///etc/passwd','https://evil.example/watch?v='+VIDEO,'https://youtube.com.evil.example/'+VIDEO,
                      'https://a:b@youtube.com/watch?v='+VIDEO,'https://youtube.com/watch?v=../../x',
                      'https://youtube.com:8080/watch?v='+VIDEO,'--exec whoami']:
            with self.subTest(value=value),self.assertRaises(ValueError):youtube_target(value)
        self.assertEqual(youtube_target('https://youtu.be/'+VIDEO),'https://www.youtube.com/watch?v='+VIDEO)
        self.assertTrue(youtube_target(CHANNEL).endswith('/streams'))

    def test_all_archives_are_listed_with_live_and_invalid_entries_excluded(self):
        listing={'id':CHANNEL,'entries':[{'id':VIDEO},{'id':VIDEO},{'id':'bad'},
                   {'id':'abc12345678','live_status':'is_live'},{'id':'xyz12345678','live_status':'is_upcoming'},
                   {'id':'end12345678','live_status':'was_live'}]}
        def runner(command,**kwargs):
            obj=listing if '--flat-playlist' in command else {'channel_id':CHANNEL}
            self.assertNotIn('--cookies',command)
            return subprocess.CompletedProcess(command,0,json.dumps(obj).encode(),b'')
        channel,rows=list_archives(VIDEO,runner=runner,yt_command=['fixture'])
        self.assertEqual(channel,CHANNEL);self.assertEqual(len(rows),2)

    def test_source_zip_csv_and_receipt_are_created_by_the_shared_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prior=root/'measurements.csv';prior.write_bytes(b'keep old private measurements')
            engine=Engine();batch=ArchiveBatch(root/'archive_batch',engine,extractor=extraction)
            result=batch.run(count=2,entries=entries(),channel_id=CHANNEL)
            self.assertEqual(engine.calls,2);self.assertEqual(result['saved_images'],2)
            self.assertFalse(result['automatic_training']);self.assertFalse(result['production_model_modified'])
            files=list((batch.folder/'samples').glob('*.zip'));self.assertEqual(len(files),2)
            with ZipFile(files[0]) as z:
                image=cv2.imdecode(np.frombuffer(z.read(next(n for n in z.namelist() if n.endswith('.png'))),np.uint8),cv2.IMREAD_COLOR)
                report=json.loads(z.read(next(n for n in z.namelist() if n.endswith('.json'))))
            self.assertEqual(image.shape,(240,320,3));self.assertEqual(np.unique(image).size,1)
            self.assertIsNone(report['pixels_per_mm']);self.assertEqual(report['candidate_status'],'unverified')
            self.assertIn('contour_source_xy',report['model_info']['instances'][0])
            with (batch.folder/'results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['unit'],'px')
            self.assertEqual(prior.read_bytes(),b'keep old private measurements')

    def test_completed_archives_are_not_downloaded_or_analyzed_again(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();first=ArchiveBatch(folder,e,extractor=extraction);first.run(count=2,entries=entries())
            def fail(*args,**kwargs):self.fail('completed video redownloaded')
            result=ArchiveBatch(folder,e,extractor=fail).run(count=2,entries=entries())
            self.assertEqual(result['skipped_completed'],1);self.assertEqual(e.calls,2)

    def test_exported_zip_round_trips_original_image_without_using_prediction_as_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            b=ArchiveBatch(folder,Engine(),extractor=extraction);b.run(count=1,entries=entries())
            image,info=read_sample_zip(next(Path(folder,'samples').glob('*.zip')))
            expected,_=next(extraction(entries()[0]['url'],1))
            np.testing.assert_array_equal(image,expected)
            self.assertEqual(info['candidate_status'],'unverified')

    def test_zip_refuses_multiple_sources_or_a_tampered_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder,'bad.zip');_,png=cv2.imencode('.png',np.zeros((20,20,3),np.uint8))
            with ZipFile(path,'w') as z:z.writestr('image.png',png.tobytes());z.writestr('info.json',json.dumps({'source_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'ハッシュ'):read_sample_zip(path)
            with ZipFile(path,'a') as z:z.writestr('other.png',png.tobytes())
            with self.assertRaisesRegex(ValueError,'元画像1枚'):read_sample_zip(path)

    def test_cancelled_download_resumes_only_missing_sample_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            cancelled=threading.Event();e=Engine()
            def partial(*args,**kwargs):
                yield next(extraction(*args,**kwargs));cancelled.set()
            first=ArchiveBatch(folder,e,extractor=partial,cancelled=cancelled)
            r=first.run(count=3,entries=entries());self.assertEqual(r['status'],'cancelled')
            seen=[]
            def resume(*args,**kwargs):
                seen.append(kwargs['skip_indices']);yield from extraction(*args,**kwargs)
            r=ArchiveBatch(folder,e,extractor=resume).run(count=3,entries=entries())
            self.assertEqual(seen,[{1}]);self.assertEqual(e.calls,3)
            with Path(folder,'results.csv').open(encoding='utf-8-sig',newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),3)

    def test_completed_video_is_not_redownloaded_just_because_an_export_is_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();b=ArchiveBatch(folder,e,extractor=extraction);b.run(count=1,entries=entries())
            next(Path(folder,'samples').glob('*.zip')).unlink()
            result=ArchiveBatch(folder,e,extractor=extraction).run(count=1,entries=entries())
            self.assertEqual(result['saved_images'],0);self.assertEqual(e.calls,1)
            self.assertEqual(result['history_only_videos'],1)
            with Path(folder,'results.csv').open(encoding='utf-8-sig',newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),1)

    def test_changed_engine_is_not_mistaken_for_a_completed_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();ArchiveBatch(folder,e,extractor=extraction).run(count=1,entries=entries())
            e.metadata={'model_version':'different'}
            def forbidden(*a,**k):self.fail('model update redownloaded existing source pixels')
            r=ArchiveBatch(folder,e,extractor=forbidden).run(count=1,entries=entries())
            self.assertEqual(r['saved_images'],1);self.assertEqual(e.calls,2)
            self.assertEqual(r['reused_images'],1);self.assertEqual(r['downloaded_images'],0)

    def test_same_bundled_texture_keeps_existing_completed_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();old=ArchiveBatch(folder,e,extractor=extraction)
            old.run(count=1,entries=entries())
            e.metadata={**e.metadata,'texture_veto_sha256':LEGACY_TEXTURE_VETO_SHA256}
            self.assertEqual(analysis_fingerprint(e),old.fingerprint)
            def forbidden(*a,**k):self.fail('unchanged texture downloaded or recomputed a source')
            result=ArchiveBatch(folder,e,extractor=forbidden).run(count=1,entries=entries())
            self.assertEqual(result['skipped_completed'],1);self.assertEqual(e.calls,1)

    def test_changed_texture_reanalyzes_cached_pixels_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();ArchiveBatch(folder,e,extractor=extraction).run(count=1,entries=entries())
            e.metadata={**e.metadata,'texture_veto_sha256':'a'*64}
            def forbidden(*a,**k):self.fail('texture update redownloaded source pixels')
            result=ArchiveBatch(folder,e,extractor=forbidden).run(count=1,entries=entries())
            self.assertEqual(e.calls,2);self.assertEqual(result['reused_images'],1)
            self.assertEqual(result['downloaded_images'],0)
            result=ArchiveBatch(folder,e,extractor=forbidden).run(count=1,entries=entries())
            self.assertEqual(result['skipped_completed'],1);self.assertEqual(e.calls,2)
            with Path(folder,'results.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['source_sha256'],rows[1]['source_sha256'])

    def test_school_receipts_still_skip_sources_after_texture_update(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();e.metadata={**e.metadata,'texture_veto_sha256':'a'*64}
            def forbidden(*a,**k):self.fail('new veto redownloaded completed home sources on school PC')
            result=ArchiveBatch(folder,e,extractor=forbidden,
                                completed_keys={completion_key(VIDEO,5)}).run(count=5,entries=entries())
            self.assertEqual(e.calls,0);self.assertEqual(result['skipped_completed'],1)
            self.assertEqual(result['downloaded_images'],0)

    def test_school_can_skip_returned_completion_history_without_local_images(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine()
            def forbidden(*a,**k):self.fail('completed home video redownloaded on school profile')
            result=ArchiveBatch(folder,e,extractor=forbidden,completed_keys={completion_key(VIDEO,5)}).run(count=5,entries=entries())
            self.assertEqual(result['skipped_completed'],1);self.assertEqual(result['history_only_videos'],1)
            self.assertEqual(e.calls,0);self.assertFalse(Path(folder,'results.csv').exists())
            # The receipt remains local even if a later package has no seed list.
            result=ArchiveBatch(folder,e,extractor=forbidden,completed_keys=set()).run(count=5,entries=entries())
            self.assertEqual(result['skipped_completed'],1)

    def test_imported_receipts_do_not_skip_unrelated_new_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            result=ArchiveBatch(folder,Engine(),extractor=extraction,completed_keys={completion_key('other000001',1)}).run(count=1,entries=entries())
            self.assertEqual(result['saved_images'],1);self.assertEqual(result['skipped_completed'],0)

    def test_more_samples_reuse_the_source_positions_which_already_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();ArchiveBatch(folder,e,extractor=extraction).run(count=5,entries=entries())
            seen=[]
            def collect(*args,**kwargs):seen.append(kwargs['skip_indices']);yield from extraction(*args,**kwargs)
            result=ArchiveBatch(folder,e,extractor=collect).run(count=10,entries=entries())
            self.assertEqual(seen,[{1,10}]);self.assertEqual(result['reused_images'],2)
            self.assertEqual(result['downloaded_images'],8)

    def test_completed_receipts_do_not_claim_that_source_images_were_transferred(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder,'history.json');key=completion_key(VIDEO,5)
            p.write_text(json.dumps({'schema':1,'keys':[key]}))
            self.assertEqual(load_completion_keys(p),{key})
            self.assertNotIn(VIDEO,p.read_text());self.assertNotIn('estimated_count',p.read_text())
            p.write_text(json.dumps({'schema':1,'keys':['not a hash']}))
            with self.assertRaises(ValueError):load_completion_keys(p)

    def test_two_failures_stop_and_are_retryable(self):
        def fail(*args,**kwargs):raise RuntimeError('fixture video blocked')
        with tempfile.TemporaryDirectory() as folder:
            r=ArchiveBatch(folder,Engine(),extractor=fail).run(count=1,entries=entries(3))
            self.assertEqual(len(r['errors']),2);self.assertEqual(r['status'],'completed_with_errors')
            r=ArchiveBatch(folder,Engine(),extractor=extraction).run(count=1,entries=entries(3))
            self.assertEqual(r['completed_videos'],3)

    def test_unknowns_enter_review_queue_and_never_train_or_replace_a_model(self):
        class Empty(Engine):
            def analyze(self,image,settings):
                self.last_info={'estimated_count':0,'measurable_count':0};return [],'fixture'
        with tempfile.TemporaryDirectory() as folder:
            r=ArchiveBatch(folder,Empty(),extractor=extraction).run(count=1,entries=entries())
            review=json.loads(Path(folder,'review_queue.json').read_text())
            self.assertEqual(r['review_images'],1);self.assertIn('no_full_length',review['samples'][0]['review_reasons'])
            self.assertEqual(review['status'],'unverified_predictions')

    def test_csv_schema_error_does_not_append_wrong_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder,'results.csv');p.write_bytes(b'unknown,header\nkeep,data\n')
            before=p.read_bytes()
            with self.assertRaises(ValueError):append_row(p,{})
            self.assertEqual(p.read_bytes(),before)

    def test_space_guard_stops_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Engine();b=ArchiveBatch(folder,e,max_bytes=0)
            with self.assertRaises(OSError):b.analyze_image(np.zeros((20,20,3),np.uint8),{},VIDEO,1,1)
            self.assertEqual(e.calls,0)

    def test_runner_cancels_only_its_own_process(self):
        stop=threading.Event();timer=threading.Timer(.15,stop.set);timer.start()
        try:
            with self.assertRaises(Cancelled):CommandRunner(stop)([sys.executable,'-c','import time;time.sleep(15)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=3)
        finally:timer.cancel()

    def test_portable_tool_install_has_no_user_or_system_target(self):
        from archive_tools import prepare_tools
        import os
        archive=io.BytesIO()
        name='deno.exe' if os.name=='nt' else 'deno'
        with ZipFile(archive,'w') as z:z.writestr(name,b'fixture-only-not-executed')
        raw=archive.getvalue();calls=[]
        def fetch(url,limit,**kwargs):return hashlib.sha256(raw).hexdigest().encode() if url.endswith('.sha256sum') else raw
        def missing(*args,**kwargs):raise RuntimeError('fixture missing')
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ),patch('archive_tools.sys.path',list(sys.path)),patch('archive_tools.find_tool',side_effect=missing),patch('archive_tools._run',side_effect=lambda command,*a:calls.append(command)),patch('archive_tools.importlib.import_module') as module:
            module.return_value.get_ffmpeg_exe.return_value='fixture-ffmpeg'
            yt,ffmpeg=prepare_tools(folder,object(),fetch=fetch)
            self.assertIn('--target',calls[0]);self.assertNotIn('--user',calls[0]);self.assertIn('--no-cache-dir',calls[0])
            # Windows temporary paths may pass through a junction or short
            # alias. Compare the exact canonical portable destinations.
            expected_root=Path(folder).resolve()/'runtime/archive_tools'
            self.assertEqual(Path(calls[0][calls[0].index('--target')+1]),expected_root)
            self.assertEqual(Path(os.environ['TEMP']),expected_root/'tmp')
            self.assertEqual(Path(os.environ['DENO_DIR']),expected_root/'deno_cache')
            self.assertEqual(ffmpeg,['fixture-ffmpeg']);self.assertIn('--js-runtimes',yt)
            self.assertTrue(Path(folder,'runtime/archive_tools',name).exists())

    def test_official_windows_and_unix_deno_checksum_formats_are_supported(self):
        from archive_tools import asset_checksum
        digest='a9'*32
        self.assertEqual(asset_checksum((digest+'  deno.zip\n').encode()),digest)
        self.assertEqual(asset_checksum(('\r\nAlgorithm : SHA256\r\nHash      : '+digest.upper()+'\r\nPath : C:\\deno.zip\r\n').encode()),digest)
        with self.assertRaises(ValueError):asset_checksum((digest+' '+digest).encode())

    def test_deno_asset_downloader_does_not_accept_unrelated_urls(self):
        from archive_tools import download_deno_asset
        for url in ['https://evil.example/deno.zip','http://github.com/denoland/deno/releases/latest/download/x','file:///tmp/deno.zip']:
            with self.assertRaises(ValueError):download_deno_asset(url,10)


if __name__=='__main__':unittest.main()
