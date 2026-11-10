import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import threading
from zipfile import ZipFile,ZIP_STORED

from archive_review import Example,infer_instar,read_examples,select_examples,make_review_exports,launch_review_export
from archive_batch import CSV_FIELDS


def row(i,stage=1,count=3,full=1,uncertain=2,video=None):
    return {'sample_id':f'{i:024x}','video_id':video or f'fixture{i:04}',
            'requested_video_seconds':'30','unit':'px','estimated_count':str(count),
            'full_length_count':str(full),'full_length_mean':'100' if full else '',
            'visible_segment_count':'0','uncertain_regions':str(uncertain),
            'review_reasons':'unverified','sample_zip':f'samples/{i:024x}.zip',
            'source_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
            'model_version':'fixture','model_sha256':'f'*64}


def fixture(folder,i,stage=1,**kwargs):
    item=row(i,stage,**kwargs);raw=(f'synthetic fixture source {i}'.encode()*80)
    item['source_sha256']=hashlib.sha256(raw).hexdigest()
    path=folder/item['sample_zip'];path.parent.mkdir(exist_ok=True)
    info={'sample_id':item['sample_id'],'source_sha256':item['source_sha256'],
          'collection':{'video_title':f'蚕{stage}齢 飼育' if stage else 'no age in title'}}
    with ZipFile(path,'w',ZIP_STORED) as z:
        z.writestr('source.png',raw);z.writestr('source.json',json.dumps(info))
    return item


def write_csv(folder,rows):
    with (folder/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS);writer.writeheader();writer.writerows(rows)
    (folder/'review_queue.json').write_text(json.dumps({'schema':1,'samples':rows}))


class ReviewExportTests(unittest.TestCase):
    def test_instar_hints_support_japanese_and_fullwidth_digits_without_guessing_age_from_size(self):
        for title,expected in [('蚕２齢（8日目）',2),('五齢の飼育',5),('蚕１齡',1),('孵化直後',1),('孵化から24日目',None),('20160527_蚕',None)]:
            self.assertEqual(infer_instar(title),expected)

    def test_selection_covers_each_stage_and_includes_measurable_comparisons(self):
        examples=[]
        for stage in range(1,6):
            for n in range(3):
                item=row(stage*10+n,count=0 if n==0 else 4,full=0 if n==0 else 3)
                examples.append(Example(item,Path('fixture'),stage,''))
        chosen=select_examples(examples,10)
        self.assertEqual(len(chosen),10);self.assertEqual({e.instar for e in chosen},{1,2,3,4,5})
        self.assertEqual(sum(e.full==0 for e in chosen),5)
        self.assertEqual(len({e.video for e in chosen}),10)

    def test_same_video_and_same_source_pixels_are_not_selected_twice(self):
        a=row(1,video='samevid0001');b=row(2,video='samevid0001');c=row(3)
        c['source_sha256']=a['source_sha256']
        selected=select_examples([Example(r,Path('x'),None,'') for r in [a,b,c]],12)
        self.assertEqual(len({e.video for e in selected}),len(selected))
        self.assertEqual(len({e.source_hash for e in selected}),len(selected))

    def test_one_hundred_samples_prioritize_different_videos_and_preserve_all_inputs(self):
        examples=[Example(row(i+1,count=0 if i%3==0 else 4,full=0 if i%3==0 else 3),Path('fixture'),1+i%5,'') for i in range(128)]
        chosen=select_examples(examples)
        self.assertEqual(len(chosen),100);self.assertEqual(len({e.video for e in chosen}),100)
        self.assertEqual(len(examples),128);self.assertEqual({e.instar for e in chosen},{1,2,3,4,5})

    def test_additional_frames_can_be_selected_after_each_video_is_represented(self):
        examples=[Example(row(i+1,video=f'fixture{i//5:04}'),Path('fixture'),None,'') for i in range(15)]
        chosen=select_examples(examples,10)
        self.assertEqual(len(chosen),10);self.assertEqual(len({e.video for e in chosen[:3]}),3)

    def test_missing_sources_and_corrupt_archives_are_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);a=fixture(folder,1);b=fixture(folder,2);c=fixture(folder,3)
            (folder/b['sample_zip']).unlink();(folder/c['sample_zip']).write_bytes(b'corrupt archive')
            write_csv(folder,[a,b,c]);examples,skipped=read_examples(folder)
            self.assertEqual(len(examples),1);self.assertEqual(len(skipped),2)

    def test_paths_cannot_escape_the_samples_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=row(1);item['sample_zip']='../../private.zip';write_csv(folder,[item])
            with self.assertRaisesRegex(ValueError,'パス'):read_examples(folder)

    def test_linked_sources_cannot_be_copied_outside_the_selected_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);target=folder/'outside.zip'
            (folder/item['sample_zip']).rename(target);(folder/item['sample_zip']).symlink_to(target)
            write_csv(folder,[item]);examples,skipped=read_examples(folder)
            self.assertFalse(examples);self.assertEqual(skipped[0]['reason'],'missing_or_linked_source_zip')

    def test_bundle_preserves_source_bytes_and_records_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);rows=[fixture(folder,i,stage=i) for i in range(1,6)]
            write_csv(folder,rows);before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            packs=make_review_exports(folder,limit=5)
            self.assertEqual(len(packs),1)
            with ZipFile(packs[0]) as archive:
                manifest=json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['selected_count'],5)
                self.assertEqual(manifest['status'],'unverified_predictions_not_training_truth')
                for item in rows:self.assertEqual(archive.read(item['sample_zip']),(folder/item['sample_zip']).read_bytes())
                self.assertEqual(archive.read('results.csv'),before[folder/'results.csv'])
            self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))
            self.assertFalse(any(p.suffix in {'.pt','.bat','.pem'} for p in folder.rglob('*')))

    def test_parts_stay_below_limit_and_each_source_is_copied_once(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);rows=[fixture(folder,i) for i in range(1,13)];write_csv(folder,rows)
            packs=make_review_exports(folder,limit=12,part_limit=85000)
            self.assertGreater(len(packs),1);seen=[]
            for pack in packs:
                self.assertLessEqual(pack.stat().st_size,85000)
                with ZipFile(pack) as z:seen.extend(n for n in z.namelist() if n.startswith('samples/'))
            self.assertEqual(len(seen),12);self.assertEqual(len(set(seen)),12)

    def test_full_default_export_contains_one_hundred_sources_and_bounded_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);rows=[fixture(folder,i,stage=1+i%5) for i in range(1,129)]
            write_csv(folder,rows);packs=make_review_exports(folder)
            seen=[]
            for pack in packs:
                self.assertLessEqual(pack.stat().st_size,24*1024**2)
                with ZipFile(pack) as z:
                    manifest=json.loads(z.read('manifest.json'));self.assertEqual(manifest['selected_count'],100)
                    seen.extend(n for n in z.namelist() if n.startswith('samples/'))
            self.assertEqual(len(seen),100);self.assertEqual(len(set(seen)),100)

    def test_tampered_pixels_are_not_exported_as_the_recorded_source(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);write_csv(folder,[item])
            path=folder/item['sample_zip']
            with ZipFile(path) as z:info=z.read('source.json')
            with ZipFile(path,'w') as z:z.writestr('source.png',b'tampered');z.writestr('source.json',info)
            with self.assertRaisesRegex(ValueError,'ハッシュ'):make_review_exports(folder)
            self.assertFalse((folder/'review_exports').exists())

    def test_adding_a_private_file_to_source_zip_prevents_its_export(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);write_csv(folder,[item])
            with ZipFile(folder/item['sample_zip'],'a') as z:z.writestr('private.pem','private fixture')
            with self.assertRaisesRegex(ValueError,'使える検証ZIP'):make_review_exports(folder)

    def test_duplicate_rows_are_deduplicated_but_conflicting_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);write_csv(folder,[item,item.copy()])
            self.assertEqual(len(read_examples(folder)[0]),1)
            altered={**item,'estimated_count':'4'};write_csv(folder,[item,altered])
            with self.assertRaisesRegex(ValueError,'一致'):read_examples(folder)

    def test_model_update_review_prefers_latest_result_for_the_same_original(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);old=fixture(folder,1,count=0,full=0);new=fixture(folder,2)
            with ZipFile(folder/old['sample_zip']) as archive:original=archive.read('source.png')
            new['source_sha256']=old['source_sha256'];new['model_version']='updated-veto'
            info={'sample_id':new['sample_id'],'source_sha256':new['source_sha256'],
                  'collection':{'video_title':'蚕1齢 飼育'},'model_info':{'model_version':'updated-veto'}}
            with ZipFile(folder/new['sample_zip'],'w') as archive:
                archive.writestr('source.png',original);archive.writestr('source.json',json.dumps(info))
            write_csv(folder,[old,new]);before={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            examples,skipped=read_examples(folder)
            self.assertFalse(skipped);self.assertEqual([e.sid for e in examples],[new['sample_id']])
            packs=make_review_exports(folder)
            with ZipFile(packs[0]) as archive:
                sources=[name for name in archive.namelist() if name.startswith('samples/')]
                self.assertEqual(sources,[new['sample_zip']])
                self.assertEqual(archive.read('results.csv'),before[folder/'results.csv'])
            self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))

    def test_missing_latest_export_can_fall_back_to_an_older_available_original(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);old=fixture(folder,1);new=fixture(folder,2)
            new['source_sha256']=old['source_sha256'];(folder/new['sample_zip']).unlink()
            write_csv(folder,[old,new]);examples,skipped=read_examples(folder)
            self.assertEqual([e.sid for e in examples],[old['sample_id']])
            self.assertEqual(skipped[0]['sample_id'],new['sample_id'])

    def test_partly_appended_final_csv_row_is_not_misread_as_a_saved_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);write_csv(folder,[item])
            with (folder/'results.csv').open('ab') as f:f.write(b'partly-written-row,not-finished')
            examples,_=read_examples(folder)
            self.assertEqual(len(examples),1);self.assertEqual(examples[0].sid,item['sample_id'])

    def test_cancellation_preserves_inputs_and_creates_no_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);item=fixture(folder,1);write_csv(folder,[item])
            event=threading.Event();event.set();before=(folder/'results.csv').read_bytes()
            with self.assertRaises(InterruptedError):make_review_exports(folder,cancelled=event)
            self.assertEqual((folder/'results.csv').read_bytes(),before)
            self.assertFalse((folder/'review_exports').exists())

    def test_launch_uses_the_selected_portable_folder_without_a_shell(self):
        with tempfile.TemporaryDirectory() as folder,patch('archive_review.subprocess.Popen') as popen:
            target=Path(folder)/'蚕 data & spaces'
            launch_review_export(target)
            args,kwargs=popen.call_args
            self.assertEqual(args[0][-1],str(target.resolve()))
            self.assertEqual(args[0][-2],'--folder');self.assertFalse(kwargs.get('shell',False))


if __name__=='__main__':unittest.main()
