import copy,hashlib,json,tempfile,unittest,shutil
from pathlib import Path
from zipfile import ZipFile
import cv2,numpy as np
from improvement_review import ReviewQueue,atomic
from test_improvement_review import fixture
from review_learning import prepare,import_exports,masks,assign_splits,compare_annotations,train_candidate


def reviewed(data,name,value,video='abcdefghijk',date='20261101'):
    fixture(data,name,value);q=ReviewQueue(data);q.scan();q.next()
    # Explicit false proposal and accepted body, spatially separated.
    q.original.append({'id':'false','polygon':[[70,10],[110,10],[110,35],[70,35]],'origin':'model','visibility':'unknown','review_status':'unreviewed'})
    q.rejected=[copy.deepcopy(q.original[-1])]
    q.info['collection']={'video_id':video,'upload_date':date}
    q.save(True);path=q.record_path(q.current['sha256'])
    return path,json.loads(path.read_text())


class ReviewLearningTests(unittest.TestCase):
    def test_unknown_pixels_and_positive_over_rejected_region(self):
        r={'objects':[{'polygon':[[10,10],[30,10],[30,30],[10,30]]}],
           'rejected_predictions':[{'polygon':[[0,0],[50,0],[50,50],[0,50]]}]}
        labels=masks(r,(80,80,3))
        self.assertEqual(labels[20,20],1);self.assertEqual(labels[8,20],255)
        self.assertEqual(labels[40,40],0);self.assertEqual(labels[70,70],255)
    def test_same_video_or_day_never_crosses_splits(self):
        def item(v,d):return {'record':{'collection':{'video_id':v,'upload_date':d}}}
        data=[item('aaaaaaaaaaa','20260101'),item('aaaaaaaaaaa','20260102'),item('bbbbbbbbbbb','20260102'),item('ccccccccccc','20260103')]
        groups,reg=assign_splits(data,{'keys':{}})
        self.assertEqual(groups['video:aaaaaaaaaaa'],groups['video:bbbbbbbbbbb'])
        self.assertNotEqual(groups['video:aaaaaaaaaaa'],groups['video:ccccccccccc'])
        again,reg2=assign_splits(data+[item('ddddddddddd','20260104')],reg)
        self.assertTrue(all(again[k]==v for k,v in groups.items()))
    def test_historical_split_conflicts_are_quarantined(self):
        data=[{'record':{'collection':{'video_id':'aaaaaaaaaaa','upload_date':'20260101'}}}]
        g,r=assign_splits(data,{'keys':{'video:aaaaaaaaaaa':'train','date:20260101':'validation'}})
        self.assertEqual(set(g.values()),{'quarantine'})
    def test_unknown_sources_do_not_create_a_fake_holdout(self):
        g,r=assign_splits([{'record':{}},{'record':{'collection':{'video_id':'aaaaaaaaaaa'}}}],{'keys':{}})
        self.assertEqual(set(g.values()),{'pending'})
    def test_change_analysis_and_missing_baseline(self):
        a={'id':'a','polygon':[[0,0],[10,0],[10,10],[0,10]]}
        b={'id':'b','polygon':[[20,20],[30,20],[30,30],[20,30]]}
        r={'objects':[a,b],'original_predictions':[a]}
        report=compare_annotations(r);self.assertEqual(report['matched_at_iou_0_5'],1)
        self.assertEqual(report['unmatched_reviewed'],1);self.assertEqual(report['added_objects'],1)
        empty=compare_annotations({'objects':[a],'original_predictions':[]})
        self.assertIsNone(empty['unmatched_reviewed'])
    def test_dataset_has_verified_hashes_and_excludes_drafts(self):
        with tempfile.TemporaryDirectory() as d:
            reviewed(d,'one',90,'aaaaaaaaaaa','20260101');reviewed(d,'two',110,'bbbbbbbbbbb','20260102')
            fixture(d,'draft',120)
            out,report=prepare(d);config=json.loads((out/'dataset.json').read_text())
            self.assertEqual(report['images'],2);self.assertEqual(report['train_images'],1);self.assertEqual(report['validation_images'],1)
            self.assertEqual(report['deleted_objects'],2)
            self.assertEqual({f['split'] for f in config['frames']},{'train','validation'})
            for f in config['frames']:
                self.assertEqual(hashlib.sha256((out/f['image']).read_bytes()).hexdigest(),f['image_sha256'])
                self.assertTrue(set(np.unique(cv2.imread(str(out/f['labels']),0)))=={0,1,255})
            self.assertEqual(prepare(d)[0],out)
    def test_import_is_idempotent_and_preserves_conflicting_local_review(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            reviewed(a,'one',90);q=ReviewQueue(a);packs,n=q.export()
            self.assertEqual(import_exports(b,packs)['imported'],1)
            self.assertEqual(import_exports(b,packs)['duplicates'],1)
            p=next((Path(b)/'improvement/reviews').glob('*.json'));r=json.loads(p.read_text());r['objects'][0]['visibility']='partial';atomic(p,r)
            before=p.read_bytes();self.assertEqual(import_exports(b,packs)['conflicts'],1);self.assertEqual(p.read_bytes(),before)
            self.assertEqual(len(list((Path(b)/'improvement/import_conflicts').glob('*.png'))),1)
    def test_import_refuses_drafts_wrong_hash_and_archive_paths(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'bad.zip'
            with ZipFile(z,'w') as f:f.writestr('../source.png',b'data')
            with self.assertRaises(ValueError):import_exports(Path(d)/'target',[z])
    def test_moved_portable_folder_can_prepare_and_export_without_fetching(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old PC';new=Path(d)/'学校 PC'
            old.mkdir()
            reviewed(old,'one',90,'aaaaaaaaaaa','20260101');reviewed(old,'two',110,'bbbbbbbbbbb','20260102')
            shutil.copytree(old,new);shutil.rmtree(old)
            out,r=prepare(new);self.assertEqual(r['images'],2)
            packs,n=ReviewQueue(new).export();self.assertEqual(n,2)
            self.assertTrue(all(p.is_relative_to(new) for p in packs))
    def test_pending_sources_cannot_train(self):
        with tempfile.TemporaryDirectory() as d:
            reviewed(d,'one',90);out,report=prepare(d)
            self.assertEqual(report['pending_images'],1)
            with self.assertRaisesRegex(ValueError,'2組'):train_candidate(out,'missing.pt',steps=1)
    def test_trial_training_preserves_live_model_and_keeps_holdout_separate(self):
        import torch
        from silkworm_texture import create_texture_model,ARCHITECTURE
        with tempfile.TemporaryDirectory() as d:
            reviewed(d,'one',90,'aaaaaaaaaaa','20260101');reviewed(d,'two',110,'bbbbbbbbbbb','20260102')
            out,report=prepare(d);base=Path(d)/'base.pt'
            torch.save({'metadata':{'architecture':ARCHITECTURE,'model_version':'fixture'},'state_dict':create_texture_model().state_dict()},base)
            raw=base.read_bytes();run,result=train_candidate(out,base,steps=2)
            self.assertEqual(base.read_bytes(),raw);self.assertFalse(result['model_applied'])
            self.assertTrue((run/'candidate.pt').is_file())
            meta=torch.load(run/'candidate.pt',weights_only=True)['metadata']
            self.assertFalse(set(meta['review_training_sources'])&set(meta['review_validation_sources']))

if __name__=='__main__':unittest.main()
