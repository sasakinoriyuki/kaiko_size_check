import hashlib,json,tempfile,unittest
from pathlib import Path
from zipfile import ZipFile
import cv2,numpy as np
from improvement_review import ReviewQueue,read_source,predictions


def fixture(data,name,value=100,crop=None):
    directory=Path(data)/'samples';directory.mkdir(exist_ok=True)
    image=np.full((80,120,3),value,np.uint8);_,png=cv2.imencode('.png',image);raw=png.tobytes()
    info={'source_sha256':hashlib.sha256(raw).hexdigest(),'model_info':{'instances':[{'contour_source_xy':[[10,10],[50,10],[50,35],[10,35]]}]}}
    if crop:info['crop_xyxy']=crop;info['source_resolution']=[300,200]
    path=directory/(name+'.zip')
    with ZipFile(path,'w') as z:z.writestr('source.png',raw);z.writestr('source.json',json.dumps(info))
    return path


class ImprovementTests(unittest.TestCase):
    def test_duplicate_images_have_one_queue_item_and_latest_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d,'a');fixture(d,'b');q=ReviewQueue(d);q.scan();self.assertEqual(len(q.items),1)
    def test_complete_advances_and_completed_image_is_not_queued_again(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d,'a');fixture(d,'b',110);q=ReviewQueue(d);q.scan();q.next();digest=q.current['sha256']
            q.complete();self.assertNotEqual(q.current['sha256'],digest)
            r=ReviewQueue(d);r.scan();self.assertEqual(len(r.items),1)
            self.assertEqual(json.loads(q.record_path(digest).read_text())['status'],'completed')
    def test_delete_undo_and_add_resume_without_reintroducing_rejected_object(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d,'a');q=ReviewQueue(d);q.scan();q.next();q.delete(0)
            self.assertEqual(len(q.rejected),1);q.undo();self.assertEqual(len(q.objects),1)
            q.delete(0);r=ReviewQueue(d);r.scan();r.next();self.assertEqual(r.objects,[])
            self.assertTrue(r.loaded_draft)
            r.add([[20,20],[40,20],[40,40]]);r.visibility(0,'partial');r.save(True)
            note=json.loads(r.record_path(r.current['sha256']).read_text())
            self.assertEqual(note['objects'][0]['review_status'],'human_confirmed')
            self.assertEqual(note['objects'][0]['visibility'],'partial')
            self.assertEqual(len(note['original_predictions']),1)
    def test_polygon_replacement_and_undo(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(d,'a');q=ReviewQueue(d);q.scan();q.next();before=q.objects[0].copy()
            q.replace(0,[[15,15],[45,15],[45,30]]);self.assertEqual(len(q.objects),1)
            self.assertEqual(q.objects[0]['id'],before['id']);q.undo();self.assertEqual(q.objects[0],before)
            with self.assertRaises(ValueError):q.add([[-1,0],[3,4],[4,5]])
    def test_crop_coordinates_are_translated_and_clipped(self):
        items=predictions({'crop_xyxy':[100,50,220,130],'model_info':{'instances':[{'contour_source_xy':[[110,60],[150,60],[150,85],[110,85]]}]}},(80,120,3))
        self.assertEqual(np.asarray(items[0]['polygon']).min(0).tolist(),[10,10])
    def test_export_contains_only_completed_reviews_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a=fixture(d,'a');fixture(d,'b',110);q=ReviewQueue(d);q.scan();q.next();raw,_,_=read_source(q.current['path'])
            q.complete();paths,count=q.export();self.assertEqual(count,1)
            with ZipFile(paths[0]) as z:
                note=json.loads(z.read(next(n for n in z.namelist() if n.endswith('review.json'))))
                self.assertNotIn('source_zip',note);self.assertEqual(note['status'],'completed')
                self.assertEqual(z.read(next(n for n in z.namelist() if n.endswith('source.png'))),raw)
    def test_empty_export_and_tampered_source_do_not_leave_temporary_zip(self):
        with tempfile.TemporaryDirectory() as d:
            path=fixture(d,'a');q=ReviewQueue(d);q.scan();q.next()
            with self.assertRaises(ValueError):q.export()
            q.save(True);path.write_bytes(b'invalid zip')
            with self.assertRaises(Exception):q.export()
            self.assertEqual(list(q.folder.glob('*.tmp')),[])
    def test_export_splits_to_bounded_parts(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):fixture(d,str(i),100+i)
            q=ReviewQueue(d);q.scan();q.next()
            for _ in range(3):q.complete()
            paths,count=q.export(max_bytes=6000)
            self.assertEqual(count,3);self.assertGreater(len(paths),1)
            self.assertTrue(all(p.stat().st_size<6000 for p in paths))

if __name__=='__main__':unittest.main()
