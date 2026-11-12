"""Instance updates must keep source coordinates, honest units and private data."""
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ai_measurement import AIJob,AIWorker
from instance_detector import ensure_instance_model,automatic_camera_roi,neutral_boost,LocalInstanceSegmenter,tile_positions,thin_recheck_seeds
from instance_geometry import measure_proposals
from measurement import Settings
from test_app import headless_app,FakeValue,read_rows

class InstanceTests(unittest.TestCase):
    def test_asset_checks_hash_before_loading_and_reuses_verified_cache(self):
        raw=b'example pinned test asset'
        with tempfile.TemporaryDirectory() as folder,patch('instance_detector.MODEL_BYTES',len(raw)),patch('instance_detector.MODEL_SHA256',hashlib.sha256(raw).hexdigest()):
            path=Path(folder)/'models/test.pt';calls=[]
            def fetch(*args,**kwargs):calls.append(args);return raw
            ensure_instance_model(path,fetch);ensure_instance_model(path,lambda *a,**k:self.fail('cache downloaded twice'))
            self.assertEqual(path.read_bytes(),raw);self.assertEqual(len(calls),1)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):ensure_instance_model(path,lambda *a,**k:b'incorrect')
            self.assertEqual(path.read_bytes(),b'corrupt')

    def test_asset_rejects_linked_output_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'real').mkdir();(root/'link').symlink_to(root/'real',target_is_directory=True)
            with self.assertRaises(ValueError):ensure_instance_model(root/'link/test.pt',lambda *a,**k:self.fail('network used'))

    def test_obs_sidebar_is_detected_without_cropping_a_plain_camera(self):
        image=np.full((500,1000,3),160,np.uint8)
        self.assertEqual(automatic_camera_roi(image),(0,0,1000,500))
        image[:,770:]=20
        roi=automatic_camera_roi(image)
        self.assertLess(abs(roi[2]-770),18)
        image[:]=20
        self.assertEqual(automatic_camera_roi(image),(0,0,1000,500))

    def test_boost_preserves_image_coordinates_and_input(self):
        image=np.zeros((31,47,3),np.uint8);image[10:20,20:30]=60;copy=image.copy()
        boosted=neutral_boost(image)
        self.assertEqual(boosted.shape,image.shape);np.testing.assert_array_equal(image,copy)
        self.assertGreater(int(boosted[15,25,0]),60)
        self.assertEqual(tile_positions(1200,512)[-1],688)

    def test_analyzer_keeps_roi_offset_in_original_pixels(self):
        engine=LocalInstanceSegmenter.__new__(LocalInstanceSegmenter)
        engine.device='test';engine.last_info={}
        mask=np.zeros((35,210),np.uint8);cv2.line(mask,(16,17),(193,17),1,18)
        # Eight repeated proposals prevent an unnecessary small-body search;
        # deduplication must still return a single original-coordinate body.
        engine.predict=lambda *a,**k:[{'mask':mask,'offset':(40,50),'box':(40,50,250,85),'score':.95} for _ in range(8)]
        cs,_=engine.analyze(np.full((400,700,3),180,np.uint8),Settings(),(100,80,600,380))
        self.assertEqual(len(cs),1);self.assertIsNotNone(cs[0].length_px)
        self.assertAlmostEqual(cs[0].path[:,1].mean(),147,delta=1)
        self.assertGreater(cs[0].path[:,0].min(),140)
        self.assertEqual(engine.last_info['measurable_count'],1)

    def test_low_confidence_does_not_become_a_measurement(self):
        mask=np.zeros((30,200),np.uint8);cv2.line(mask,(10,15),(185,15),1,16)
        cs=measure_proposals([{'mask':mask,'offset':(30,30),'score':.55}],(300,500))
        self.assertIsNone(cs[0].length_px);self.assertEqual(cs[0].reason,'low_confidence')

    def test_black_frame_clears_all_previous_instance_metadata(self):
        engine=LocalInstanceSegmenter.__new__(LocalInstanceSegmenter)
        engine.device='test';engine.last_info={'estimated_count':8,'measurable_count':6,
                                            'visible_segment_count':2,'thin_recheck_matches':1}
        engine.predict=lambda *a,**k:self.fail('black frame used the detector')
        cs,_=engine.analyze(np.zeros((240,320,3),np.uint8),Settings())
        self.assertEqual(cs,[])
        for key in ['estimated_count','measured_count','unmeasured_count','uncertain_regions',
                    'detected_count','measurable_count','visible_segment_count','thin_recheck_matches']:
            self.assertEqual(engine.last_info[key],0,key)
        self.assertEqual(engine.last_info['effective_roi'],[0,0,320,240])

    def test_recheck_seeds_reject_round_fragmented_and_too_large_objects(self):
        thin=np.zeros((80,70),np.uint8);cv2.line(thin,(10,10),(60,70),1,9)
        round_mask=np.zeros((80,80),np.uint8);cv2.circle(round_mask,(40,40),25,1,-1)
        split=np.zeros((80,70),np.uint8);split[5:25,5:15]=1;split[45:65,45:55]=1
        masks=[thin,round_mask,split,np.ones((220,10),np.uint8)]
        ps=[{'mask':m,'score':.55,'offset':(30,40)} for m in masks]
        self.assertEqual(thin_recheck_seeds(ps),[ps[0]])
        self.assertEqual(thin_recheck_seeds([{**ps[0],'score':.44}]),[])
        self.assertEqual(thin_recheck_seeds([{**ps[0],'score':.65}]),[])
        self.assertEqual(len(thin_recheck_seeds([ps[0]]*10)),2)

    def test_recheck_requires_two_matching_views_and_keeps_source_coordinates(self):
        engine=LocalInstanceSegmenter.__new__(LocalInstanceSegmenter)
        view=np.full((500,600,3),70,np.uint8);prob=np.ones((500,600),np.float32)
        mask=np.zeros((80,30),np.uint8);cv2.line(mask,(15,8),(15,70),1,9)
        seed={'mask':mask,'offset':(230,210),'score':.55}
        def predict(patch,size,confidence):
            self.assertEqual(size,960);self.assertEqual(confidence,.6)
            # The 384px crop has source origin (77,34).
            return [{'mask':mask.copy(),'offset':(153,176),'box':(153,176,183,256),'score':.8}]
        engine.predict=predict
        extras,info=engine.recheck_thin_bodies(view,[seed],prob)
        self.assertEqual(info['thin_recheck_views'],2);self.assertEqual(info['thin_recheck_matches'],1)
        self.assertEqual(len(extras),2);self.assertEqual(extras[0]['offset'],seed['offset'])
        self.assertEqual(extras[0]['full_length_hold'],'rechecked_body')
        cs=measure_proposals(extras,view.shape)
        self.assertEqual(len(cs),1);self.assertTrue(cs[0].counted)
        self.assertIsNone(cs[0].length_px);self.assertEqual(cs[0].count_reason,'detector_recheck')
        calls=[]
        def once(*args):
            calls.append(1);return predict(*args) if len(calls)==1 else []
        engine.predict=once
        self.assertEqual(engine.recheck_thin_bodies(view,[seed],prob)[0],[])
        engine.predict=predict
        self.assertEqual(engine.recheck_thin_bodies(view,[seed],prob*0)[0],[])

    def test_worker_routes_instance_model_and_returns_provenance(self):
        class Engine:
            metadata={'model_version':'test-instance','inference_threshold':.35,
                      'texture_veto_sha256':'a'*64}
            model_sha256='test';last_info={'detected_count':1,'measurement_status':'automatic_estimate'}
            runtime_info={'python_version':'3.12','torch_version':'test','gpu_name':None}
            def analyze(self,*args):return [],'test'
        worker=AIWorker(instance_factory=Engine)
        try:
            worker.submit(AIJob(np.zeros((20,20,3),np.uint8),None,Settings(),None,'token',backend='instance'))
            until=time.monotonic()+5;result=None
            while result is None and time.monotonic()<until:time.sleep(.01);result=worker.receive()
            self.assertIsNone(result['error']);self.assertEqual(result['measurement_info']['measurement_status'],'automatic_estimate')
            self.assertEqual(result['model_version'],'test-instance')
            self.assertEqual(result['measurement_info']['texture_veto_sha256'],'a'*64)
            self.assertEqual(result['measurement_info']['runtime']['device'],'test')
            self.assertEqual(result['measurement_info']['runtime']['python_version'],'3.12')
            self.assertGreaterEqual(result['measurement_info']['analysis_seconds'],0)
            self.assertGreaterEqual(result['measurement_info']['worker_seconds'],result['measurement_info']['analysis_seconds'])
        finally:worker.close()

    def test_static_image_automatically_starts_measurement_without_manual_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            app=headless_app(folder);app.ai_engine_choice=FakeValue();app.ai_engine_choice.set('自動計測（1〜5齢）')
            called=[];app.request_ai=lambda *a:called.append(a)
            app.set_static(np.full((200,300,3),200,np.uint8),'real.png')
            self.assertEqual(app.ai_mode,'single');self.assertEqual(len(called),1)
            self.assertFalse(app.csv_path.exists())

    def test_csv_preserves_estimate_method_confidence_and_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            app=headless_app(folder);app.ppmm=None;app.paused=True
            mask=np.zeros((30,200),np.uint8);cv2.line(mask,(10,15),(185,15),1,16)
            app.candidates=measure_proposals([{'mask':mask,'offset':(30,30),'score':.95}],(300,500))
            app.last_model_info={'model_version':'test','measurement_status':'automatic_estimate','detected_count':1}
            app.publish_candidates('auto_instance_estimate',force_record=True)
            row=read_rows(app.csv_path)[0]
            self.assertEqual(row['unit'],'px');self.assertEqual(row['method'],'auto_instance_estimate')
            self.assertIn('automatic_estimate',row['settings']);self.assertIn('推定',app.obs_path.read_text(encoding='utf-8-sig'))
            app.ppmm=10;app.publish_candidates('auto_instance_estimate',force_record=True)
            second=read_rows(app.csv_path)[1]
            self.assertEqual(second['unit'],'mm');self.assertAlmostEqual(float(row['mean'])/10,float(second['mean']))
