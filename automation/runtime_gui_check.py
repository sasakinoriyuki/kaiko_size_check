"""Exercise an installed release repeatedly using synthetic, non-private inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def run(installed_root,expected_version,rounds=3):
    installed_root=Path(installed_root).resolve()
    os.environ['KAIKO_STORAGE_ROOT']=str(installed_root)
    sys.path.insert(0,str(installed_root/'_internal'))
    import cv2
    import numpy as np
    import tkinter as tk
    import torch
    from app import App
    from ai_measurement import AIWorker
    from instance_detector import LocalInstanceSegmenter
    from archive_batch import read_sample_zip
    from measurement import demo_frame

    def factory():
        engine=LocalInstanceSegmenter()
        torch.set_num_threads(1);cv2.setNumThreads(1)
        return engine

    root=tk.Tk();app=None
    baseline={};results=[]
    try:
        app=App(root);assert expected_version in root.title()
        app.ai_worker=AIWorker(instance_factory=factory)
        cases={'synthetic':demo_frame(),'black':np.zeros((240,320,3),np.uint8),
               'neutral':np.full((300,410,3),100,np.uint8)}
        def completed():
            until=time.monotonic()+90
            while time.monotonic()<until:
                root.update();time.sleep(.01)
                if 'analysis_seconds' in app.last_model_info and not app.ai_worker.busy:return
                if app.ai_failed:raise AssertionError(app.status.get())
            raise AssertionError('GUI analysis did not finish')

        for turn in range(rounds):
            names=list(cases) if turn%2==0 else list(reversed(cases))
            for name in names:
                frame=cases[name]
                app.set_static(frame.copy(),name+'.png');completed()
                assert np.array_equal(app.frame,frame)
                info=app.last_model_info
                count=sum(c.counted for c in app.candidates)
                full=sum(c.length_px is not None for c in app.candidates)
                assert info['estimated_count']==count and info['measurable_count']==full
                if name=='black':assert count==full==info['visible_segment_count']==0
                values=[{'counted':c.counted,'reason':c.reason,'length':c.length_px,
                         'visible':c.visible_length_px,'contour':c.contour.tolist()}
                        for c in app.candidates]
                signature=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
                if name in baseline:assert signature==baseline[name],name+' changed between rounds'
                baseline[name]=signature
                saved=app.save_sample();assert saved is not None
                image,report=read_sample_zip(saved)
                assert np.array_equal(image,frame)
                assert report['app_version']==expected_version
                assert report['model_info']['estimated_count']==count
                assert report['model_info']['runtime']['device'] in {'cpu','cuda'}
                results.append({'round':turn+1,'case':name,'count':count,'full':full,
                                'zip_pixels_and_metadata_valid':True})
        # Open the ZIP through the real file-open route, then re-analyze it.
        with patch('app.filedialog.askopenfilename',return_value=str(saved)):
            app.open_file()
        completed();assert np.array_equal(app.frame,cases[names[-1]])
        assert app.ppmm is None and app.data==installed_root/'data'
        assert (app.data/'counts.csv').is_file()
        # Geometry fixture checks per-animal pixel/cm rendering and CSV output;
        # it is deliberately independent of biological model predictions.
        from instance_geometry import measure_proposals
        from body_dimensions import populate_body_widths,dimension_text
        import csv
        app.reset_source('dimension_fixture.png')
        app.frame=np.full((300,500,3),180,np.uint8)
        mask=np.zeros((40,240),np.uint8);cv2.line(mask,(20,20),(220,20),1,18)
        app.candidates=measure_proposals([{'mask':mask,'offset':(80,100),'score':.95}],app.frame.shape)
        app.candidates[0].instance_id=1;app.ppmm=5;app.paused=True;app.ai_mode='single'
        populate_body_widths(app.candidates)
        app.publish_candidates('auto_instance_estimate',force_record=True);root.update()
        text=app.items.get(0)
        assert 'px' in text and 'cm' in text and '×' in text
        c=app.candidates[0];assert abs(c.width_px-19)<1
        assert dimension_text(c,5) in text
        with (app.data/'individual_sizes.csv').open(encoding='utf-8-sig',newline='') as stream:
            sizes=list(csv.DictReader(stream))
        assert abs(float(sizes[-1]['body_width_cm'])-c.width_px/50)<1e-6
        sample=app.save_sample();_,dimensions=read_sample_zip(sample)
        assert abs(dimensions['individual_sizes'][0]['body_width_cm']-c.width_px/50)<1e-6
        app.ppmm=None;app.publish_candidates('auto_instance_estimate',force_record=True)
        assert 'cm' not in app.items.get(0) and 'px' in app.items.get(0)
        from improvement_ui import ImprovementWindow
        from zipfile import ZipFile
        review_data=installed_root/'data/review_ci_fixture'
        source_dir=review_data/'samples';source_dir.mkdir(parents=True)
        for i in range(2):
            _,encoded=cv2.imencode('.png',np.full((80,120,3),100+i,np.uint8))
            metadata={'model_info':{'instances':[{'contour_source_xy':[[10,10],[50,10],[50,35],[10,35]]}]}}
            with ZipFile(source_dir/(str(i)+'.zip'),'w') as archive:
                archive.writestr('source.png',encoded.tobytes());archive.writestr('source.json',json.dumps(metadata))
        review=ImprovementWindow(tk.Toplevel(root),review_data)
        until=time.monotonic()+15
        while review.queue.current is None and time.monotonic()<until:root.update();time.sleep(.03)
        assert review.queue.current and len(review.queue.objects)==1
        review.selected=0;review.delete();review.undo();assert len(review.queue.objects)==1
        review.begin();review.points=[[60,40],[90,40],[90,60]];review.finish_polygon()
        assert len(review.queue.objects)==2
        review.complete();assert review.queue.position==1
        review.complete();assert review.queue.current is None
        exported,review_count=review.queue.export();assert review_count==2 and exported
        review.close()
        report={'release':expected_version,'rounds':rounds,'cases':results,
                'zip_reopened_and_reanalyzed':True,'portable_data':True,
                'synthetic_inputs_only':True,'biological_accuracy_test':False,
                'individual_pixel_and_cm_dimensions':True,'improvement_mode_gui_verified':True,
                'runtime':info['runtime']}
        print(json.dumps(report),flush=True)
        return report
    finally:
        if app is not None:
            worker=app.ai_worker;app.close()
            if worker is not None:
                worker.thread.join(timeout=90)
                assert not worker.thread.is_alive(),'analysis thread did not stop'
        else:root.destroy()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--installed-root',required=True)
    p.add_argument('--expected-version',required=True);p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args();run(a.installed_root,a.expected_version,a.rounds)
