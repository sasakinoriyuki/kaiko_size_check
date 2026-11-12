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
        report={'release':expected_version,'rounds':rounds,'cases':results,
                'zip_reopened_and_reanalyzed':True,'portable_data':True,
                'synthetic_inputs_only':True,'biological_accuracy_test':False,
                'runtime':app.last_model_info['runtime']}
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
