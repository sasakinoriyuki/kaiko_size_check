import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
from body_dimensions import central_body_width,populate_body_widths,dimension_record,dimension_text,DIMENSION_COLUMNS
from instance_geometry import measure_proposals
from test_app import headless_app
from test_visible_measurement import evaluate,proposal


def body(points,width=18,offset=(30,30)):
    mask=np.zeros((280,320),np.uint8)
    cv2.polylines(mask,[np.rint(points).astype(np.int32)],False,1,width)
    cs=measure_proposals([{'mask':mask,'offset':offset,'score':.95}],(600,700))
    populate_body_widths(cs)
    return cs[0]


class BodyDimensionsTests(unittest.TestCase):
    def test_width_follows_body_and_not_diagonal_bounding_box(self):
        c=body([(45,45),(240,240)])
        self.assertIsNotNone(c.length_px)
        self.assertAlmostEqual(c.width_px,19,delta=1.5)
        self.assertLess(c.width_px,30)

    def test_curved_width_and_source_translation_preserve_native_size(self):
        angles=np.linspace(-1.3,1.3,100)
        points=np.c_[140+70*np.cos(angles),140+70*np.sin(angles)]
        a=body(points,offset=(30,30));b=body(points,offset=(180,120))
        self.assertIsNotNone(a.length_px)
        self.assertAlmostEqual(a.width_px,19,delta=1.5)
        self.assertAlmostEqual(a.width_px,b.width_px,places=5)
        self.assertEqual(a.length_px,b.length_px)

    def test_width_uses_both_boundaries_even_for_offcentre_path(self):
        contour=np.array([[[20,40]],[[220,40]],[[220,60]],[[20,60]]],np.int32)
        a=central_body_width(contour,np.array([[30,50],[210,50]]))
        b=central_body_width(contour,np.array([[30,54],[210,54]]))
        self.assertAlmostEqual(a,21,delta=.2);self.assertAlmostEqual(a,b,places=5)

    def test_scale_converts_both_dimensions_without_hiding_pixels(self):
        c=body([(40,120),(240,120)])
        before=(c.length_px,c.width_px)
        r=dimension_record(c,5)
        self.assertAlmostEqual(r['full_length_cm'],c.length_px/50)
        self.assertAlmostEqual(r['body_width_cm'],c.width_px/50)
        self.assertIn('px',dimension_text(c,5));self.assertIn('cm',dimension_text(c,5))
        self.assertEqual(before,(c.length_px,c.width_px))
        for scale in [None,0,float('nan')]:
            self.assertIsNone(dimension_record(c,scale)['full_length_cm'])
            self.assertNotIn('cm',dimension_text(c,scale))
        self.assertIn('0.0019 cm',dimension_text(c,1000))

    def test_visible_width_does_not_become_full_body_dimensions(self):
        c=evaluate([proposal([(-40,100),(150,100)])])[0]
        populate_body_widths([c]);r=dimension_record(c,10)
        self.assertIsNone(c.length_px);self.assertIsNone(r['body_width_px'])
        self.assertIsNone(r['full_length_cm']);self.assertIsNone(r['body_width_cm'])
        self.assertAlmostEqual(r['visible_body_width_px'],21,delta=1)
        self.assertAlmostEqual(r['visible_body_width_cm'],.21,delta=.01)

    def test_unconfirmed_or_missing_path_keeps_unknown_width(self):
        c=body([(40,120),(240,120)]);c.counted=False
        populate_body_widths([c]);self.assertIsNone(c.width_px)
        self.assertIsNone(central_body_width(c.contour,None))
        self.assertIsNone(central_body_width(c.contour,[[900,900],[1000,900]]))

    def test_a_sideways_crop_cannot_measure_the_missing_body_width(self):
        mask=np.zeros((280,100),np.uint8);cv2.line(mask,(0,30),(0,240),1,20)
        from instance_geometry import merge_views
        from visible_measurement import estimate_visible_segments
        proposals=merge_views([{'mask':mask,'offset':(0,0),'score':.95}],mask.shape)
        cs=measure_proposals(proposals,mask.shape,merged=True);estimate_visible_segments(proposals,cs)
        self.assertIsNotNone(cs[0].visible_length_px)
        populate_body_widths(cs,mask.shape)
        self.assertIsNone(cs[0].visible_width_px)
        populate_body_widths(cs)  # Rendering must retain the conservative result.
        self.assertIsNone(cs[0].visible_width_px)

    def test_overlap_does_not_use_a_shared_region_as_body_width(self):
        cs=evaluate([proposal([(35,120),(255,120)]),proposal([(140,25),(140,215)])])
        populate_body_widths(cs)
        self.assertTrue(all(c.visible_length_px is not None for c in cs))
        self.assertTrue(all(c.visible_width_px is None for c in cs))

    def test_old_individual_csv_is_upgraded_without_losing_records(self):
        with tempfile.TemporaryDirectory() as folder:
            app=headless_app(folder);app.candidates=[body([(40,120),(240,120)])]
            self.assertTrue(app.log_individual_sizes())
            path=Path(folder)/'individual_sizes.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            old={k:v for k,v in rows[0].items() if k not in DIMENSION_COLUMNS}
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(old));writer.writeheader();writer.writerow(old)
            self.assertTrue(app.log_individual_sizes())
            with path.open(encoding='utf-8-sig',newline='') as f:after=list(csv.DictReader(f))
            self.assertEqual(len(after),2)
            self.assertEqual({k:after[0][k] for k in old},old)
            self.assertTrue(all(after[0][k]=='' for k in DIMENSION_COLUMNS))
            self.assertAlmostEqual(float(after[1]['body_width_cm']),app.candidates[0].width_px/100)

    def test_failed_csv_upgrade_preserves_original_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as folder:
            app=headless_app(folder);app.candidates=[body([(40,120),(240,120)])]
            self.assertTrue(app.log_individual_sizes());path=Path(folder)/'individual_sizes.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            old={k:v for k,v in rows[0].items() if k not in DIMENSION_COLUMNS}
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(old));writer.writeheader();writer.writerow(old)
            original=path.read_bytes()
            with patch('app.os.replace',side_effect=PermissionError('Excel holds this file')):
                self.assertFalse(app.log_individual_sizes())
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(folder).glob('.individual_sizes-*.tmp')),[])

    def test_validation_zip_saves_pixel_and_cm_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            app=headless_app(folder);app.candidates=[body([(40,120),(240,120)])]
            app.candidates[0].instance_id=1
            saved=app.save_sample()
            with ZipFile(saved) as z:info=json.loads(z.read(saved.stem+'.json'))
            self.assertEqual(info['individual_sizes'][0]['frame_instance_id'],1)
            self.assertAlmostEqual(info['individual_sizes'][0]['body_width_px'],19,delta=1)
            self.assertAlmostEqual(info['individual_sizes'][0]['body_width_cm'],.19,delta=.01)


if __name__=='__main__':unittest.main()
