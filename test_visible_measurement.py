"""A clipped or occluded segment must never become a claimed full-body length."""
import csv,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from instance_geometry import merge_views,measure_proposals
from instance_counting import assign_instance_ids,count_summary
from visible_measurement import estimate_visible_segments,visible_segment_count,_visible_runs
from measurement import polyline_length,summarize
from test_app import headless_app,read_rows


def proposal(points,width=20,shape=(240,300),offset=(0,0),score=.95):
 mask=np.zeros(shape,np.uint8);cv2.polylines(mask,[np.int32(points)],False,1,width)
 return {'mask':mask,'offset':offset,'score':score,'source':'whole'}


def evaluate(ps,shape=(400,500)):
 merged=merge_views(ps,shape);cs=measure_proposals(merged,shape,merged=True)
 estimate_visible_segments(merged,cs);assign_instance_ids(cs);return cs


class VisibleLengthTests(unittest.TestCase):
 def test_frame_cut_has_visible_length_but_no_full_length(self):
  cs=evaluate([proposal([(-40,100),(150,100)])]);c=cs[0]
  self.assertEqual(c.reason,'edge');self.assertIsNone(c.length_px)
  self.assertAlmostEqual(c.visible_length_px,160,delta=4)
  self.assertEqual(summarize(cs)['count'],0);self.assertEqual(visible_segment_count(cs),1)

 def test_overlap_remains_unknown_and_does_not_join_the_two_sides(self):
  a=proposal([(35,120),(255,120)]);b=proposal([(140,25),(140,215)])
  cs=evaluate([a,b])
  self.assertEqual(count_summary(cs)['estimated_count'],2)
  self.assertEqual(visible_segment_count(cs),2)
  for c in cs:
   self.assertIsNone(c.length_px);self.assertLess(c.visible_length_px,130)
   self.assertIn('overlap_excluded',c.visible_note)

 def test_disconnected_pieces_are_not_summed(self):
  p=proposal([(25,100),(105,100)]);q=proposal([(160,100),(220,100)])
  p['mask']|=q['mask'];cs=evaluate([p]);c=cs[0]
  self.assertIsNone(c.length_px);self.assertAlmostEqual(c.visible_length_px,100,delta=4)
  self.assertLess(c.visible_path[:,0].max(),130)

 def test_full_length_is_preserved_and_not_duplicated_as_a_partial(self):
  cs=evaluate([proposal([(40,100),(230,100)])]);c=cs[0]
  self.assertIsNotNone(c.length_px);self.assertIsNone(c.visible_length_px)
  self.assertEqual(visible_segment_count(cs),0)

 def test_weak_unconfirmed_detection_does_not_get_a_size(self):
  cs=evaluate([proposal([(-40,100),(150,100)],score=.4)])
  self.assertFalse(cs[0].counted);self.assertIsNone(cs[0].visible_length_px)

 def test_source_offset_is_retained(self):
  cs=evaluate([proposal([(25,0),(25,110)],shape=(160,100),offset=(120,0))]);c=cs[0]
  self.assertIsNone(c.length_px);self.assertIsNotNone(c.visible_length_px)
  self.assertAlmostEqual(c.visible_path[:,0].mean(),145,delta=2)

 def test_closed_loop_is_not_given_invented_visible_endpoints(self):
  p=proposal([(20,20),(21,20)]);p['mask'][:]=0;cv2.circle(p['mask'],(140,100),50,1,14)
  cs=evaluate([p]);self.assertIsNone(cs[0].length_px);self.assertIsNone(cs[0].visible_length_px)

 def test_a_smoothed_line_may_not_bridge_an_excluded_pixel_region(self):
  excluded=np.zeros((80,200),bool);excluded[:,90:110]=True
  runs=_visible_runs(np.array([[20.,40.],[180.,40.]]),(0,0),excluded)
  self.assertEqual(len(runs),2)
  self.assertLess(max(polyline_length(r) for r in runs),80)

 def test_csv_keeps_unknown_full_length_blank_and_preserves_existing_records(self):
  with tempfile.TemporaryDirectory() as folder:
   app=headless_app(folder);app.ppmm=10;app.paused=True
   app.candidates=evaluate([proposal([(40,100),(230,100)]),proposal([(-40,100),(150,100)],offset=(0,180))])
   app.publish_candidates('auto_instance_estimate',force_record=True)
   self.assertEqual(read_rows(app.csv_path)[0]['count'],'1')
   path=Path(folder)/'individual_sizes.csv'
   with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
   full=next(r for r in rows if r['length_status']=='full_estimate')
   part=next(r for r in rows if r['length_status']=='visible_segment_estimate')
   self.assertEqual(full['visible_segment_length'],'');self.assertEqual(part['full_length'],'')
   self.assertEqual(part['unit'],'mm');self.assertAlmostEqual(float(part['visible_segment_length']),16,delta=.4)
   original=path.read_bytes();app.publish_candidates('auto_instance_estimate',force_record=True)
   self.assertTrue(path.read_bytes().startswith(original))
   self.assertIn('可視部 1',app.result.get())
