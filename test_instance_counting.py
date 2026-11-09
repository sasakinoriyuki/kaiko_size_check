"""A visible headcount must not be confused with measurable full-body samples."""
import csv,json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from instance_geometry import merge_views,measure_proposals
from instance_counting import assign_instance_ids,count_summary
from test_app import headless_app,read_rows


def tube(start=(35,60),end=(215,60),shape=(150,260)):
 m=np.zeros(shape,np.uint8);cv2.line(m,start,end,1,18);return m


def proposal(mask,score=.9,offset=(20,20),source='whole'):
 return {'mask':mask,'score':score,'offset':offset,'source':source}


def analyze(ps,shape=(500,600)):
 cs=measure_proposals(merge_views(ps,shape),shape,merged=True);assign_instance_ids(cs);return cs


class VisibleCountsTests(unittest.TestCase):
 def test_edge_body_counts_even_when_full_length_is_unavailable(self):
  cs=analyze([proposal(tube(start=(0,60)),offset=(0,20))])
  self.assertEqual(count_summary(cs)['estimated_count'],1)
  self.assertEqual(count_summary(cs)['measured_count'],0)
  self.assertEqual(cs[0].reason,'edge');self.assertEqual(cs[0].instance_id,1)

 def test_crossing_instances_keep_two_counts_but_no_full_lengths(self):
  a=tube((35,70),(215,70),shape=(240,260));b=tube((130,20),(130,210),shape=(240,260))
  cs=analyze([proposal(a),proposal(b)])
  self.assertEqual(count_summary(cs)['estimated_count'],2)
  self.assertEqual(count_summary(cs)['measured_count'],0)

 def test_repeated_views_do_not_double_count_and_keep_valid_whole_mask(self):
  m=tube();ragged=m.copy();ragged[:,110:120]=0
  whole=proposal(m,.8);detail=proposal(ragged,.97,source='detail_tile')
  before=analyze([whole])[0].length_px;cs=analyze([detail,whole])
  self.assertEqual(len(cs),1);self.assertEqual(count_summary(cs)['estimated_count'],1)
  self.assertAlmostEqual(cs[0].length_px,before)

 def test_uncertain_supplement_does_not_block_a_strong_body(self):
  m=tube();noise=m.copy();noise[30:90,130:160]=1
  cs=analyze([proposal(m,.95),proposal(noise,.4,offset=(170,20))])
  self.assertEqual(count_summary(cs)['estimated_count'],1)
  self.assertEqual(count_summary(cs)['measured_count'],1)
  self.assertEqual(count_summary(cs)['uncertain_regions'],1)

 def test_weak_detail_only_detection_is_not_promoted_to_a_headcount(self):
  cs=analyze([proposal(tube(),.8,source='detail_tile')])
  self.assertEqual(count_summary(cs)['estimated_count'],0)
  self.assertEqual(count_summary(cs)['uncertain_regions'],1)
  self.assertIsNone(cs[0].length_px);self.assertIsNone(cs[0].instance_id)

 def test_separate_neighbors_are_not_collapsed_into_one(self):
  m=tube();cs=analyze([proposal(m),proposal(m,offset=(20,70))])
  self.assertEqual(count_summary(cs)['estimated_count'],2)
  self.assertEqual([c.instance_id for c in cs],[1,2])

 def test_a_short_valid_fragment_cannot_replace_a_large_branched_region(self):
  big=tube((35,70),(215,70),shape=(240,260))|tube((130,20),(130,210),shape=(240,260))
  part=tube((35,70),(105,70),shape=(240,260))
  cs=analyze([proposal(big,.9),proposal(part,.98,source='detail_tile')])
  self.assertEqual(len(cs),1);self.assertIsNone(cs[0].length_px)

 def test_count_csv_keeps_length_csv_count_as_number_measured(self):
  with tempfile.TemporaryDirectory() as folder:
   app=headless_app(folder);app.ppmm=None;app.paused=True
   app.candidates=analyze([proposal(tube()),proposal(tube(start=(0,60)),offset=(0,180))])
   app.last_model_info={'model_version':'test-counter','estimated_count':2}
   app.publish_candidates('auto_instance_estimate',force_record=True)
   length_rows=read_rows(app.csv_path)
   with (Path(folder)/'counts.csv').open(encoding='utf-8-sig',newline='') as f:counts=list(csv.DictReader(f))
   self.assertEqual(length_rows[0]['count'],'1');self.assertEqual(counts[0]['estimated_count'],'2')
   self.assertEqual(counts[0]['measured_count'],'1');self.assertIn('頭数推定 2',app.result.get())
   self.assertIn('頭数推定 2',app.obs_path.read_text(encoding='utf-8-sig'))
