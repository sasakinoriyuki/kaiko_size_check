import unittest
import cv2
import numpy as np

from instance_geometry import measure_mask,measure_proposals,deduplicate


class InstanceGeometryTests(unittest.TestCase):
    def tube(self,points,width=14,shape=(240,300)):
        mask=np.zeros(shape,np.uint8)
        cv2.polylines(mask,[np.int32(points)],False,1,width)
        for point in (points[0],points[-1]):cv2.circle(mask,tuple(np.int32(point)),width//2,1,-1)
        return mask

    def test_straight_body_length_includes_both_caps(self):
        mask=self.tube([(50,120),(230,120)],20)
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNotNone(c.length_px)
        self.assertAlmostEqual(c.length_px,200,delta=3)

    def test_curved_body_follows_arc_instead_of_bounding_box(self):
        angles=np.linspace(-1.35,1.35,180)
        points=np.c_[130+65*np.cos(angles),120+65*np.sin(angles)]
        mask=self.tube(points,14)
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNotNone(c.length_px,c.reason)
        self.assertAlmostEqual(c.length_px,65*2.7+14,delta=7)

    def test_tiny_islands_do_not_discard_a_complete_body(self):
        mask=self.tube([(50,120),(230,120)])
        mask[20:22,20:22]=1
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNotNone(c.length_px)

    def test_large_disconnected_fragments_are_not_joined(self):
        mask=self.tube([(35,120),(115,120)])|self.tube([(145,120),(235,120)])
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNone(c.length_px)
        self.assertEqual(c.reason,'fragmented')

    def test_single_pixel_segmentation_crack_does_not_split_the_body(self):
        mask=self.tube([(50,120),(230,120)],20)
        mask[:,140]=0
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNotNone(c.length_px)
        self.assertAlmostEqual(c.length_px,200,delta=3)

    def test_crossing_two_bodies_does_not_form_one_long_animal(self):
        mask=self.tube([(35,120),(245,120)])|self.tube([(140,30),(140,205)])
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNone(c.length_px)
        self.assertEqual(c.reason,'branch')

    def test_closed_loop_is_not_assigned_an_invented_head_and_tail(self):
        mask=np.zeros((240,300),np.uint8);cv2.circle(mask,(150,120),60,1,14)
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNone(c.length_px)

    def test_frame_edge_body_is_not_measured(self):
        mask=self.tube([(0,120),(180,120)])
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNone(c.length_px);self.assertEqual(c.reason,'edge')

    def test_round_food_blob_is_rejected_by_geometry(self):
        mask=np.zeros((240,300),np.uint8);cv2.ellipse(mask,(140,120),(35,26),0,0,360,1,-1)
        c=measure_mask(mask,frame_shape=mask.shape)
        self.assertIsNone(c.length_px)

    def test_source_offset_changes_coordinates_but_not_length(self):
        mask=self.tube([(50,120),(230,120)])
        a=measure_mask(mask);b=measure_mask(mask,(400,200))
        self.assertEqual(a.length_px,b.length_px)
        np.testing.assert_allclose(b.path-a.path,[400,200]+np.zeros_like(a.path))

    def test_duplicate_tile_mask_is_kept_once_and_neighbor_stays_separate(self):
        mask=self.tube([(50,100),(230,100)],shape=(260,300))
        p={'mask':mask,'offset':(0,0),'score':.9}
        duplicate={**p,'score':.8}
        other={'mask':mask,'offset':(0,55),'score':.8}
        self.assertEqual(len(deduplicate([p,duplicate,other])),2)

    def test_overlap_is_retained_as_detection_but_not_full_length(self):
        a=self.tube([(35,120),(245,120)])
        b=self.tube([(140,30),(140,205)])
        result=measure_proposals([{'mask':a,'offset':(0,0),'score':.9},{'mask':b,'offset':(0,0),'score':.8}],a.shape)
        self.assertEqual(len(result),2)
        self.assertTrue(all(c.length_px is None for c in result))


if __name__=='__main__':unittest.main()
