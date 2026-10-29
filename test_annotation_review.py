import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from build_annotation_review import build, export_confirmed, validate


class AnnotationReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'sample.png'
        Image.new('RGB', (32, 24), (81, 144, 27)).save(self.path)
        self.data = {'schema_version': 1, 'frames': [{
            'id': 'sample', 'image': 'sample.png', 'resolution': [32, 24],
            'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
            'unlabelled_pixels': 'ignore', 'display_name': '</script><script>bad()</script>',
            'objects': [{'id': 'a', 'visibility': 'full', 'review_status': 'draft',
                         'polygon': [[1, 1], [20, 1], [20, 12], [1, 12]],
                         'eligible_for_length_training': True}]}]}

    def test_only_confirmed_full_is_eligible_without_mutating_input(self):
        objects = self.data['frames'][0]['objects']
        for i, visibility in enumerate(('partial', 'uncertain', 'ignore_region', 'full')):
            obj = copy.deepcopy(objects[0])
            obj.update(id=str(i), visibility=visibility, review_status='human_confirmed')
            objects.append(obj)
        result = export_confirmed(self.data)
        self.assertEqual([o['eligible_for_length_training'] for o in result['frames'][0]['objects']],
                         [False, False, False, False, True])
        self.assertEqual(result['frames'][0]['unlabelled_pixels'], 'ignore')
        self.assertTrue(objects[0]['eligible_for_length_training'])

    def test_rejects_background_assumption_invalid_points_and_duplicate_ids(self):
        cases = []
        d = copy.deepcopy(self.data); d['frames'][0]['unlabelled_pixels'] = 'background'; cases.append(d)
        d = copy.deepcopy(self.data); d['frames'][0]['objects'][0]['polygon'][0] = [32, 0]; cases.append(d)
        d = copy.deepcopy(self.data); d['frames'][0]['objects'][0]['polygon'][0] = [float('nan'), 0]; cases.append(d)
        d = copy.deepcopy(self.data); d['frames'][0]['objects'][0]['polygon'] = [[1, 1], [2, 2], [3, 3]]; cases.append(d)
        d = copy.deepcopy(self.data); d['frames'][0]['objects'].append(copy.deepcopy(d['frames'][0]['objects'][0])); cases.append(d)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate(data)

    def test_build_keeps_original_bytes_and_escapes_embedded_metadata(self):
        out = self.root/'review.html'
        build(self.data, self.root, out)
        text = out.read_text()
        embedded = text.split('<script id="payload" type="application/json">', 1)[1].split('</script>', 1)[0]
        payload = json.loads(embedded)
        raw = base64.b64decode(payload['images']['sample'].split(',', 1)[1])
        self.assertEqual(raw, self.path.read_bytes())
        self.assertNotIn('</script>', embedded)
        self.assertIn("connect-src 'none'", text)
        self.assertFalse(payload['annotations']['frames'][0]['objects'][0]['eligible_for_length_training'])

    def test_rejects_modified_images_and_paths_outside_package(self):
        d = copy.deepcopy(self.data); d['frames'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'hash'):
            build(d, self.root, self.root/'review.html')
        for path in ('../sample.png', str(self.path)):
            d = copy.deepcopy(self.data); d['frames'][0]['image'] = path
            with self.assertRaisesRegex(ValueError, 'path'):
                build(d, self.root, self.root/'review.html')


if __name__ == '__main__':
    unittest.main()
