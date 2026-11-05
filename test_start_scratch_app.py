import io
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image
from start_scratch_app import local_reference


class LocalStartupTests(unittest.TestCase):
    def test_saved_local_sample_opens_without_a_distributed_photo(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'data/samples';directory.mkdir(parents=True)
            buffer=io.BytesIO();Image.new('RGB',(51,37),(20,30,40)).save(buffer,format='PNG')
            with ZipFile(directory/'sample_1.zip','w') as z:z.writestr('original.png',buffer.getvalue());z.writestr('note.json','{"note":"data only"}')
            image,label=local_reference(root)
            self.assertEqual(image.shape,(37,51,3))
            np.testing.assert_array_equal(image[0,0],[40,30,20])
            self.assertIn('保存済み',label)

    def test_no_sample_and_corrupt_sample_are_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(local_reference(root),(None,None))
            (root/'data/samples').mkdir(parents=True);(root/'data/samples/sample_bad.zip').write_bytes(b'not zip')
            self.assertEqual(local_reference(root),(None,None))
