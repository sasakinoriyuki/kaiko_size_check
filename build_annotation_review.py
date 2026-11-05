"""Build an offline, editable review page. No model or network is used."""
import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def validate(data):
    if data.get('schema_version') != 1 or not data.get('frames'):
        raise ValueError('Unsupported or empty annotation document')
    seen = set()
    for frame in data['frames']:
        if frame['id'] in seen:
            raise ValueError('Duplicate frame ID')
        seen.add(frame['id'])
        w, h = frame['resolution']
        if w <= 0 or h <= 0 or frame['unlabelled_pixels'] != 'ignore':
            raise ValueError('Unlabelled pixels must remain ignored')
        ids = set()
        for obj in frame['objects']:
            if obj['id'] in ids:
                raise ValueError('Duplicate object ID')
            ids.add(obj['id'])
            if obj['visibility'] not in ('full', 'partial', 'uncertain', 'ignore_region'):
                raise ValueError('Invalid visibility')
            if obj['review_status'] not in ('draft', 'human_confirmed'):
                raise ValueError('Invalid review status')
            if len(obj['polygon']) < 3:
                raise ValueError('Polygon needs at least three points')
            for x, y in obj['polygon']:
                if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x < w and 0 <= y < h):
                    raise ValueError('Polygon outside source image')
            p = obj['polygon']
            area2 = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(p, p[1:]+p[:1]))
            if abs(area2) < 2:
                raise ValueError('Polygon has no area')
    return data


def export_confirmed(data):
    """Partial annotations must never become a complete foreground/background mask."""
    result = copy.deepcopy(validate(data))
    for frame in result['frames']:
        for obj in frame['objects']:
            obj['eligible_for_length_training'] = (
                obj['review_status'] == 'human_confirmed' and obj['visibility'] == 'full')
    return result


def build(data, image_root, output):
    data = export_confirmed(data)
    root = Path(image_root).resolve()
    embedded = {}
    for frame in data['frames']:
        relative = Path(frame['image'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Image path must stay in image folder')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != frame['sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(path) as im:
            if im.format != 'PNG' or list(im.size) != frame['resolution']:
                raise ValueError('Expected original PNG with matching dimensions')
        embedded[frame['id']] = 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')
    template = Path(__file__).with_name('annotation_review_template.html').read_text(encoding='utf-8')
    payload = json.dumps({'annotations': data, 'images': embedded}, ensure_ascii=False).replace('<', '\\u003c')
    Path(output).write_text(template.replace('__ANNOTATION_PAYLOAD__', payload), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('image_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(json.loads(args.annotations.read_text(encoding='utf-8')), args.image_root, args.output)
