"""Body-centred widths and explicit pixel/cm dimensions; no box-size proxy."""
import math
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

WIDTH_METHOD='median_normal_cross_section_middle_50_percent'
DIMENSION_COLUMNS=['full_length_px','body_width_px','visible_segment_length_px',
                   'visible_body_width_px','full_length_cm','body_width_cm',
                   'visible_segment_length_cm','visible_body_width_cm','width_method']


def central_body_width(contour,path,frame_shape=None):
    """Median perpendicular span across the middle half of a measured path.

    Follow the first boundary on each side, rather than the whole bounding box
    or a diameter across another bend. Measurements remain in source pixels.
    """
    if path is None:return None
    points=np.asarray(path,dtype=float).reshape(-1,2)
    outline=np.asarray(contour,dtype=float).reshape(-1,2)
    if len(points)<2 or len(outline)<3 or not np.isfinite(points).all() or not np.isfinite(outline).all():return None
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-6]
    points=points[keep]
    if len(points)<2:return None
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    length=float(arc[-1])
    if length<8:return None
    x,y,w,h=cv2.boundingRect(np.rint(outline).astype(np.int32))
    if w*h>20_000_000:return None
    origin=np.asarray([x-2,y-2]);mask=np.zeros((h+4,w+4),np.float32)
    cv2.fillPoly(mask,[np.rint(outline-origin).astype(np.int32)],1.)
    targets=np.linspace(.25*length,.75*length,21)
    def at(distances):
        return np.column_stack([np.interp(distances,arc,points[:,i]) for i in [0,1]])
    centres=at(targets)-origin
    step=max(2.,min(10.,.03*length))
    tangent=at(np.minimum(length,targets+step))-at(np.maximum(0,targets-step))
    magnitude=np.linalg.norm(tangent,axis=1)
    valid=magnitude>1e-6
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])/np.maximum(magnitude[:,None],1e-6)
    limit=min(max(mask.shape),max(3.,length*.5))
    distances=np.linspace(0,limit,int(math.ceil(limit*2))+1)
    spans=[]
    for direction in [-1,1]:
        rays=centres[:,None,:]+direction*normal[:,None,:]*distances[None,:,None]
        values=map_coordinates(mask,[rays[:,:,1],rays[:,:,0]],order=1,mode='constant',cval=0,prefilter=False)
        outside=values<.5
        boundary=np.argmax(outside,axis=1)
        usable=outside.any(axis=1)&(boundary>0)
        valid&=usable
        before=np.maximum(0,boundary-1);rows=np.arange(len(targets))
        hi=values[rows,before];lo=values[rows,boundary]
        fraction=np.clip((hi-.5)/np.maximum(hi-lo,1e-6),0,1)
        span=distances[before]+fraction*(distances[1]-distances[0])
        spans.append(span)
        if frame_shape is not None:
            edge=centres+direction*normal*span[:,None]+origin
            fh,fw=frame_shape[:2]
            valid&=(edge[:,0]>.5)&(edge[:,1]>.5)&(edge[:,0]<fw-1.5)&(edge[:,1]<fh-1.5)
    widths=spans[0]+spans[1]
    valid&=np.isfinite(widths)&(widths>=1)
    if valid.sum()<15:return None
    return float(np.median(widths[valid]))


def populate_body_widths(candidates,frame_shape=None):
    for c in candidates:
        if not c.counted:
            c.width_px=None;c.visible_width_px=None
            continue
        if c.width_evaluated:continue
        if c.length_px is not None and c.path is not None:
            c.visible_width_px=None
            if c.width_px is None:c.width_px=central_body_width(c.contour,c.path,frame_shape)
        elif c.visible_length_px is not None and c.visible_path is not None:
            c.width_px=None
            if c.visible_width_px is None and 'overlap_excluded' not in c.visible_note:
                c.visible_width_px=central_body_width(c.contour,c.visible_path,frame_shape)
        else:c.width_px=None;c.visible_width_px=None
        c.width_evaluated=True


def dimension_record(c,pixels_per_mm=None):
    calibrated=(pixels_per_mm is not None and math.isfinite(pixels_per_mm) and pixels_per_mm>0)
    def cm(value):return None if value is None or not calibrated else float(value)/(10*pixels_per_mm)
    width=getattr(c,'width_px',None) if c.length_px is not None else None
    visible_width=getattr(c,'visible_width_px',None) if c.length_px is None else None
    return {'frame_instance_id':c.instance_id,'full_length_px':c.length_px,'body_width_px':width,
            'visible_segment_length_px':c.visible_length_px,'visible_body_width_px':visible_width,
            'full_length_cm':cm(c.length_px),'body_width_cm':cm(width),
            'visible_segment_length_cm':cm(c.visible_length_px),'visible_body_width_cm':cm(visible_width),
            'width_method':WIDTH_METHOD if width is not None or visible_width is not None else None}


def dimension_text(c,pixels_per_mm=None):
    r=dimension_record(c,pixels_per_mm)
    length,width=(r['full_length_px'],r['body_width_px']) if c.length_px is not None else (r['visible_segment_length_px'],r['visible_body_width_px'])
    if length is None:return None
    def number(value,precision):return '—' if value is None else f'{value:.{precision}f}'
    text=f'{number(length,1)}×{number(width,1)} px'
    length_cm,width_cm=(r['full_length_cm'],r['body_width_cm']) if c.length_px is not None else (r['visible_segment_length_cm'],r['visible_body_width_cm'])
    if length_cm is not None:
        # Preserve small hatchling dimensions instead of rounding to 0.00cm.
        cm_width='—' if width_cm is None else f'{width_cm:.3g}'
        text+=f' | {length_cm:.3g}×{cm_width} cm'
    return text
