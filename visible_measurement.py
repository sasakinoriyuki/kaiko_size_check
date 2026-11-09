"""Visible continuous segments are estimates, never replacements for full length."""
import cv2
import numpy as np

from instance_geometry import measure_mask
from measurement import polyline_length


def _visible_runs(path,offset,excluded):
    """Check between vertices too; a smoothed line may not cross an exclusion."""
    dense=[]
    for a,b in zip(path,path[1:]):
        steps=max(1,int(np.ceil(np.linalg.norm(b-a)*2)))
        dense.extend(a+(b-a)*i/steps for i in range(steps))
    dense=np.vstack([*dense,path[-1]])
    local=np.rint(dense-np.asarray(offset)).astype(int)
    h,w=excluded.shape;valid=(local[:,0]>=0)&(local[:,1]>=0)&(local[:,0]<w)&(local[:,1]<h)
    inside=np.flatnonzero(valid)
    valid[inside]&=~excluded[local[inside,1],local[inside,0]]
    edges=np.diff(np.r_[False,valid,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return [dense[a:b] for a,b in zip(starts,ends) if b-a>=2 and polyline_length(dense[a:b])>=8]


def estimate_visible_segments(proposals,candidates):
    """Measure the longest unambiguous connected piece of each held instance.

    Do not join separated pieces, sum across an occlusion or extend outside a
    source mask. A frame cut is allowed only for this explicitly partial value.
    The caller retains full-length hold reasons and excludes these values from
    the full-length average.
    """
    if len(proposals)!=len(candidates):
        raise ValueError('Instance masks and candidates must stay aligned')
    for i,(p,c) in enumerate(zip(proposals,candidates)):
        c.visible_path=None;c.visible_length_px=None;c.visible_note=''
        if not c.counted or c.length_px is not None:continue
        mask=(np.asarray(p['mask'])>0).astype(np.uint8)
        x,y=p['offset'];h,w=mask.shape
        # Repair quantisation before subtraction. Never bridge the excluded
        # shared region when finding connected pieces.
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        excluded=np.zeros(mask.shape,bool)
        removed=False
        for j,(other,oc) in enumerate(zip(proposals,candidates)):
            if i==j or not oc.counted:continue
            ox,oy=other['offset'];oh,ow=other['mask'].shape
            left,top=max(x,ox),max(y,oy);right,bottom=min(x+w,ox+ow),min(y+h,oy+oh)
            if right<=left or bottom<=top:continue
            selected=other['mask'][top-oy:bottom-oy,left-ox:right-ox]>0
            local=mask[top-y:bottom-y,left-x:right-x]
            removed=removed or bool(np.any(local[selected]))
            local[selected]=0
            excluded[top-y:bottom-y,left-x:right-x]|=selected
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        pieces=[]
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA]<20:continue
            component=(labels==label).astype(np.uint8)
            piece=measure_mask(component,(x,y),frame_shape=None)
            if piece is not None and piece.length_px is not None:
                for run in _visible_runs(piece.path,(x,y),excluded):pieces.append((run,polyline_length(run)))
        if not pieces:continue
        path,length=max(pieces,key=lambda piece:piece[1])
        c.visible_path=path
        c.visible_length_px=length
        c.visible_note=('longest_visible_segment_overlap_excluded' if removed else
                        'longest_visible_segment_frame_cut' if c.reason=='edge' else
                        'longest_visible_connected_segment')


def visible_segment_count(candidates):
    return sum(c.counted and c.length_px is None and c.visible_length_px is not None for c in candidates)
