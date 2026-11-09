"""Source-pixel centreline estimates for separately detected visible bodies.

Geometry cannot establish biological identity or reveal an occluded endpoint.
Keep those decisions separate from the centreline calculation.
"""
import heapq
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize

from measurement import Candidate, _extend, _graph, polyline_length


def _distances(graph, origin):
    distance={origin:0.0};previous={origin:None};queue=[(0.0,origin)]
    while queue:
        cost,node=heapq.heappop(queue)
        if cost!=distance[node]:continue
        for neighbor in graph[node]:
            step=math.hypot(neighbor[0]-node[0],neighbor[1]-node[1])
            new=cost+step
            if new<distance.get(neighbor,float('inf')):
                distance[neighbor]=new;previous[neighbor]=node
                heapq.heappush(queue,(new,neighbor))
    return distance,previous


def _main_path(mask):
    # Subpixel mask interpolation can leave one-pixel whiskers. Remove only
    # these narrow protrusions before skeletonisation, at source resolution.
    radius=max(1,min(3,int(distance_transform_edt(mask>0).max()*.08)))
    cleaned=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    graph=_graph(skeletonize(cleaned>0))
    if len(graph)<5:return None,'short'
    distance,_=_distances(graph,next(iter(graph)))
    if len(distance)!=len(graph):return None,'fragmented'
    if sum(len(n) for n in graph.values())//2>=len(graph):return None,'loop'
    first=max(distance,key=distance.get)
    distance,parents=_distances(graph,first)
    last=max(distance,key=distance.get);length=distance[last]
    route=[];node=last
    while node is not None:route.append(node);node=parents[node]
    main=set(route);radii=distance_transform_edt(mask>0)
    # Tiny legs/contour twigs may be removed; a substantial second body may not.
    for end in [p for p,n in graph.items() if len(n)==1 and p not in main]:
        node=end;previous=None;branch=[];branch_length=0.0
        while node not in main:
            branch.append(node)
            choices=[p for p in graph[node] if p!=previous]
            if len(choices)!=1:return None,'branch'
            target=choices[0];branch_length+=math.dist(node,target)
            previous,node=node,target
        radius=max(1.0,float(radii[node]))
        if branch_length>max(2.5,1.8*radius,.025*length):return None,'branch'
        if branch_length>1.5*radius and np.median([radii[p] for p in branch])>.65*radius:
            return None,'branch'
    xy=np.asarray([(x,y) for y,x in route],dtype=float)
    smooth=gaussian_filter1d(xy,1.2,axis=0,mode='nearest')
    smooth[0],smooth[-1]=xy[0],xy[-1]
    width=float(np.median([radii[p] for p in route]))
    n=min(max(3,round(width*.6)),len(xy)-1)
    start=_extend(xy[0],xy[0]-xy[n],mask)
    end=_extend(xy[-1],xy[-1]-xy[-1-n],mask)
    path=np.vstack([start,smooth,end])
    pixels=np.rint(path).astype(int)
    if not np.all(mask[pixels[:,1],pixels[:,0]]):return None,'outside_mask'
    return path,'ok'


def measure_mask(mask, offset=(0,0), frame_shape=None, minimum_aspect=3.5):
    """Measure one instance crop; offset maps it back to the original frame."""
    raw=np.asarray(mask)
    if raw.ndim!=2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError('Expected a finite 2D mask')
    binary=(raw>0).astype(np.uint8)
    # Repair a one-pixel crack from mask quantisation, before assessing islands.
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:return None
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    area=int(binary.sum());main_area=int(stats[largest,cv2.CC_STAT_AREA])
    ys,xs=np.nonzero(labels==largest)
    x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    component=(labels[y0:y1,x0:x1]==largest).astype(np.uint8)
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea)+np.asarray([x0+offset[0],y0+offset[1]])
    candidate=Candidate(contour,None,None,'fragmented')
    if main_area<10 or main_area/max(area,1)<.97:return candidate
    if frame_shape is not None:
        h,w=frame_shape[:2]
        if x0+offset[0]<=1 or y0+offset[1]<=1 or x1+offset[0]>=w-1 or y1+offset[1]>=h-1:
            candidate.reason='edge';return candidate
    padded=np.pad(component,3)
    # Smooth at most one pixel. Large gaps and separate body fragments stay rejected.
    padded=cv2.morphologyEx(padded,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    holes=1-padded
    n,regions,sizes,_=cv2.connectedComponentsWithStats(holes,8)
    exterior=set(regions[0])|set(regions[-1])|set(regions[:,0])|set(regions[:,-1])
    for i in range(1,n):
        if i not in exterior and sizes[i,cv2.CC_STAT_AREA]<=max(4,.025*main_area):padded[regions==i]=1
    path,reason=_main_path(padded)
    candidate.reason=reason
    if path is None:return candidate
    length=polyline_length(path)
    if length<8 or length*length/main_area<minimum_aspect:
        candidate.reason='shape';return candidate
    candidate.path=path+np.asarray([offset[0]+x0-3,offset[1]+y0-3])
    candidate.length_px=length
    return candidate


def intersection(a,b):
    """Intersection area between cropped instance masks in source coordinates."""
    ax,ay=a['offset'];bx,by=b['offset']
    h,w=a['mask'].shape;hh,ww=b['mask'].shape
    x0,y0=max(ax,bx),max(ay,by);x1,y1=min(ax+w,bx+ww),min(ay+h,by+hh)
    if x1<=x0 or y1<=y0:return 0
    return int(np.logical_and(a['mask'][y0-ay:y1-ay,x0-ax:x1-ax],b['mask'][y0-by:y1-by,x0-bx:x1-bx]).sum())


def deduplicate(proposals):
    """Merge tile duplicates using masks, preserving nearby distinct bodies."""
    kept=[]
    for p in sorted(proposals,key=lambda p:p['score'],reverse=True):
        area=int(np.count_nonzero(p['mask']))
        if not area:continue
        duplicate=None
        for i,k in enumerate(kept):
            overlap=intersection(p,k)
            if overlap/max(1,min(area,int(np.count_nonzero(k['mask']))))>.72:
                duplicate=i;break
        if duplicate is None:kept.append(p)
        elif area>1.25*np.count_nonzero(kept[duplicate]['mask']) and p['score']>=.75*kept[duplicate]['score']:
            kept[duplicate]=p
    return kept


def measure_proposals(proposals,frame_shape):
    proposals=deduplicate(proposals)
    result=[]
    for i,p in enumerate(proposals):
        c=measure_mask(p['mask'],p['offset'],frame_shape)
        if c is None:continue
        c.confidence=float(p['score'])
        if c.path is not None and c.confidence<.6:
            c.path=None;c.length_px=None;c.reason='low_confidence'
        if c.path is not None:
            # An endpoint hidden behind another detected body cannot yield full length.
            endpoints=np.rint(c.path[[0,-1]]).astype(int)
            for j,other in enumerate(proposals):
                if i==j:continue
                if intersection(p,other)/max(1,np.count_nonzero(p['mask']))>.04:
                    c.path=None;c.length_px=None;c.reason='overlap';break
                ox,oy=other['offset'];mh,mw=other['mask'].shape
                for x,y in endpoints:
                    xx,yy=x-ox,y-oy
                    if 0<=xx<mw and 0<=yy<mh:
                        patch=other['mask'][max(0,yy-2):yy+3,max(0,xx-2):xx+3]
                        if np.any(patch):
                            c.path=None;c.length_px=None;c.reason='overlap_at_end';break
                if c.path is None:break
        result.append(c)
    return result
