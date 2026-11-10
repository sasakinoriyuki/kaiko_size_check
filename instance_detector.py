"""Local, per-silkworm segmentation and conservative visible-length estimates."""
import hashlib
import importlib
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from ai_measurement import content_view
from instance_geometry import merge_views,measure_proposals
from instance_counting import assign_instance_ids
from visible_measurement import estimate_visible_segments,visible_segment_count

MODEL_URL='https://raw.githubusercontent.com/BillTheDing/YOLOv8-DSRI/18085596e7e950cac86b377807f05046c6b9a2cf/yolov8-main/YOLOv8-DSRI.pt'
MODEL_SHA256='ddd6e7d9fb2911588f4e6d13fc78a2680f585c95ec13c1e651ba6b3197a8e217'
MODEL_BYTES=7295538
MODEL_FILE='silkworm_dsri.pt'


def ensure_instance_model(path=None,fetch=None):
    from secure_updates import download
    from update_and_start import is_link
    path=Path(path) if path else Path(__file__).parent/'models'/MODEL_FILE
    if any(is_link(p) for p in [path,*path.parents]):
        raise ValueError('モデル保存先にリンクされたフォルダがあります。')
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==MODEL_SHA256:return path
    raw=(fetch or download)(MODEL_URL,MODEL_BYTES,timeout=60)
    if len(raw)!=MODEL_BYTES or hashlib.sha256(raw).hexdigest()!=MODEL_SHA256:
        raise ValueError('蚕検出モデルの取得内容が一致しません。')
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix='.silkworm-model-',dir=path.parent)
    try:
        with os.fdopen(handle,'wb') as f:f.write(raw)
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)
    return path


def load_detector(path,device='cpu'):
    """Accept the pinned checkpoint with a narrow, explicit safe-global mapping."""
    import torch
    from kaiko_dsri.nn.modules.head import Detect
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ValueError('蚕検出モデルが破損しています。')
    def guarded_getattr(obj,name):
        if obj is Detect and name=='forward':return Detect.forward
        raise ValueError('Unexpected checkpoint attribute')
    allowed=[]
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        if name=='builtins.getattr':allowed.append((guarded_getattr,name));continue
        if not name.startswith(('torch.nn.modules.','ultralytics.')) and name not in {'numpy.dtype','numpy.core.multiarray.scalar'}:
            raise ValueError('Unexpected checkpoint global: '+name)
        module,attribute=name.rsplit('.',1)
        if module=='numpy.core.multiarray':module='numpy._core.multiarray'
        module=module.replace('ultralytics.','kaiko_dsri.',1)
        allowed.append((getattr(importlib.import_module(module),attribute),name))
    allowed.extend([np.dtypes.Float16DType,np.dtypes.Float32DType,np.dtypes.Float64DType])
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=checkpoint['model'].float().eval().to(device)
    head=model.model[-1];head.export=False;head.dynamic=True;head.stride=head.stride.to(device)
    return model


def neutral_boost(frame):
    """Improve dark grey body contrast without resizing or changing coordinates."""
    a=frame.astype(np.float32);s=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32)
    neutral=np.clip((115-s)/100,0,1)*np.clip((np.maximum(a[:,:,0],a[:,:,2])-a[:,:,1]+30)/55,0,1)
    return np.clip(a+neutral[:,:,None]*125,0,255).astype(np.uint8)


def tile_positions(extent,size):
    if extent<=size:return [0]
    result=list(range(0,extent-size+1,max(1,int(size*.75))))
    if result[-1]!=extent-size:result.append(extent-size)
    return result


def automatic_camera_roi(frame):
    """Recognise a dark, neutral OBS sidebar; keep plain camera inputs whole."""
    h,w=frame.shape[:2]
    if w<300:return (0,0,w,h)
    mx=frame.max(2);mn=frame.min(2)
    neutral=(mx<=60)&((mx.astype(np.int16)-mn)<=6)
    if neutral[:,:w//2].mean()>.3:return (0,0,w,h)
    profile=cv2.GaussianBlur(neutral.mean(0).astype(np.float32)[None],(31,1),0)[0]
    for x in range(int(w*.58),int(w*.88)):
        if profile[x:x+max(10,w//80)].mean()>.62 and neutral[:,x:].mean()>.50:
            return (0,0,max(1,x-4),h)
    return (0,0,w,h)


class LocalInstanceSegmenter:
    def __init__(self,model_path=None,texture_model_path=None):
        import torch
        torch.set_num_threads(3)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=load_detector(ensure_instance_model(model_path),self.device)
        self.model_sha256=MODEL_SHA256
        self.metadata={'model_version':'DSRI instances + separate visible-segment lengths v3',
                       'inference_threshold':.35,'pretrained':True,'model_source':'YOLOv8-DSRI / AGPL-3.0',
                       'measurement_status':'automatic_estimate','stages':'1-5; hatchling validation pending'}
        self.last_info={}
        self.texture_model_path=Path(texture_model_path) if texture_model_path else Path(__file__).parent/'models/kaiko_texture.pt'
        self.texture_model=None
        self.metadata['texture_veto_sha256']=(hashlib.sha256(self.texture_model_path.read_bytes()).hexdigest()
                                              if self.texture_model_path.is_file() else None)

    def predict(self,frame,size=960,confidence=.35):
        import torch
        h,w=frame.shape[:2];scale=min(size/w,size/h)
        nw,nh=round(w*scale),round(h*scale);dx,dy=(size-nw)//2,(size-nh)//2
        image=np.full((size,size,3),114,np.uint8)
        image[dy:dy+nh,dx:dx+nw]=cv2.resize(frame,(nw,nh))
        tensor=torch.from_numpy(image[:,:,::-1].copy().transpose(2,0,1)[None]).to(self.device).float()/255
        with torch.inference_mode():
            output=self.model(tensor)
            prediction=output[0][0].T;proto=output[1][2][0]
            nc=self.model.model[-1].nc
            values=prediction.detach().cpu().numpy();scores=values[:,4:4+nc].max(1)
            selected=np.flatnonzero(scores>=confidence)
            if not len(selected):return []
            values=values[selected];scores=scores[selected]
            boxes=values[:,:4].copy();boxes[:,:2]-=boxes[:,2:]/2
            keep=np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),confidence,.65),dtype=int).reshape(-1)
            result=[]
            for i in keep:
                bx,by,bw,bh=boxes[i]
                x0,y0,x1,y1=np.asarray([bx-dx,by-dy,bx+bw-dx,by+bh-dy])/scale
                left,top=max(0,int(np.floor(x0))),max(0,int(np.floor(y0)))
                right,bottom=min(w,int(np.ceil(x1))),min(h,int(np.ceil(y1)))
                if right<=left or bottom<=top:continue
                coefficients=prediction[int(selected[i]),4+nc:]
                logits=(coefficients@proto.reshape(proto.shape[0],-1)).reshape(proto.shape[1:]).cpu().numpy()
                logits=cv2.resize(logits,(size,size))[dy:dy+nh,dx:dx+nw]
                native=cv2.resize(logits,(w,h))
                mask=(native[top:bottom,left:right]>0).astype(np.uint8)
                if mask.sum()<10:continue
                result.append({'mask':mask,'offset':(left,top),'score':float(scores[i]),
                               'box':(left,top,right,bottom),'source':'whole'})
            return result

    def analyze(self,frame,settings,roi=None):
        settings.validate()
        effective=roi or automatic_camera_roi(frame)
        view,offset=content_view(frame,effective)
        if view is None:return [],self.device
        proposals=self.predict(view)
        sizes=[max(p['mask'].shape) for p in proposals]
        small=len(proposals)<8 or (sizes and np.median(sizes)<150)
        detailed=getattr(settings,'ai_search','standard')=='fine'
        views=1
        if small or detailed or max(view.shape[:2])>1152:
            side=(384 if detailed else 512) if small else (896 if detailed else 1152)
            search=neutral_boost(view) if small else view
            for y in tile_positions(view.shape[0],side):
                for x in tile_positions(view.shape[1],side):
                    patch=search[y:y+side,x:x+side]
                    original=view[y:y+side,x:x+side]
                    views+=1
                    for p in self.predict(patch,640 if small else 960,.25 if small else .35):
                        left,top,right,bottom=p['box']
                        # Keep real frame-edge animals for counting; reject only
                        # masks cut by an artificial interior tile boundary.
                        h,w=patch.shape[:2]
                        if ((left<2 and x>0) or (top<2 and y>0)
                            or (right>w-2 and x+w<view.shape[1])
                            or (bottom>h-2 and y+h<view.shape[0])):continue
                        if small:
                            sat=cv2.cvtColor(original[top:bottom,left:right],cv2.COLOR_BGR2HSV)[:,:,1]
                            if (sat[p['mask']>0]<110).mean()<.65:continue
                        p['offset']=(p['offset'][0]+x,p['offset'][1]+y)
                        p['source']='small_tile' if small else 'detail_tile'
                        proposals.append(p)
        if small:
            # The grey-body enhancement also brightens leaf veins. The
            # channel's own texture model acts only as a foreground veto;
            # it never joins masks or invents an individual body.
            from silkworm_scratch import load_model,predict
            if self.texture_model is None:
                self.texture_model,_=load_model(self.texture_model_path)
            probability=predict(self.texture_model,cv2.cvtColor(view,cv2.COLOR_BGR2RGB))
            guarded=[]
            for p in proposals:
                x,y=p['offset'];h,w=p['mask'].shape
                values=probability[y:y+h,x:x+w][p['mask']>0]
                if len(values) and np.median(values)>=.5:guarded.append(p)
            proposals=guarded
        filtered=merge_views(proposals,view.shape)
        candidates=measure_proposals(filtered,view.shape,merged=True)
        estimate_visible_segments(filtered,candidates)
        for c in candidates:
            c.contour=c.contour+np.asarray(offset)
            if c.path is not None:c.path=c.path+np.asarray(offset)
            if c.visible_path is not None:c.visible_path=c.visible_path+np.asarray(offset)
        counts=assign_instance_ids(candidates)
        self.last_info={'effective_roi':list(effective),'small_body_search':bool(small),
                        'measurement_status':'automatic_estimate','detected_count':len(candidates),
                        'measurable_count':sum(c.length_px is not None for c in candidates),
                        'confidence_threshold':.35,'inference_views':views,
                        'count_thresholds':{'whole':.6,'small_with_texture_guard':.6,'detail_only':.85},
                        'visible_segment_count':visible_segment_count(candidates),
                        'visible_segment_policy':'longest_continuous_piece; full-length means exclude partials',
                        **counts}
        return candidates,self.device
