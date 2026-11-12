"""Private, resumable per-image human correction queue. No automatic training."""
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import uuid
from zipfile import ZipFile,ZIP_DEFLATED,BadZipFile
import cv2
import numpy as np
from PIL import Image


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.review-',suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def read_source(path):
    with ZipFile(path) as z:
        entries=z.infolist()
        if len(entries)>20 or sum(e.file_size for e in entries)>64*1024**2:raise ValueError('画像ZIPが大きすぎます。')
        images=[e for e in entries if e.filename.lower().endswith('.png')]
        notes=[e for e in entries if e.filename.lower().endswith('.json')]
        if len(images)!=1 or len(notes)!=1:raise ValueError('元画像1枚の検証ZIPが必要です。')
        raw=z.read(images[0]);info=json.loads(z.read(notes[0]))
    if not isinstance(info,dict):raise ValueError('画像の記録が不正です。')
    for key in ['model_info','collection']:
        if info.get(key) is not None and not isinstance(info[key],dict):raise ValueError('画像の補足情報が不正です。')
    if info.get('source_sha256') and hashlib.sha256(raw).hexdigest()!=info['source_sha256']:raise ValueError('元画像のハッシュが一致しません。')
    with Image.open(io.BytesIO(raw)) as image:
        if image.width*image.height>40_000_000:raise ValueError('画像が大きすぎます。')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('画像を開けません。')
    return raw,image,info


def resolve_source(data,record,index=None):
    """Find a verified original after a portable data folder has moved PCs."""
    data=Path(data).resolve();digest=record['source_sha256']
    directories=[data/'samples',data/'archive_batch/samples']
    supplied=Path(record.get('source_zip','')).resolve()
    if supplied.parent in {p.resolve() for p in directories} and supplied.is_file() and not supplied.is_symlink():
        raw,image,info=read_source(supplied)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('元画像が変更されました。')
        return supplied
    index={} if index is None else index
    if not index.get('_scanned'):
        for directory in directories:
            if directory.is_symlink():continue
            for path in sorted(directory.glob('*.zip')):
                if path.is_symlink():continue
                try:
                    raw,image,info=read_source(path);index[hashlib.sha256(raw).hexdigest()]=path
                except (OSError,ValueError,KeyError,RuntimeError,BadZipFile,cv2.error):continue
        index['_scanned']=True
    if digest not in index:raise ValueError('修正した元画像が見つかりません。data/samplesとdata/archive_batch/samplesも引き継いでください。')
    return index[digest]


def predictions(info,shape):
    h,w=shape[:2];crop=info.get('crop_xyxy',[0,0,w,h]);offset=np.zeros(2)
    if isinstance(crop,list) and len(crop)==4 and crop[2]-crop[0]==w and crop[3]-crop[1]==h:offset=np.asarray(crop[:2])
    objects=[]
    model=info.get('model_info') or {}
    for p in model.get('instances',[]):
        try:
            points=np.asarray(p['contour_source_xy'],dtype=float).reshape(-1,2)-offset
            if len(points)<3 or not np.isfinite(points).all():continue
            mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
            cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not cs:continue
            polygon=max(cs,key=cv2.contourArea).reshape(-1,2).tolist()
            if len(polygon)<3:continue
            objects.append({'id':uuid.uuid4().hex[:12],'polygon':polygon,'origin':'model',
                            'visibility':'unknown','review_status':'unreviewed'})
        except (KeyError,ValueError,TypeError):continue
    return objects


class ReviewQueue:
    def __init__(self,data):
        self.data=Path(data).resolve();self.folder=self.data/'improvement';self.folder.mkdir(parents=True,exist_ok=True)
        self.items=[];self.position=-1;self.current=None;self.objects=[];self.rejected=[];self.history=[];self.errors=[]
    def record_path(self,digest):return self.folder/'reviews'/(digest+'.json')
    def scan(self):
        found={}
        for directory in [self.data/'samples',self.data/'archive_batch/samples']:
            if directory.is_symlink():continue
            for path in sorted(directory.glob('*.zip')):
                if path.is_symlink():continue
                try:
                    raw,image,info=read_source(path);digest=hashlib.sha256(raw).hexdigest()
                    previous=self.record_path(digest)
                    if previous.is_file() and json.loads(previous.read_text(encoding='utf-8')).get('status')=='completed':continue
                    instances=(info.get('model_info') or {}).get('instances') or []
                    if not isinstance(instances,list):raise ValueError('候補一覧の形式が不正です。')
                    priority=len(instances)
                    item={'path':str(path),'sha256':digest,'width':image.shape[1],'height':image.shape[0],
                          'priority':priority,'title':str((info.get('collection') or {}).get('video_title',path.stem))}
                    # Prefer exports containing editable model contours.
                    if digest not in found or priority>=found[digest]['priority']:found[digest]=item
                except (OSError,ValueError,KeyError,RuntimeError,BadZipFile,cv2.error) as exc:self.errors.append(str(exc))
        self.items=sorted(found.values(),key=lambda r:(r['priority']!=0,r['title'],r['sha256']))
        return self.items
    def next(self):
        if self.current is not None:self.save(False)
        self.position+=1;self.history=[]
        if self.position>=len(self.items):self.current=None;self.objects=[];return None
        self.current=self.items[self.position];raw,self.image,self.info=read_source(self.current['path'])
        if hashlib.sha256(raw).hexdigest()!=self.current['sha256']:raise ValueError('元画像が変更されました。')
        path=self.record_path(self.current['sha256'])
        self.loaded_draft=path.is_file()
        if path.is_file():
            record=json.loads(path.read_text(encoding='utf-8'));self.objects=record['objects'];self.rejected=record.get('rejected_predictions',[])
        else:self.objects=predictions(self.info,self.image.shape);self.rejected=[]
        self.original=record.get('original_predictions',predictions(self.info,self.image.shape)) if path.is_file() else copy.deepcopy(self.objects)
        self.baseline_available=(record.get('baseline_available',bool(self.original)) if path.is_file() else
                                 isinstance((self.info.get('model_info') or {}).get('instances'),list))
        self.save(False);return self.image
    def change(self):self.history.append(copy.deepcopy((self.objects,self.rejected)))
    def add(self,points,visibility='unknown'):
        p=np.asarray(points,dtype=float).reshape(-1,2)
        if len(p)<3 or not np.isfinite(p).all():raise ValueError('輪郭を3点以上で囲んでください。')
        if abs(cv2.contourArea(p.astype(np.float32)))<1:raise ValueError('体を囲む輪郭を指定してください。')
        if np.any(p<0) or np.any(p[:,0]>=self.current['width']) or np.any(p[:,1]>=self.current['height']):raise ValueError('輪郭は画像内で指定してください。')
        self.change();self.objects.append({'id':uuid.uuid4().hex[:12],'polygon':p.tolist(),'origin':'human','visibility':visibility,'review_status':'edited'})
        self.save(False)
    def delete(self,index):
        self.change();obj=self.objects.pop(index)
        if obj['origin']=='model':self.rejected.append(obj)
        self.save(False)
    def replace(self,index,points):
        old=self.objects[index];self.add(points,old['visibility']);new=self.objects.pop();self.objects[index]={**new,'id':old['id']};self.save(False)
    def visibility(self,index,value):
        if value not in {'full','partial','unknown'}:raise ValueError('全身／部分／不明から選んでください。')
        self.change();self.objects[index]['visibility']=value;self.objects[index]['review_status']='edited';self.save(False)
    def undo(self):
        if self.history:self.objects,self.rejected=self.history.pop();self.save(False)
    def save(self,completed=False):
        if self.current is None:return
        record={'schema':1,'source_sha256':self.current['sha256'],'source_zip':self.current['path'],
                'image_size':[self.current['width'],self.current['height']],
                'status':'completed' if completed else 'draft','objects':copy.deepcopy(self.objects),
                'rejected_predictions':copy.deepcopy(self.rejected),
                'original_predictions':copy.deepcopy(self.original),
                'baseline_available':self.baseline_available,
                'collection':self.info.get('collection',{}),'timestamp':datetime.now(timezone.utc).isoformat(),
                'unreviewed_pixels':'ignore','note':'Visible-instance annotations; no hidden full bodies inferred.'}
        if completed:
            for obj in record['objects']:obj['review_status']='human_confirmed'
        atomic(self.record_path(self.current['sha256']),record)
    def complete(self):
        self.save(True);self.current=None;return self.next()
    def export(self,max_bytes=24*1024**2):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
        outputs=[];count=0;z=None;temporary=None;source_index={}
        def close_part():
            nonlocal z,temporary
            z.close();target=temporary.with_suffix('.zip');os.replace(temporary,target);outputs.append(target)
            z=None;temporary=None
        try:
            for path in sorted((self.folder/'reviews').glob('*.json')):
                if path.is_symlink():continue
                r=json.loads(path.read_text(encoding='utf-8'))
                if r.get('status')!='completed':continue
                source=resolve_source(self.data,r,source_index)
                raw,image,info=read_source(source);digest=hashlib.sha256(raw).hexdigest()
                if digest!=r['source_sha256']:raise ValueError('元画像が変更されました。')
                cleaned={k:v for k,v in r.items() if k!='source_zip'}
                note=json.dumps(cleaned,ensure_ascii=False).encode('utf-8')
                required=len(raw)+len(note)+4096
                if required>max_bytes:raise ValueError('1枚分の修正データがZIP上限を超えています。')
                if z is not None and z.fp.tell()+required>max_bytes:close_part()
                if z is None:
                    temporary=self.folder/(f'reviewed_{stamp}_part{len(outputs)+1:02}.tmp')
                    z=ZipFile(temporary,'w',ZIP_DEFLATED)
                z.writestr(digest+'/source.png',raw);z.writestr(digest+'/review.json',note);count+=1
            if not count:raise ValueError('完了した画像がまだありません。')
            if z is not None:close_part()
        finally:
            if z is not None:z.close()
            if temporary is not None:temporary.unlink(missing_ok=True)
        return outputs,count
