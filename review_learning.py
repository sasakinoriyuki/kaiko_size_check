"""Turn completed human corrections into sparse supervision and honest reports.

No predictions become training labels without explicit review completion.
No candidate model is installed or uploaded by this module.
"""
import copy
import hashlib
import io
import json
import re
import shutil
from pathlib import Path
import uuid
from urllib.parse import urlparse,parse_qs
from zipfile import ZipFile,ZIP_DEFLATED
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from improvement_review import atomic,read_source,resolve_source

SHA=re.compile(r'^[0-9a-f]{64}$')


def validate_review(record,image):
    if record.get('schema')!=1 or record.get('status')!='completed':raise ValueError('完了した修正データだけを使えます。')
    h,w=image.shape[:2]
    if record.get('image_size')!=[w,h]:raise ValueError('修正と元画像の大きさが一致しません。')
    if not SHA.fullmatch(str(record.get('source_sha256',''))):raise ValueError('画像の識別値が不正です。')
    for key in ['objects','original_predictions','rejected_predictions']:
        objects=record.get(key,[])
        if not isinstance(objects,list) or len(objects)>5000:raise ValueError('輪郭一覧の形式が不正です。')
        ids=set()
        for obj in objects:
            if not isinstance(obj,dict) or not isinstance(obj.get('id'),str) or obj['id'] in ids:raise ValueError('輪郭の番号が不正です。')
            ids.add(obj['id'])
            p=np.asarray(obj.get('polygon'),dtype=float)
            if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=50000 or not np.isfinite(p).all():raise ValueError('輪郭の座標が不正です。')
            if np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h) or cv2.contourArea(p.astype(np.float32))<1:raise ValueError('輪郭が画像の外か、面積がありません。')
            if key=='objects' and (obj.get('review_status')!='human_confirmed' or obj.get('visibility') not in {'full','partial','unknown'}):raise ValueError('未確認の輪郭は学習に使えません。')
    if not isinstance(record.get('collection',{}),dict):raise ValueError('動画の情報が不正です。')
    return record


def read_completed(data):
    data=Path(data).resolve();reviews=data/'improvement/reviews';items=[];source_index={}
    for path in sorted(reviews.glob('*.json')):
        if path.is_symlink():continue
        review_bytes=path.read_bytes();record=json.loads(review_bytes)
        if record.get('status')!='completed':continue
        source=resolve_source(data,record,source_index)
        raw,image,info=read_source(source);digest=hashlib.sha256(raw).hexdigest()
        if digest!=record.get('source_sha256') or path.stem!=digest:raise ValueError('修正した元画像が変更されています。')
        validate_review(record,image)
        items.append({'record':record,'source_path':str(source),'review_sha256':hashlib.sha256(review_bytes).hexdigest()})
    return items


def import_exports(data,archives):
    data=Path(data).resolve();result={'imported':0,'duplicates':0,'conflicts':0}
    for archive in archives:
        with ZipFile(archive) as z:
            members=z.infolist()
            if len(members)>10000 or sum(m.file_size for m in members)>128*1024**2:raise ValueError('修正ZIPが大きすぎます。')
            names=[m.filename for m in members]
            if len(set(names))!=len(names) or any(not re.fullmatch(r'[0-9a-f]{64}/(?:source\.png|review\.json)',n) for n in names):raise ValueError('修正ZIPの構成が不正です。')
            hashes=sorted({n.split('/')[0] for n in names})
            for digest in hashes:
                raw=z.read(digest+'/source.png');record=json.loads(z.read(digest+'/review.json'))
                if hashlib.sha256(raw).hexdigest()!=digest or record.get('source_sha256')!=digest:raise ValueError('元画像のハッシュが一致しません。')
                with Image.open(io.BytesIO(raw)) as probe:
                    if probe.width*probe.height>40_000_000:raise ValueError('画像が大きすぎます。')
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError('元画像を開けません。')
                validate_review(record,image)
                destination=data/'improvement/reviews'/(digest+'.json')
                if destination.exists():
                    old=json.loads(destination.read_text(encoding='utf-8'))
                    keys=['status','objects','rejected_predictions','original_predictions','image_size','collection']
                    if all(old.get(k,[] if k.endswith('predictions') else None)==record.get(k,[] if k.endswith('predictions') else None) for k in keys):result['duplicates']+=1;continue
                    fingerprint=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()[:16]
                    atomic(data/'improvement/import_conflicts'/(digest+'_'+fingerprint+'.json'),record)
                    (data/'improvement/import_conflicts'/(digest+'_'+fingerprint+'.png')).write_bytes(raw)
                    result['conflicts']+=1;continue
                source=data/'samples'/('review_import_'+digest+'.zip');source.parent.mkdir(parents=True,exist_ok=True)
                info={'source_sha256':digest,'source_resolution':record['image_size'],'collection':record.get('collection',{}),
                      'model_info':{'instances':[{'contour_source_xy':o['polygon']} for o in record.get('original_predictions',[])]}}
                temporary=source.with_suffix('.tmp')
                try:
                    with ZipFile(temporary,'w',ZIP_DEFLATED) as out:out.writestr('source.png',raw);out.writestr('source.json',json.dumps(info,ensure_ascii=False))
                    temporary.replace(source)
                finally:temporary.unlink(missing_ok=True)
                atomic(destination,{**record,'source_zip':str(source)})
                result['imported']+=1
    return result


def source_keys(record):
    c=record.get('collection',{});keys=set()
    video=c.get('video_id','')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',str(video)):
        url=urlparse(str(c.get('video_url','')))
        video=url.path.strip('/') if url.hostname=='youtu.be' else parse_qs(url.query).get('v',[''])[0]
    if re.fullmatch(r'[A-Za-z0-9_-]{11}',str(video)):keys.add('video:'+video)
    date=str(c.get('upload_date') or c.get('recording_date') or '')
    if not re.fullmatch(r'20\d{6}',date):
        match=re.search(r'(?<!\d)(20\d{6})(?!\d)',str(c.get('video_title','')));date=match[1] if match else ''
    if date:keys.add('date:'+date)
    return keys or {'unknown_source'}


def assign_splits(items,registry):
    """Union same videos/days; never move an old training source to validation."""
    components=[]
    for item in items:
        keys=source_keys(item['record']);touch=[g for g in components if g&keys]
        for g in touch:components.remove(g);keys|=g
        components.append(keys)
    mapping=copy.deepcopy(registry.get('keys',{}));groups={}
    eligible=[g for g in components if 'unknown_source' not in g]
    if len(eligible)<2 and not mapping:return {k:'pending' for g in components for k in g},registry
    fresh=[g for g in eligible if not any(k in mapping for k in g)]
    initial_validation=max(fresh,key=lambda g:hashlib.sha256('|'.join(sorted(g)).encode()).hexdigest()) if fresh and not mapping else None
    for g in components:
        if 'unknown_source' in g:
            for k in g:groups[k]='pending'
            continue
        old={mapping[k] for k in g if k in mapping}
        if len(old)>1 or 'quarantine' in old:split='quarantine'
        elif old:split=next(iter(old))
        elif initial_validation is g:split='validation'
        else:split='validation' if int(hashlib.sha256('|'.join(sorted(g)).encode()).hexdigest()[:8],16)%5==0 and initial_validation is None else 'train'
        for k in g:mapping[k]=groups[k]=split
    return groups,{'schema':1,'keys':mapping}


def masks(record,shape):
    positive=np.zeros(shape[:2],np.uint8);background=np.zeros_like(positive)
    for obj in record.get('objects',[]):cv2.fillPoly(positive,[np.rint(obj['polygon']).astype(np.int32)],1)
    for obj in record.get('rejected_predictions',[]):cv2.fillPoly(background,[np.rint(obj['polygon']).astype(np.int32)],1)
    # Deleting a merged proposal must not teach the corrected worms as background.
    protected=cv2.dilate(positive,np.ones((5,5),np.uint8))>0
    negative=(cv2.erode(background,np.ones((3,3),np.uint8))>0)&~protected
    labels=np.full(shape[:2],255,np.uint8);labels[negative]=0;labels[positive>0]=1
    return labels


def polygon_crop(obj):
    p=np.rint(obj['polygon']).astype(np.int32);x,y,w,h=cv2.boundingRect(p)
    m=np.zeros((h,w),np.uint8);cv2.fillPoly(m,[p-[x,y]],1)
    return m,(x,y)


def compare_annotations(record):
    truth=record.get('objects',[]);original=record.get('original_predictions',[])
    a=[polygon_crop(p) for p in original];b=[polygon_crop(p) for p in truth]
    iou=np.zeros((len(a),len(b)))
    for i,(m,(x,y)) in enumerate(a):
        for j,(n,(xx,yy)) in enumerate(b):
            left,top=max(x,xx),max(y,yy);right,bottom=min(x+m.shape[1],xx+n.shape[1]),min(y+m.shape[0],yy+n.shape[0])
            if right<=left or bottom<=top:continue
            common=np.logical_and(m[top-y:bottom-y,left-x:right-x],n[top-yy:bottom-yy,left-xx:right-xx]).sum()
            iou[i,j]=common/max(1,int(m.sum())+int(n.sum())-common)
    pairs=[]
    if iou.size:
        weights=(iou>=.5)*(min(iou.shape)+1)+iou
        rows,cols=linear_sum_assignment(-weights)
        pairs=[(int(i),int(j)) for i,j in zip(rows,cols) if iou[i,j]>=.5]
    old={o['id']:o for o in original};new={o['id']:o for o in truth}
    comparable=record.get('baseline_available',bool(original))
    return {'reviewed_objects':len(truth),'baseline_available':bool(comparable),
            'original_proposals':len(original) if comparable else None,'matched_at_iou_0_5':len(pairs) if comparable else None,
            'unmatched_reviewed':len(truth)-len(pairs) if comparable else None,'unmatched_original':len(original)-len(pairs) if comparable else None,
            'added_objects':len(new.keys()-old.keys()),'deleted_objects':len(old.keys()-new.keys()),
            'changed_outlines':sum(old[k]['polygon']!=new[k]['polygon'] for k in old.keys()&new.keys()),
            'scope':'proposal contours versus reviewed visible objects; not deployed headcount accuracy'}


def prepare(data):
    data=Path(data).resolve();folder=data/'improvement';items=read_completed(data)
    if not items:raise ValueError('改善モードで完了した画像がまだありません。')
    registry_path=folder/'split_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {'schema':1,'keys':{}}
    groups,registry=assign_splits(items,registry)
    atomic(registry_path,registry)
    identity=hashlib.sha256(json.dumps([(i['record']['source_sha256'],i['review_sha256']) for i in items],sort_keys=True).encode()+json.dumps(groups,sort_keys=True).encode()).hexdigest()[:20]
    target=folder/'datasets'/identity
    if (target/'report.json').is_file():
        summary=json.loads((target/'report.json').read_text(encoding='utf-8'))
        atomic(folder/'latest_dataset.json',{'path':str(target),'summary':{k:v for k,v in summary.items() if k!='details'}})
        return target,summary
    stage=target.parent/('.prepare-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    frames=[];reports=[]
    try:
        for item in items:
            r=item['record'];digest=r['source_sha256'];raw,image,info=read_source(item['source_path']);h,w=image.shape[:2]
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('準備中に元画像が変更されました。')
            labels=masks(r,image.shape);image_name=digest+'.png';label_name=digest+'_labels.png'
            (stage/image_name).write_bytes(raw)
            Image.fromarray(labels).save(stage/label_name)
            split=groups[next(iter(source_keys(r)))];analysis=compare_annotations(r)
            frames.append({'id':digest,'image':image_name,'image_sha256':digest,'labels':label_name,'roi':[0,0,w,h],
                           'labels_sha256':hashlib.sha256((stage/label_name).read_bytes()).hexdigest(),
                           'split':split,'source_keys':sorted(source_keys(r)),'review_sha256':item['review_sha256'],
                           'positive_pixels':int((labels==1).sum()),'negative_pixels':int((labels==0).sum())})
            reports.append({'source_sha256':digest,'split':split,**analysis})
        manifest={'schema_version':1,'ignore_value':255,'annotation_origin':'completed_human_reviews',
                  'frames':frames,'instance_annotations':[{'source_sha256':i['record']['source_sha256'],'objects':i['record']['objects']} for i in items]}
        summary={'images':len(frames),'train_images':sum(f['split']=='train' for f in frames),
                 'validation_images':sum(f['split']=='validation' for f in frames),'quarantined_images':sum(f['split']=='quarantine' for f in frames),
                 'pending_images':sum(f['split']=='pending' for f in frames),'reviewed_objects':sum(r['reviewed_objects'] for r in reports),
                 'added_objects':sum(r['added_objects'] for r in reports),'deleted_objects':sum(r['deleted_objects'] for r in reports),
                 'changed_outlines':sum(r['changed_outlines'] for r in reports),'details':reports,
                 'training_performed':False,'model_applied':False,'split_policy':'same video or date never crosses train/validation; conflicting historical groups quarantined'}
        atomic(stage/'dataset.json',manifest);atomic(stage/'report.json',summary);stage.rename(target)
    finally:
        if stage.exists():shutil.rmtree(stage)
    atomic(folder/'latest_dataset.json',{'path':str(target),'summary':{k:v for k,v in summary.items() if k!='details'}})
    return target,summary


def sampled_features(image,labels,rng,per_class=2000):
    from silkworm_texture import features
    coords=[];classes=[]
    for cls in [0,1]:
        available=np.argwhere(labels==cls)
        if not len(available):continue
        chosen=available[rng.choice(len(available),min(per_class,len(available)),replace=False)]
        coords.extend(chosen);classes.extend([cls]*len(chosen))
    if not coords:return np.empty((0,36),np.float32),np.empty((0,1),np.float32)
    coords=np.asarray(coords);result=np.empty((len(coords),36),np.float32)
    tiles=coords//512
    for tile in np.unique(tiles,axis=0):
        selected=np.flatnonzero(np.all(tiles==tile,axis=1));top,left=tile*512
        y0,x0=max(0,top-136),max(0,left-136)
        y1,x1=min(image.shape[0],top+512+136),min(image.shape[1],left+512+136)
        f=features(image[y0:y1,x0:x1]);points=coords[selected]-[y0,x0]
        result[selected]=f[points[:,0],points[:,1]]
    return result,np.asarray(classes,np.float32)[:,None]


def load_frame(folder,frame):
    paths=[folder/frame[k] for k in ['image','labels']]
    if any(not p.resolve().is_relative_to(folder.resolve()) for p in paths):raise ValueError('学習データのパスが不正です。')
    if hashlib.sha256(paths[0].read_bytes()).hexdigest()!=frame['image_sha256'] or hashlib.sha256(paths[1].read_bytes()).hexdigest()!=frame['labels_sha256']:raise ValueError('学習データが変更されています。')
    image=np.asarray(Image.open(paths[0]).convert('RGB'));labels=np.asarray(Image.open(paths[1]))
    if labels.shape!=image.shape[:2] or not np.isin(labels,[0,1,255]).all():raise ValueError('学習用ラベルが不正です。')
    return image,labels


def pixel_metrics(model,folder,frames):
    from silkworm_texture import predict_texture
    tp=fp=fn=tn=0
    for frame in frames:
        image,labels=load_frame(folder,frame);p=predict_texture(model,image,tile_size=512)>=.5
        tp+=int((p&(labels==1)).sum());fn+=int((~p&(labels==1)).sum())
        fp+=int((p&(labels==0)).sum());tn+=int((~p&(labels==0)).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'foreground_recall':tp/(tp+fn) if tp+fn else None,
            'background_specificity':tn/(tn+fp) if tn+fp else None,
            'scope':'explicitly labelled pixels only, threshold0.5; not animal counts or length accuracy'}


def train_candidate(dataset,base_model,steps=300):
    """Warm-start a background-veto experiment, always leaving live weights alone."""
    import torch
    from silkworm_scratch import load_model
    if not 1<=steps<=5000:raise ValueError('学習回数は1〜5000です。')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    folder=Path(dataset).resolve();config=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    training=[f for f in config['frames'] if f['split']=='train']
    validation=[f for f in config['frames'] if f['split']=='validation']
    if not training or not validation:raise ValueError('別の動画・撮影日の完了画像が2組以上必要です。出所不明の画像は検証用に使いません。')
    rng=np.random.default_rng(42);banks=[np.empty((0,36),np.float32) for _ in range(2)];priorities=[np.empty(0) for _ in range(2)]
    for frame in training:
        image,labels=load_frame(folder,frame);x,y=sampled_features(image,labels,rng)
        for cls in [0,1]:
            new=x[y[:,0]==cls]
            if not len(new):continue
            bank=np.concatenate([banks[cls],new]);keys=np.r_[priorities[cls],rng.random(len(new))]
            keep=np.argpartition(keys,-50000)[-50000:] if len(keys)>50000 else np.arange(len(keys))
            banks[cls]=bank[keep];priorities[cls]=keys[keep]
    if not sum(map(len,banks)):raise ValueError('学習に使える修正がありません。')
    x=np.concatenate(banks);y=np.concatenate([np.full((len(bank),1),cls,np.float32) for cls,bank in enumerate(banks)])
    if set(np.unique(y))!={0.,1.}:raise ValueError('蚕の輪郭と、削除した誤検出の両方が学習用に必要です。')
    base_bytes=Path(base_model).read_bytes();base_sha=hashlib.sha256(base_bytes).hexdigest()
    model,metadata=load_model(base_model)
    if metadata.get('architecture')!='kaiko_texture_mlp_v1':raise ValueError('背景除外モデルの種類が違います。')
    before=pixel_metrics(model,folder,validation)
    if not before['tp']+before['fn'] or not before['tn']+before['fp']:raise ValueError('検証用にも蚕の輪郭と、削除した誤検出の両方が必要です。')
    original=copy.deepcopy(model.state_dict());model.train()
    tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    torch.manual_seed(42);optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    pools=[torch.where(ty[:,0]==cls)[0] for cls in [0,1]]
    for _ in range(steps):
        ids=torch.cat([pool[torch.randint(len(pool),(256,))] for pool in pools])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx[ids]),ty[ids])
        anchor=sum((parameter-original[name]).square().mean() for name,parameter in model.named_parameters())
        loss=loss+.01*anchor
        if not torch.isfinite(loss):raise ValueError('学習の値が不正になりました。')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    model.eval();after=pixel_metrics(model,folder,validation)
    no_regression=after['foreground_recall']>=before['foreground_recall'] and after['background_specificity']>=before['background_specificity']
    improved=no_regression and (after['foreground_recall']>before['foreground_recall'] or after['background_specificity']>before['background_specificity'])
    run=folder/'experiments'/uuid.uuid4().hex[:12];run.mkdir(parents=True)
    candidate_metadata={**metadata,'model_version':'review_candidate','purpose':'semantic_region_preview_only_no_counts_no_lengths',
                        'base_checkpoint_sha256':base_sha,'review_dataset_sha256':hashlib.sha256((folder/'dataset.json').read_bytes()).hexdigest(),
                        'training_steps':steps,'review_training_sources':[f['image_sha256'] for f in training],
                        'review_validation_sources':[f['image_sha256'] for f in validation]}
    torch.save({'metadata':candidate_metadata,'state_dict':model.state_dict()},run/'candidate.pt')
    report={'before':before,'candidate':after,'steps':steps,'pixel_improvement_without_regression':improved,
            'model_applied':False,'automatic_promotion_allowed':False,'active_model_sha256':base_sha,
            'validation_scope':'held out from this review update only; base model exposure not audited',
            'base_training_overlap_checked':False,
            'candidate_sha256':hashlib.sha256((run/'candidate.pt').read_bytes()).hexdigest(),
            'note':'Background veto experiment only. Detector/count evaluation and additional held-out checks are required before publication.'}
    atomic(run/'evaluation.json',report)
    if hashlib.sha256(Path(base_model).read_bytes()).hexdigest()!=base_sha:raise ValueError('本番モデルが変更されました。')
    return run,report
