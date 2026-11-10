"""Resume public archive collection and use the desktop's exact analysis engine.

Images, reports and review exports stay under the selected portable data folder.
Automatic predictions are evidence for review, not verified training labels.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np

from collect_youtube_samples import find_tool, youtube_samples, _run
from measurement import Settings
from archive_history import completion_key,load_completion_keys

SEED_VIDEOS = ['0rijis5y8c0', 'RzZ0gH6DMF8', 'pbcVIyIBjig']
VIDEO = re.compile(r'^[A-Za-z0-9_-]{11}$')
CHANNEL = re.compile(r'^UC[A-Za-z0-9_-]{22}$')
CSV_FIELDS = ['sample_id', 'video_id', 'requested_video_seconds', 'unit',
              'estimated_count', 'full_length_count', 'full_length_mean',
              'visible_segment_count', 'uncertain_regions', 'review_reasons',
              'sample_zip', 'source_sha256', 'model_version', 'model_sha256']


def sample_fraction(index,count):
    if not 1<=index<=count<=20:raise ValueError('画像の抽出位置が不正です。')
    return Fraction(1,2) if count==1 else Fraction(1,10)+Fraction(4*(index-1),5*(count-1))


class Cancelled(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def youtube_target(value):
    """Only public YouTube inputs; external text never becomes a local path."""
    value=value.strip()
    if CHANNEL.fullmatch(value):return 'https://www.youtube.com/channel/'+value+'/streams'
    if VIDEO.fullmatch(value):return 'https://www.youtube.com/watch?v='+value
    p=urlparse(value)
    if p.scheme not in {'https','http'} or p.hostname not in {'youtube.com','www.youtube.com','m.youtube.com','youtu.be'} or p.username or p.password or p.port not in {None,80,443}:
        raise ValueError('YouTubeのチャンネルまたは動画URLを入力してください。')
    if p.hostname=='youtu.be':
        video=p.path.strip('/')
        if not VIDEO.fullmatch(video):raise ValueError('動画IDが不正です。')
        return 'https://www.youtube.com/watch?v='+video
    if p.path=='/watch':
        video=parse_qs(p.query).get('v',[''])[0]
        if not VIDEO.fullmatch(video):raise ValueError('動画IDが不正です。')
        return 'https://www.youtube.com/watch?v='+video
    parts=p.path.strip('/').split('/')
    if len(parts)>=2 and parts[0]=='channel' and CHANNEL.fullmatch(parts[1]):
        return 'https://www.youtube.com/channel/'+parts[1]+'/streams'
    if parts and re.fullmatch(r'@[^/\s?#]{1,100}',parts[0]):
        return 'https://www.youtube.com/'+parts[0]+'/streams'
    raise ValueError('チャンネルURLまたは通常の動画URLを指定してください。')


class CommandRunner:
    """Cancellation ends only processes started by this batch, including ffmpeg."""
    def __init__(self,cancelled=None):self.cancelled=cancelled or threading.Event()

    def __call__(self,command,timeout=None,check=False,**options):
        if self.cancelled.is_set():raise Cancelled('中止しました。')
        options.pop('capture_output',None)
        if os.name=='nt':options['creationflags']=options.get('creationflags',0)|subprocess.CREATE_NEW_PROCESS_GROUP
        else:options['start_new_session']=True
        process=subprocess.Popen(command,**options)
        started=time.monotonic()
        try:
            while True:
                if self.cancelled.is_set():raise Cancelled('中止しました。')
                if timeout and time.monotonic()-started>timeout:raise subprocess.TimeoutExpired(command,timeout)
                try:
                    output,error=process.communicate(timeout=.2)
                    result=subprocess.CompletedProcess(command,process.returncode,output,error)
                    if check:result.check_returncode()
                    return result
                except subprocess.TimeoutExpired:pass
        except BaseException:
            if process.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                else:
                    import signal
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            raise
        finally:
            for pipe in (process.stdout,process.stderr,process.stdin):
                if pipe is not None:pipe.close()


def list_archives(target='',*,runner=subprocess.run,yt_command=None,remembered_channel=None):
    yt=yt_command or find_tool('yt-dlp',runner=runner)
    common=[*yt,'--ignore-config','--no-cache-dir','--no-warnings','--socket-timeout','20','--retries','1','--sleep-requests','1']
    if not target and remembered_channel:target=remembered_channel
    targets=[youtube_target(target)] if target else [youtube_target(v) for v in SEED_VIDEOS]
    last_error=None
    for source in targets:
        try:
            if '/watch?' in source:
                info=json.loads(_run([*common,'--no-playlist','--skip-download','--dump-single-json',source],runner,180))
                channel=info.get('channel_id','')
                if not CHANNEL.fullmatch(channel):raise ValueError('動画からチャンネルIDを確認できませんでした。')
                source=youtube_target(channel)
            listing=json.loads(_run([*common,'--flat-playlist','--dump-single-json',source],runner,300))
            channel=listing.get('channel_id') or listing.get('id','')
            if not CHANNEL.fullmatch(channel):
                match=re.search(r'/channel/(UC[A-Za-z0-9_-]{22})/',source)
                channel=match[1] if match else None
            result=[];seen=set()
            for entry in listing.get('entries',[]):
                if not isinstance(entry,dict):continue
                vid=str(entry.get('id',''))
                if not VIDEO.fullmatch(vid) or vid in seen:continue
                seen.add(vid)
                if entry.get('is_live') or entry.get('live_status') in {'is_live','is_upcoming','post_live'}:continue
                result.append({'id':vid,'title':str(entry.get('title',''))[:500],
                               'url':'https://www.youtube.com/watch?v='+vid})
            if not result:raise ValueError('完了したライブのアーカイブが見つかりませんでした。')
            return channel,result
        except (RuntimeError,ValueError,OSError,json.JSONDecodeError) as exc:last_error=exc
    raise RuntimeError('アーカイブ一覧を取得できませんでした。'+str(last_error))


def review_reasons(candidates,info):
    reasons=[]
    if not info.get('estimated_count'):reasons.append('no_accepted_instance')
    if not info.get('measurable_count'):reasons.append('no_full_length')
    if any(c.counted and c.length_px is None for c in candidates):reasons.append('full_length_held')
    if info.get('uncertain_regions',0):reasons.append('uncertain_detections')
    return reasons


def append_row(path,row):
    path=Path(path);header=not path.exists() or path.stat().st_size==0
    if not header:
        with path.open(encoding='utf-8-sig',newline='') as f:
            if next(csv.reader(f),None)!=CSV_FIELDS:raise ValueError('一括結果CSVの列が異なります。既存データは変更しません。')
    with path.open('a',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        if header:writer.writeheader()
        writer.writerow(row)


def read_sample_zip(path):
    """Decode one source image without extracting paths or applying predictions."""
    from PIL import Image
    with ZipFile(path) as archive:
        members=archive.infolist()
        if len(members)>20 or sum(m.file_size for m in members)>64*1024**2:
            raise ValueError('検証ZIPが大きすぎます。')
        if any(m.filename.startswith(('/', '\\')) or '..' in Path(m.filename).parts or ':' in m.filename for m in members):
            raise ValueError('ZIP内のパスが不正です。')
        images=[m for m in members if m.filename.lower().endswith('.png')]
        metadata=[m for m in members if m.filename.lower().endswith('.json')]
        if len(images)!=1 or len(metadata)!=1:raise ValueError('元画像1枚とJSON1個の検証ZIPを選んでください。')
        raw=archive.read(images[0]);info=json.loads(archive.read(metadata[0]))
    if not isinstance(info,dict):raise ValueError('検証ZIPのJSONが不正です。')
    with Image.open(io.BytesIO(raw)) as header:
        if header.width*header.height>40_000_000:raise ValueError('元画像の画素数が大きすぎます。')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('ZIP内の画像を読み込めません。')
    if info.get('source_sha256') and hashlib.sha256(raw).hexdigest()!=info['source_sha256']:
        raise ValueError('元画像のハッシュが一致しません。')
    if info.get('source_resolution') and info['source_resolution']!=[image.shape[1],image.shape[0]]:
        raise ValueError('元画像の大きさが記録と一致しません。')
    return image,info


class ArchiveBatch:
    def __init__(self,folder,engine,*,extractor=youtube_samples,runner=subprocess.run,
                 cancelled=None,log=None,max_bytes=10*1024**3,yt_command=None,ffmpeg_command=None,completed_keys=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.engine=engine;self.extractor=extractor;self.runner=runner
        self.cancelled=cancelled or threading.Event();self.log=log or (lambda message:None)
        self.max_bytes=max_bytes;self.yt_command=yt_command;self.ffmpeg_command=ffmpeg_command
        self.state_path=self.folder/'state.json'
        self.state=json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {'schema':1,'channels':{},'videos':{}}
        if self.state.get('schema')!=1:raise ValueError('一括解析の記録形式が未対応です。')
        self.fingerprint=hashlib.sha256((str(engine.metadata.get('model_version'))+str(engine.model_sha256)).encode()).hexdigest()[:16]
        self.completed_keys=(load_completion_keys() if completed_keys is None else set(completed_keys)) | set(self.state.get('completed_collection_keys',[]))
        self.state['completed_collection_keys']=sorted(self.completed_keys)

    def checkpoint(self):atomic_json(self.state_path,self.state)

    def cached_samples(self,video_id,count):
        """Locate source pixels independently of the model which analyzed them."""
        cached={};was_completed=False
        target_slots={sample_fraction(i,count):str(i) for i in range(1,count+1)}
        for key,item in self.state['videos'].items():
            pieces=key.split(':')
            if len(pieces)!=3 or pieces[0]!=video_id:continue
            old_count=int(pieces[1])
            was_completed=was_completed or (old_count==count and item.get('status')=='done')
            for slot,sid in item.get('samples',{}).items():
                if not str(slot).isdigit() or not 1<=int(slot)<=old_count<=20 or not re.fullmatch('[a-f0-9]{24}',sid):
                    raise ValueError('保存済み画像の再開記録が不正です。')
                target=target_slots.get(sample_fraction(int(slot),old_count))
                if target is None:continue
                path=self.folder/'samples'/(sid+'.zip')
                if path.is_symlink():raise ValueError('保存済み画像がリンク先へ変更されています。')
                if path.is_file():cached[target]=sid
        return cached,was_completed

    def source_images(self,entry,count,done,cached):
        """Read cached original PNGs first, then fetch only truly missing slots."""
        skipped={int(slot) for slot in done}
        for slot,sid in sorted(cached.items(),key=lambda item:int(item[0])):
            if int(slot) in skipped:continue
            if self.cancelled.is_set():raise Cancelled('中止しました。')
            image,info=read_sample_zip(self.folder/'samples'/(sid+'.zip'))
            metadata={**info.get('collection',{}),'sample_index':int(slot),'requested_count':count,
                      'video_url':entry['url'],'reused_from_sample_id':sid}
            if 'requested_video_seconds' not in metadata:metadata['requested_video_seconds']=info.get('video_seconds')
            skipped.add(int(slot));yield image,metadata
        if len(skipped)<count:
            yield from self.extractor(entry['url'],count=count,cancelled=self.cancelled,runner=self.runner,
                                      yt_command=self.yt_command,ffmpeg_command=self.ffmpeg_command,skip_indices=skipped)

    def check_space(self):
        total=sum(p.stat().st_size for p in self.folder.rglob('*') if p.is_file())
        if total>=self.max_bytes:raise OSError('保存容量の上限に達しました。容量を増やすか保存先を整理してから再開してください。')
        if shutil.disk_usage(self.folder).free<1024**3:raise OSError('保存先の空き容量が1GB未満です。処理を停止します。')

    def analyze_image(self,image,metadata,video_id,index,count):
        if self.cancelled.is_set():raise Cancelled('中止しました。')
        self.check_space()
        if image is None or image.ndim!=3:raise ValueError('画像を読み込めません。')
        ok,encoded=cv2.imencode('.png',image)
        if not ok:raise ValueError('元画像を保存できません。')
        raw=encoded.tobytes();digest=hashlib.sha256(raw).hexdigest()
        sid=hashlib.sha256(f'{video_id}:{index}:{count}:{self.fingerprint}:{digest}'.encode()).hexdigest()[:24]
        folder=self.folder/'samples';folder.mkdir(exist_ok=True)
        output=folder/(sid+'.zip');receipt=folder/(sid+'.json')
        if output.exists() and receipt.exists():return json.loads(receipt.read_text(encoding='utf-8'))
        self.log(f'{video_id}：画像 {index}/{count} を解析しています。')
        candidates,device=self.engine.analyze(image,Settings())
        info={**self.engine.metadata,**self.engine.last_info,'model_sha256':self.engine.model_sha256,'device':device}
        info['instances']=[{'id':c.instance_id,'counted':c.counted,'confidence':c.confidence,
                           'count_reason':c.count_reason,'length_px':c.length_px,'reason':c.reason,
                           'visible_length_px':c.visible_length_px,'visible_note':c.visible_note,
                           'contour_source_xy':c.contour.reshape(-1,2).tolist(),
                           'centerline_source_xy':None if c.path is None else c.path.tolist(),
                           'visible_centerline_source_xy':None if c.visible_path is None else c.visible_path.tolist()} for c in candidates]
        reasons=review_reasons(candidates,info)
        h,w=image.shape[:2]
        report={'timestamp':now(),'source':metadata.get('video_url',video_id),
                'video_seconds':metadata.get('requested_video_seconds'),
                'source_resolution':[w,h],'crop_xyxy':[0,0,w,h],'pixels_per_mm':None,
                'settings':asdict(Settings()),'candidate_method':'instance','candidate_status':'unverified',
                'model_info':info,'collection':metadata,'source_sha256':digest,
                'review_reasons':reasons,'sample_id':sid,
                'note':'Automatic predictions, not verified labels. Scale is unknown; lengths are px.'}
        temporary=output.with_suffix('.zip.tmp')
        try:
            with ZipFile(temporary,'w',ZIP_DEFLATED) as z:
                z.writestr(sid+'.png',raw)
                z.writestr(sid+'.json',json.dumps(report,ensure_ascii=False,indent=2))
            os.replace(temporary,output)
        finally:temporary.unlink(missing_ok=True)
        overlay=image.copy()
        for c in candidates:
            if not c.counted:continue
            color=(40,250,90) if c.length_px is not None else (20,215,255)
            cv2.drawContours(overlay,[c.contour],-1,color,2)
            p=c.path if c.path is not None else c.visible_path
            if p is not None:cv2.polylines(overlay,[np.rint(p).astype(np.int32)],False,color,2)
            x,y=c.contour.reshape(-1,2).mean(0).astype(int)
            cv2.putText(overlay,str(c.instance_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        # Save a preview separately; it is never used as a training image.
        success,preview=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,88])
        if success:(folder/(sid+'_preview.jpg')).write_bytes(preview.tobytes())
        full=[c.length_px for c in candidates if c.counted and c.length_px is not None]
        row={'sample_id':sid,'video_id':video_id,'requested_video_seconds':metadata.get('requested_video_seconds'),
             'unit':'px','estimated_count':info.get('estimated_count',0),'full_length_count':len(full),
             'full_length_mean':float(np.mean(full)) if full else None,
             'visible_segment_count':info.get('visible_segment_count',0),'uncertain_regions':info.get('uncertain_regions',0),
             'review_reasons':'|'.join(reasons),'sample_zip':output.relative_to(self.folder).as_posix(),
             'source_sha256':digest,'model_version':info.get('model_version'),'model_sha256':self.engine.model_sha256}
        # Recover cleanly after a crash between CSV append and receipt creation.
        csv_path=self.folder/'results.csv';already=False
        if csv_path.exists():
            with csv_path.open(encoding='utf-8-sig',newline='') as f:already=any(r.get('sample_id')==sid for r in csv.DictReader(f))
        if not already:append_row(csv_path,row)
        atomic_json(receipt,row)
        return row

    def run(self,target='',count=5,max_videos=0,entries=None,channel_id=None):
        if not 1<=count<=20:raise ValueError('抽出枚数は1〜20です。')
        if max_videos<0:raise ValueError('動画数は0（全件）以上です。')
        if entries is None:
            channel_id,entries=list_archives(target,runner=self.runner,yt_command=self.yt_command,
                                             remembered_channel=self.state.get('last_channel_id'))
        if channel_id:self.state['last_channel_id']=channel_id
        summary={'started_at':now(),'listed_archives':len(entries),'completed_videos':0,
                 'skipped_completed':0,'saved_images':0,'review_images':0,'errors':[],
                 'history_only_videos':0,'reused_images':0,'downloaded_images':0,
                 'automatic_training':False,'production_model_modified':False,'channel_id':channel_id}
        self.log(f'完了済みライブ {len(entries)} 本を確認しました。処理済みを除き、未取得分だけ各 {count} 枚を処理します。')
        consecutive_failures=0;attempted=0
        try:
            for entry in entries:
                if self.cancelled.is_set():raise Cancelled('中止しました。')
                video_id=entry['id']
                if not VIDEO.fullmatch(video_id):raise ValueError('動画IDが不正です。')
                key=f'{video_id}:{count}:{self.fingerprint}'
                previous=self.state['videos'].get(key,{})
                saved_ids=list(previous.get('samples',{}).values())
                if previous.get('status')=='done' and len(saved_ids)==count:
                    if not all((self.folder/'samples'/(sid+'.zip')).is_file() for sid in saved_ids):summary['history_only_videos']+=1
                    summary['skipped_completed']+=1;continue
                cached,was_completed=self.cached_samples(video_id,count)
                if len(cached)<count and (was_completed or completion_key(video_id,count) in self.completed_keys):
                    summary['skipped_completed']+=1;summary['history_only_videos']+=1
                    continue
                if max_videos and attempted>=max_videos:break
                attempted+=1
                item=self.state['videos'].setdefault(key,{'id':video_id,'samples':{}})
                item['samples']={slot:sid for slot,sid in item['samples'].items() if (self.folder/'samples'/(sid+'.zip')).is_file() and (self.folder/'samples'/(sid+'.json')).is_file()}
                item['status']='running';self.checkpoint()
                try:
                    for fallback,(image,meta) in enumerate(self.source_images(entry,count,item['samples'],cached),1):
                        index=int(meta.get('sample_index',fallback))
                        if not 1<=index<=count:raise ValueError('抽出画像の番号が不正です。')
                        if self.cancelled.is_set():raise Cancelled('中止しました。')
                        slot=str(index)
                        if slot in item['samples']:continue
                        row=self.analyze_image(image,meta,video_id,index,count)
                        item['samples'][slot]=row['sample_id'];summary['saved_images']+=1
                        if meta.get('reused_from_sample_id'):summary['reused_images']+=1
                        else:summary['downloaded_images']+=1
                        summary['review_images']+=bool(row['review_reasons']);self.checkpoint()
                    if self.cancelled.is_set():raise Cancelled('中止しました。')
                    if len(item['samples'])!=count:raise ValueError(f'抽出枚数が不足しています（{len(item["samples"])}/{count}）。次回再試行します。')
                    item['status']='done';item['completed_at']=now();item.pop('error',None)
                    self.completed_keys.add(completion_key(video_id,count));self.state['completed_collection_keys']=sorted(self.completed_keys)
                    summary['completed_videos']+=1;consecutive_failures=0
                except Cancelled:raise
                except (OSError,RuntimeError,ValueError) as exc:
                    item['status']='error';item['error']=str(exc)[-2000:]
                    summary['errors'].append({'video_id':video_id,'error':str(exc)[-2000:]})
                    consecutive_failures+=1;self.log(f'{video_id}：取得または解析を保留しました。{str(exc)[-500:]}')
                self.checkpoint()
                if consecutive_failures>=2:
                    self.log('連続2本で失敗したため停止します。YouTubeの制限または接続状態を確認してください。');break
            summary['status']='completed_with_errors' if summary['errors'] else 'completed'
            self.log(f'処理済み {summary["skipped_completed"]} 本は再取得していません。元画像の再利用 {summary["reused_images"]} 枚、新規取得 {summary["downloaded_images"]} 枚です。')
            if summary['history_only_videos']:
                self.log(f'{summary["history_only_videos"]} 本は完了履歴のみで、この保存先には元画像がそろっていません。元画像を使う場合は、取得したPCのdataフォルダも引き継いでください。')
        except Cancelled:
            summary['status']='cancelled'
        finally:
            summary['finished_at']=now();self.checkpoint();atomic_json(self.folder/'last_run.json',summary)
            queue=[]
            for receipt in (self.folder/'samples').glob('*.json') if (self.folder/'samples').exists() else []:
                row=json.loads(receipt.read_text(encoding='utf-8'))
                if row.get('review_reasons'):queue.append(row)
            atomic_json(self.folder/'review_queue.json',{'schema':1,'status':'unverified_predictions','samples':queue})
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',default='');parser.add_argument('--folder',required=True)
    parser.add_argument('--count',type=int,default=5);parser.add_argument('--max-videos',type=int,default=0)
    args=parser.parse_args()
    from instance_detector import LocalInstanceSegmenter
    from archive_tools import prepare_tools
    event=threading.Event();runner=CommandRunner(event)
    yt,ffmpeg=prepare_tools(Path(args.folder),runner,print)
    engine=LocalInstanceSegmenter()
    result=ArchiveBatch(args.folder,engine,runner=runner,cancelled=event,log=print,yt_command=yt,ffmpeg_command=ffmpeg).run(args.target,args.count,args.max_videos)
    print(json.dumps(result,ensure_ascii=False))
