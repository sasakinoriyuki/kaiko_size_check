"""Select varied, unverified examples and make bounded review ZIPs locally."""
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import queue
import unicodedata
import uuid
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED, BadZipFile

ID=re.compile(r'^[a-f0-9]{24}$')


@dataclass
class Example:
    row: dict
    path: Path
    instar: int | None
    title: str

    @property
    def video(self):return self.row['video_id']
    @property
    def sid(self):return self.row['sample_id']
    @property
    def source_hash(self):return self.row['source_sha256']
    @property
    def count(self):return int(self.row['estimated_count'])
    @property
    def full(self):return int(self.row['full_length_count'])
    @property
    def uncertain(self):return int(self.row['uncertain_regions'])
    @property
    def priority(self):
        # Zero predictions require inspection too: they may be empty scenes.
        return (100 if self.count==0 else 90 if self.full==0 else
                40+30*(self.count-self.full)/max(1,self.count)) + min(20,self.uncertain)


def infer_instar(title):
    title=unicodedata.normalize('NFKC',str(title))
    found=re.search(r'([1-5一二三四五])\s*[齢齡]',title)
    if found:
        return int(found[1]) if found[1].isdigit() else '一二三四五'.index(found[1])+1
    if '孵化直後' in title or '孵化当日' in title:return 1
    return None


def _number(row,key):
    value=float(row[key])
    if not math.isfinite(value) or value<0 or value!=int(value):raise ValueError('結果CSVの数値が不正です：'+key)
    return int(value)


def csv_snapshot(path):
    raw=Path(path).read_bytes()
    # The producer appends complete newline-terminated rows after its source
    # ZIP is closed. Ignore an in-progress final row while collecting.
    end=raw.rfind(b'\n')
    return raw[:end+1] if end>=0 else b''


def read_examples(folder,cancelled=None):
    folder=Path(folder).resolve();source=folder/'results.csv'
    if not source.is_file():raise ValueError('この保存先には解析結果と元画像がありません。画像を取得したPCで作成するか、そのPCのdata/archive_batchフォルダをコピーしてください。')
    rows=list(csv.DictReader(io.StringIO(csv_snapshot(source).decode('utf-8-sig'))))
    examples=[];skipped=[];seen={}
    for row in rows:
        if cancelled is not None and cancelled.is_set():raise InterruptedError('改善用ZIPの作成を停止しました。')
        sid=row.get('sample_id','')
        if not ID.fullmatch(sid) or row.get('sample_zip')!='samples/'+sid+'.zip':
            raise ValueError('結果CSVの検証ZIPパスが不正です。')
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}',row.get('video_id','')) or not re.fullmatch(r'[a-f0-9]{64}',row.get('source_sha256','')):
            raise ValueError('結果CSVの動画IDまたは画像ハッシュが不正です。')
        if sid in seen:
            if row!=seen[sid]:raise ValueError('同じ画像番号の記録が一致しません。')
            continue
        seen[sid]=row
        count,full,partial,uncertain=[_number(row,k) for k in ('estimated_count','full_length_count','visible_segment_count','uncertain_regions')]
        if full>count or partial>count-full:raise ValueError('結果CSVの頭数と計測数が一致しません。')
        path=folder/'samples'/(sid+'.zip')
        if not path.is_file() or path.is_symlink() or path.resolve().parent!=(folder/'samples').resolve() or (folder/'samples').is_symlink():
            skipped.append({'sample_id':sid,'reason':'missing_or_linked_source_zip'});continue
        try:
            with ZipFile(path) as archive:
                all_members=archive.infolist()
                images=[m for m in all_members if m.filename.lower().endswith('.png')]
                members=[m for m in all_members if m.filename.lower().endswith('.json')]
                if len(all_members)!=2 or len(images)!=1 or sum(m.file_size for m in all_members)>64*1024**2:
                    raise ValueError('source members')
                if len(members)!=1 or members[0].file_size>8*1024**2:raise ValueError('metadata')
                info=json.loads(archive.read(members[0]))
                if not isinstance(info,dict):raise ValueError('metadata')
                if info.get('sample_id') and info['sample_id']!=sid:raise ValueError('sample id')
                if info.get('source_sha256') and info['source_sha256']!=row['source_sha256']:raise ValueError('source hash')
                collection=info.get('collection',{})
                title=str(collection.get('video_title','') if isinstance(collection,dict) else '')
                examples.append(Example(row,path,infer_instar(title),title[:1000]))
        except (OSError,ValueError,KeyError,RuntimeError,BadZipFile):
            skipped.append({'sample_id':sid,'reason':'unreadable_or_mismatched_metadata'})
    return examples,skipped


def select_examples(examples,limit=100):
    if not 1<=limit<=1000:raise ValueError('選別枚数は1〜1000です。')
    chosen=[];used_videos=set();used_hashes=set()
    def available(e,distinct_video=True):return (not distinct_video or e.video not in used_videos) and e.source_hash not in used_hashes
    def add(e,distinct_video=True):
        if len(chosen)<limit and available(e,distinct_video):
            chosen.append(e);used_videos.add(e.video);used_hashes.add(e.source_hash)
    # One difficult and one measurable comparison from each known stage.
    # Titles are only stage hints; none of these masks are ground-truth labels.
    for stage in [1,2,3,4,5,None]:
        pool=[e for e in examples if e.instar==stage and available(e)]
        if not pool:continue
        add(max(pool,key=lambda e:(e.priority,e.sid)))
        controls=[e for e in pool if available(e) and e.full>0]
        if controls:add(min(controls,key=lambda e:(e.uncertain/max(1,e.count),-e.full,e.sid)))
    while len(chosen)<limit:
        pool=[e for e in examples if available(e)]
        distinct=True
        if not pool:pool=[e for e in examples if available(e,False)];distinct=False
        if not pool:break
        populations={s:sum(e.instar==s for e in chosen) for s in [1,2,3,4,5,None]}
        add(min(pool,key=lambda e:(populations[e.instar],-e.priority,e.sid)),distinct)
    return chosen


def make_review_exports(folder,*,limit=100,part_limit=24*1024**2,log=lambda message:None,cancelled=None):
    """Copies selected source ZIPs; never edits results or runs inference again."""
    folder=Path(folder).resolve()
    if part_limit<4096:raise ValueError('ZIPの容量上限が小さすぎます。')
    log('記録と検証ZIPを確認し、齢・失敗の種類・動画の重複を見て選別しています。')
    examples,skipped=read_examples(folder,cancelled)
    extras={}
    for name in ['results.csv','review_queue.json']:
        path=folder/name
        if path.is_file():extras[name]=csv_snapshot(path) if name=='results.csv' else path.read_bytes()
    overhead=sum(len(raw) for raw in extras.values())+65536
    if overhead>=part_limit:raise ValueError('結果記録が大きすぎるため、分割容量内にまとめられません。')
    eligible=[]
    for e in examples:
        if e.path.stat().st_size+overhead+512<=part_limit:eligible.append(e)
        else:skipped.append({'sample_id':e.sid,'reason':'source_zip_exceeds_part_limit'})
    chosen=select_examples(eligible,limit)
    if not chosen:raise ValueError('送信用に使える検証ZIPがありません。results.csvと同じ保存先のsamplesフォルダを確認してください。')
    records=[]
    for e in chosen:
        if cancelled is not None and cancelled.is_set():raise InterruptedError('改善用ZIPの作成を停止しました。')
        with ZipFile(e.path) as archive:
            image=next(m for m in archive.infolist() if m.filename.lower().endswith('.png'))
            if hashlib.sha256(archive.read(image)).hexdigest()!=e.source_hash:
                raise ValueError('元画像のハッシュが一致しません：'+e.sid)
        records.append({'sample_id':e.sid,'video_id':e.video,'inferred_instar':e.instar,
                        'video_title':e.title,'source_sha256':e.source_hash,
                        'source_zip_sha256':hashlib.sha256(e.path.read_bytes()).hexdigest(),
                        'purpose':'inspect_zero_detections' if e.count==0 else 'inspect_held_full_lengths' if e.full==0 else 'unverified_measurable_comparison',
                        'automatic_result':e.row})
    base={'schema':1,'selected_count':len(chosen),'available_count':len(examples),
          'status':'unverified_predictions_not_training_truth',
          'selection_policy':'stage hints from titles; difficult and measurable examples; distinct videos and source hashes',
          'skipped':skipped[:1000],'selected':records,
          'note':'No model is retrained or replaced. Full-length estimates and visible segments are distinct. Missing scales stay px.'}
    metadata_bound=len(json.dumps({**base,'part':1000,'total_parts':1000,'part_sample_ids':[e.sid for e in chosen]},ensure_ascii=False,indent=2).encode())+4096
    overhead=sum(len(raw) for raw in extras.values())+max(65536,metadata_bound)
    if overhead>=part_limit:raise ValueError('選別情報が分割容量内に収まりません。')
    groups=[];group=[];size=overhead
    for e in chosen:
        needed=e.path.stat().st_size+512
        if group and size+needed>part_limit:groups.append(group);group=[];size=overhead
        group.append(e);size+=needed
    if group:groups.append(group)
    required=sum(e.path.stat().st_size for e in chosen)+overhead*len(groups)
    if shutil.disk_usage(folder).free<required+64*1024**2:raise OSError('改善用ZIPを作る空き容量が不足しています。')
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    output=folder/'review_exports'/stamp;output.mkdir(parents=True)
    outputs=[]
    try:
        for index,group in enumerate(groups,1):
            target=output/f'kaiko_review_{stamp}_part{index:02}.zip'
            temporary=target.with_suffix('.tmp')
            manifest={**base,'part':index,'total_parts':len(groups),'part_sample_ids':[e.sid for e in group]}
            with ZipFile(temporary,'w',ZIP_STORED) as archive:
                archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2),compress_type=ZIP_DEFLATED)
                for name,raw in extras.items():archive.writestr(name,raw,compress_type=ZIP_DEFLATED)
                for e in group:
                    if cancelled is not None and cancelled.is_set():raise InterruptedError('改善用ZIPの作成を停止しました。')
                    raw=e.path.read_bytes()
                    recorded=next(r['source_zip_sha256'] for r in records if r['sample_id']==e.sid)
                    if hashlib.sha256(raw).hexdigest()!=recorded:raise ValueError('検証ZIPが作成中に変更されました。もう一度作成してください。')
                    archive.writestr('samples/'+e.path.name,raw)
            if temporary.stat().st_size>part_limit:raise ValueError('改善用ZIPの分割容量を超えました。')
            temporary.replace(target);outputs.append(target)
        log(f'改善用に {len(chosen)} 枚を選び、ZIP {len(outputs)} 個へまとめました。')
        return outputs
    except BaseException:
        shutil.rmtree(output,ignore_errors=True)
        raise


def launch_review_export(folder):
    return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--folder',str(Path(folder).resolve())],
                            cwd=str(Path(__file__).resolve().parent),
                            **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))


def open_result_folder(folder):
    if os.name=='nt':os.startfile(str(folder))
    else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(folder)])


class ReviewWindow:
    def __init__(self,root,folder):
        import tkinter as tk
        from tkinter import ttk
        self.root=root;self.folder=Path(folder);self.events=queue.SimpleQueue()
        self.cancelled=threading.Event();self.closing=False;self.done=False;self.paths=[]
        root.title('蚕CHANNEL 改善用ZIPを作る');root.geometry('680x230')
        panel=ttk.Frame(root,padding=18);panel.pack(fill='both',expand=True)
        ttk.Label(panel,text='保存済みの画像から、改善の確認に使う画像を選んでZIPにまとめます。',wraplength=630).pack(anchor='w')
        self.status=tk.StringVar(value='選別を開始しています。');ttk.Label(panel,textvariable=self.status,wraplength=630).pack(anchor='w',pady=15)
        self.open_button=ttk.Button(panel,text='作ったZIPの場所を開く',command=self.open_output,state='disabled');self.open_button.pack(side='left')
        ttk.Button(panel,text='閉じる／停止',command=self.close).pack(side='right')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.timer=root.after(100,self.poll)
        def work():
            try:
                from instance_lock import FileLock
                with FileLock(self.folder/'review_exports.lock'):
                    paths=make_review_exports(self.folder,log=lambda message:self.events.put(('log',message)),cancelled=self.cancelled)
                self.events.put(('done',paths))
            except Exception as exc:self.events.put(('error',str(exc)))
        self.thread=threading.Thread(target=work,daemon=False);self.thread.start()

    def poll(self):
        self.timer=None
        while not self.events.empty():
            kind,value=self.events.get()
            if kind=='log':self.status.set(value)
            else:
                self.done=True
                if kind=='error':self.status.set('作成できませんでした：'+value)
                else:
                    self.paths=value;self.open_button.configure(state='normal')
                    self.status.set(f'ZIPを {len(value)} 個作成しました。開いたフォルダのZIPをこのチャットへ送ってください。')
                    if not self.closing:self.open_output()
        if self.closing and self.done:self.root.destroy();return
        self.timer=self.root.after(100,self.poll)

    def open_output(self):
        if self.paths:
            try:open_result_folder(self.paths[0].parent)
            except OSError:self.status.set('ZIPの保存先：'+str(self.paths[0].parent))

    def close(self):
        self.closing=True;self.cancelled.set()
        if self.done:
            if self.timer:self.root.after_cancel(self.timer);self.timer=None
            self.root.destroy()
        else:self.status.set('停止しています。')


if __name__=='__main__':
    import argparse
    import tkinter as tk
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);args=parser.parse_args()
    root=tk.Tk();ReviewWindow(root,args.folder);root.mainloop()
