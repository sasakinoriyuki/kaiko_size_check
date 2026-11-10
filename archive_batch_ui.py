"""One window runs collection, analysis, exports and a resumable review queue."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

from archive_batch import ArchiveBatch,CommandRunner,Cancelled,atomic_json


class BatchWindow:
    def __init__(self,root,storage_root):
        self.root=root;self.storage=Path(storage_root).resolve();self.output=self.storage/'data/archive_batch'
        self.config=self.output/'config.json';self.events=queue.SimpleQueue();self.cancelled=threading.Event()
        self.busy=False;self.closing=False
        self.timer=None
        self.poll_timer=None;self.startup_timer=None
        saved=json.loads(self.config.read_text(encoding='utf-8')) if self.config.exists() else {}
        root.title('蚕CHANNEL アーカイブの自動収集・解析');root.geometry('860x560')
        panel=ttk.Frame(root,padding=16);panel.pack(fill='both',expand=True)
        ttk.Label(panel,text='チャンネルの全ライブから画像を取得 → 解析 → 検証ZIP・結果・改善対象を保存します。',wraplength=810).pack(anchor='w')
        ttk.Label(panel,text='チャンネル／動画URL（空欄なら、以前の動画から自動で調べます）').pack(anchor='w',pady=(12,2))
        self.target=tk.StringVar(value=saved.get('target',''));ttk.Entry(panel,textvariable=self.target).pack(fill='x')
        options=ttk.Frame(panel);options.pack(fill='x',pady=12)
        ttk.Label(options,text='1本あたり').pack(side='left');self.count=tk.StringVar(value=str(saved.get('count',5)))
        ttk.Spinbox(options,from_=1,to=20,textvariable=self.count,width=5).pack(side='left',padx=5)
        ttk.Label(options,text='枚　保存上限').pack(side='left');self.capacity=tk.StringVar(value=str(saved.get('capacity_gb',10)))
        ttk.Spinbox(options,from_=1,to=500,textvariable=self.capacity,width=5).pack(side='left',padx=5)
        ttk.Label(options,text='GB').pack(side='left')
        self.repeat=tk.BooleanVar(value=saved.get('repeat',False))
        ttk.Checkbutton(panel,text='この画面を開いている間、6時間ごとに新しいアーカイブも調べる',variable=self.repeat,command=self.save_preferences).pack(anchor='w')
        self.auto_start=tk.BooleanVar(value=saved.get('auto_start',True))
        ttk.Checkbutton(panel,text='次回の蚕チェック.bat起動時も自動処理を再開する',variable=self.auto_start,command=self.save_preferences).pack(anchor='w')
        buttons=ttk.Frame(panel);buttons.pack(fill='x',pady=12)
        self.start_button=ttk.Button(buttons,text='全アーカイブを自動処理／続きから再開',command=self.start);self.start_button.pack(side='left')
        ttk.Button(buttons,text='停止',command=self.stop).pack(side='left',padx=8)
        ttk.Button(buttons,text='結果の保存先を開く',command=self.open_output).pack(side='left')
        self.pack_button=ttk.Button(buttons,text='改善用ZIPを作る',command=self.make_review_exports)
        self.pack_button.pack(side='left',padx=8)
        self.status=tk.StringVar(value='開始を押すと、画像選択・解析・保存を順番に実行します。途中停止後も再開できます。')
        ttk.Label(panel,textvariable=self.status,wraplength=810).pack(anchor='w',pady=5)
        self.log=tk.Text(panel,height=12,wrap='word',state='disabled');self.log.pack(fill='both',expand=True)
        ttk.Label(panel,text='結果は '+str(self.output)+'\n自動判定は未確認の候補です。検証ZIPを改善に使えます。学習済みモデルを勝手に置き換える処理はまだ行いません。',wraplength=810).pack(anchor='w',pady=10)
        root.protocol('WM_DELETE_WINDOW',self.close);self.poll_timer=root.after(100,self.poll)
        if saved.get('auto_start'):self.startup_timer=root.after(400,self.start)

    def start(self):
        if self.busy:return
        if self.closing:return
        if self.startup_timer:self.root.after_cancel(self.startup_timer);self.startup_timer=None
        if self.timer:self.root.after_cancel(self.timer);self.timer=None
        try:
            count=int(self.count.get());capacity=float(self.capacity.get())
            if not 1<=count<=20 or not 1<=capacity<=500:raise ValueError()
        except ValueError:self.status.set('枚数は1〜20、保存上限は1〜500GBです。');return
        config={'target':self.target.get().strip(),'count':count,'capacity_gb':capacity,'repeat':self.repeat.get(),'auto_start':self.auto_start.get()}
        atomic_json(self.config,config)
        self.cancelled.clear();self.busy=True;self.start_button.configure(state='disabled')
        self.events.put(('log','準備を開始しました。取得・解析には時間がかかります。'))
        def work():
            try:
                from archive_tools import prepare_tools
                from instance_detector import LocalInstanceSegmenter
                from instance_lock import FileLock
                runner=CommandRunner(self.cancelled)
                log=lambda message:self.events.put(('log',message))
                lock=FileLock(self.output/'batch.lock')
                if not lock.acquire():raise RuntimeError('別の自動収集画面ですでに処理中です。')
                program_lock=FileLock(Path(__file__).resolve().parent/'data/.batch.lock')
                try:
                    if not program_lock.acquire():raise RuntimeError('別の自動収集画面ですでに処理中です。')
                    yt,ffmpeg=prepare_tools(self.storage,runner,log)
                    engine=LocalInstanceSegmenter()
                    batch=ArchiveBatch(self.output,engine,runner=runner,cancelled=self.cancelled,log=log,
                                       max_bytes=int(capacity*1024**3),yt_command=yt,ffmpeg_command=ffmpeg)
                    result=batch.run(config['target'],count)
                finally:program_lock.release();lock.release()
                self.events.put(('done',result))
            except Cancelled:self.events.put(('error','停止しました。次回は続きから処理できます。'))
            except Exception as exc:
                atomic_json(self.output/'last_error.json',{'status':'failed','error':str(exc)})
                self.events.put(('error',str(exc)))
        self.thread=threading.Thread(target=work,daemon=False);self.thread.start()

    def make_review_exports(self):
        from archive_review import launch_review_export
        try:launch_review_export(self.output)
        except OSError as exc:self.status.set('改善用ZIPの作成を起動できません：'+str(exc))

    def save_preferences(self):
        if not self.config.exists():return
        saved=json.loads(self.config.read_text(encoding='utf-8'))
        saved.update({'auto_start':self.auto_start.get(),'repeat':self.repeat.get()})
        atomic_json(self.config,saved)

    def stop(self):
        self.repeat.set(False);self.cancelled.set()
        self.save_preferences()
        if self.timer:self.root.after_cancel(self.timer);self.timer=None
        if self.startup_timer:self.root.after_cancel(self.startup_timer);self.startup_timer=None
        self.status.set('処理を停止しています。保存済みの結果は残します。')

    def poll(self):
        self.poll_timer=None
        while not self.events.empty():
            kind,value=self.events.get()
            if kind=='log':
                self.status.set(value);self.log.configure(state='normal');self.log.insert('end',value+'\n');self.log.see('end');self.log.configure(state='disabled')
            else:
                self.busy=False;self.start_button.configure(state='normal')
                if kind=='error':self.status.set('処理を保留しました：'+value)
                else:
                    self.status.set(f'保存 {value["saved_images"]} 枚／処理済みを除外 {value["skipped_completed"]} 本（履歴のみ {value.get("history_only_videos",0)} 本）／元画像再利用 {value.get("reused_images",0)} 枚／エラー {len(value["errors"])} 件。')
                    if self.repeat.get() and not self.closing:
                        self.timer=self.root.after(6*60*60*1000,lambda:self.start() if self.repeat.get() and not self.closing else None)
        if self.closing and not self.busy:self.root.destroy();return
        self.poll_timer=self.root.after(100,self.poll)

    def open_output(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.open_folder(self.output)

    def open_folder(self,folder):
        try:
            if os.name=='nt':os.startfile(str(folder))
            else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(folder)])
        except OSError:self.status.set('フォルダを開けませんでした。保存先：'+str(folder))

    def close(self):
        self.closing=True;self.stop()
        if not self.busy:
            if self.poll_timer:self.root.after_cancel(self.poll_timer);self.poll_timer=None
            self.root.destroy()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--storage-root',required=True);args=parser.parse_args()
    root=tk.Tk();BatchWindow(root,args.storage_root);root.mainloop()
