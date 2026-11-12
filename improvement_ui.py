"""Sequential human correction window, launched from the existing application."""
import argparse
import threading
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
import cv2
import numpy as np
from PIL import Image,ImageTk
from improvement_review import ReviewQueue


class ImprovementWindow:
    def __init__(self,root,data):
        self.root=root;root.title('蚕CHANNEL — 改善モード');root.geometry('1350x900')
        self.queue=ReviewQueue(data);self.points=[];self.selected=None;self.replace_index=None;self.scale=1
        from instance_lock import FileLock
        self.lock=FileLock(self.queue.folder/'.window.lock')
        if not self.lock.acquire():raise RuntimeError('改善モードは既に開いています。')
        self.scan_result=None;self.closed=False;self.worker=None;self.revision=0;self.pending_auto=False
        self.work_thread=None;self.work_result=None
        top=ttk.Frame(root,padding=8);top.pack(fill='x')
        for label,command in [('輪郭を囲んで追加',self.begin),('選択した輪郭を描き直す',self.redraw),
                              ('選択を削除',self.delete),('1つ戻す',self.undo),('この画像を再解析',self.analyze),
                              ('前の画像',self.previous),('今回は飛ばす',self.skip),('完了して次へ',self.complete),('修正データをZIPへ',self.export)]:
            ttk.Button(top,text=label,command=command).pack(side='left',padx=3)
        self.status=tk.StringVar(value='保存済み画像を探しています…')
        learning=ttk.Frame(root,padding=(8,0));learning.pack(fill='x')
        ttk.Button(learning,text='修正ZIPを取り込む',command=self.import_reviews).pack(side='left',padx=3)
        ttk.Button(learning,text='修正を集計・学習準備',command=self.prepare_reviews).pack(side='left',padx=3)
        ttk.Button(learning,text='背景除外モデルの試験学習',command=self.train_reviews).pack(side='left',padx=3)
        ttk.Label(learning,text='本番モデルは自動で変更しません。').pack(side='left',padx=8)
        root.report_callback_exception=lambda kind,value,tb:self.status.set('操作・保存に失敗しました：'+str(value))
        ttk.Label(root,textvariable=self.status,padding=8,wraplength=1250).pack(fill='x')
        ttk.Label(root,text='追加・描き直し：体の輪郭を順にクリック → Enterで確定。Escで中止。選択：輪郭の内側をクリック。完了は「見える蚕を確認し終えた」の意味です。',padding=8,wraplength=1250).pack(fill='x')
        area=ttk.Frame(root);area.pack(fill='both',expand=True);area.columnconfigure(0,weight=1);area.rowconfigure(0,weight=1)
        self.canvas=tk.Canvas(area,bg='#18251e',highlightthickness=0);self.canvas.grid(row=0,column=0,sticky='nsew')
        vertical=ttk.Scrollbar(area,orient='vertical',command=self.canvas.yview);vertical.grid(row=0,column=1,sticky='ns')
        horizontal=ttk.Scrollbar(area,orient='horizontal',command=self.canvas.xview);horizontal.grid(row=1,column=0,sticky='ew')
        self.canvas.configure(xscrollcommand=horizontal.set,yscrollcommand=vertical.set)
        side=ttk.Frame(area,padding=8);side.grid(row=0,column=2,sticky='ns')
        self.zoom=tk.DoubleVar(value=1)
        ttk.Label(side,text='画像の拡大（スクロールで移動）').pack()
        ttk.Scale(side,from_=1,to=8,variable=self.zoom,command=lambda value:self.render()).pack(fill='x')
        self.list=tk.Listbox(side,width=28,height=20);self.list.pack(fill='x');self.list.bind('<<ListboxSelect>>',self.select)
        ttk.Label(side,text='選択した蚕の見え方').pack(pady=(12,3))
        for text,value in [('頭から尾まで見える','full'),('部分・重なり・見切れ','partial'),('判断できない','unknown')]:
            ttk.Button(side,text=text,command=lambda v=value:self.visibility(v)).pack(fill='x',pady=3)
        ttk.Label(side,text='青：未確認の自動候補\n緑：修正した輪郭\n黄：選択中\n\n隠れた体を想像して囲まないでください。\n全身か不明なら「判断できない」で進められます。\n\n途中の修正は自動保存します。\n学習・送信は自動では行いません。',wraplength=210).pack(pady=15)
        self.canvas.bind('<Button-1>',self.click);self.canvas.bind('<Configure>',lambda e:self.render())
        root.bind('<Return>',lambda e:self.finish_polygon());root.bind('<Escape>',lambda e:self.cancel())
        root.bind('<Control-z>',lambda e:self.undo());root.protocol('WM_DELETE_WINDOW',self.close)
        root.bind('<Control-Return>',lambda e:self.complete())
        def scan():
            try:self.queue.scan();self.scan_result=True
            except Exception as exc:self.scan_result=str(exc)
        threading.Thread(target=scan,daemon=True).start();root.after(80,self.poll)
    def poll(self):
        if self.closed:return
        if self.work_result is not None:
            self.status.set(self.work_result);self.work_result=None
        if self.scan_result is True:
            self.scan_result=None;self.advance()
        elif isinstance(self.scan_result,str):self.status.set(self.scan_result);self.scan_result=None
        if self.worker:
            result=self.worker.receive()
            if result is not None:
                current=self.queue.current
                if current and result['job'].token==(current['sha256'],self.revision):
                    if result.get('error'):self.status.set('再解析できません：'+result['error'])
                    else:
                        self.queue.change()
                        self.queue.objects=[{'id':__import__('uuid').uuid4().hex[:12],
                                             'polygon':c.contour.reshape(-1,2).tolist(),'origin':'model',
                                             'visibility':'unknown','review_status':'unreviewed'} for c in result['candidates']]
                        self.queue.original=__import__('copy').deepcopy(self.queue.objects)
                        self.queue.baseline_available=True
                        self.queue.save(False);self.refresh();self.status.set('自動候補を表示しました。見逃し・誤検出を直してください。')
                if self.pending_auto:self.pending_auto=False;self.analyze()
        self.root.after(80,self.poll)
    def advance(self):
        self.points=[];self.selected=None;self.replace_index=None;self.revision+=1
        try:image=self.queue.next()
        except Exception as exc:self.status.set('画像を開けません：'+str(exc));return
        self.refresh()
        if image is None:
            self.status.set('今回の画像は終わりです。「修正データをZIPへ」で完了分をまとめられます。' if self.queue.items else '保存済みの画像がありません。通常画面で「アーカイブを自動収集・解析」を実行するか、data/archive_batchを引き継いでください。')
        else:
            self.status.set(f'{self.queue.position+1}/{len(self.queue.items)}　{self.queue.current["title"]}　修正後は「完了して次へ」')
            if not self.queue.objects and not self.queue.loaded_draft:self.analyze()
    def refresh(self):
        self.list.delete(0,'end')
        for i,obj in enumerate(self.queue.objects):
            flag={'full':'全身','partial':'部分','unknown':'不明'}[obj['visibility']]
            self.list.insert('end',f'{i+1:02d} {flag}　'+('自動候補' if obj['review_status']=='unreviewed' else '修正済み'))
        if self.selected is not None and self.selected<len(self.queue.objects):self.list.selection_set(self.selected)
        self.render()
    def render(self):
        self.canvas.delete('all')
        if self.queue.current is None:return
        h,w=self.queue.image.shape[:2]
        self.scale=min(max(50,self.canvas.winfo_width())/w,max(50,self.canvas.winfo_height())/h)*self.zoom.get()
        image=Image.fromarray(cv2.cvtColor(self.queue.image,cv2.COLOR_BGR2RGB)).resize((max(1,int(w*self.scale)),max(1,int(h*self.scale))))
        self.photo=ImageTk.PhotoImage(image);self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.canvas.configure(scrollregion=(0,0,w*self.scale,h*self.scale))
        for i,obj in enumerate(self.queue.objects):
            pts=(np.asarray(obj['polygon'])*self.scale).reshape(-1).tolist()
            color='#ffdf30' if i==self.selected else '#55ff86' if obj['review_status']!='unreviewed' else '#54cfff'
            self.canvas.create_polygon(pts,outline=color,fill='',width=2)
            self.canvas.create_text(pts[0],pts[1],text=str(i+1),fill=color,anchor='sw')
        if self.points:
            pts=(np.asarray(self.points)*self.scale).reshape(-1).tolist()
            if len(pts)>=4:self.canvas.create_line(*pts,fill='#ff58dc',width=2)
            for x,y in np.asarray(self.points)*self.scale:self.canvas.create_oval(x-3,y-3,x+3,y+3,fill='#ff58dc')
    def select(self,event=None):
        sel=self.list.curselection()
        if sel:self.selected=sel[0];self.render()
    def begin(self):
        if self.queue.current is None:return
        self.points=[];self.replace_index=None;self.selected=None;self.drawing=True
        self.revision+=1
        self.status.set('体の輪郭を順にクリックし、Enterで確定します。Escで中止。');self.render()
    def redraw(self):
        if self.selected is None:return
        index=self.selected;self.begin();self.replace_index=index
    def click(self,event):
        if self.queue.current is None:return
        x,y=self.canvas.canvasx(event.x)/self.scale,self.canvas.canvasy(event.y)/self.scale;h,w=self.queue.image.shape[:2]
        if not (0<=x<w and 0<=y<h):return
        if getattr(self,'drawing',False):self.points.append([x,y]);self.render();return
        self.selected=None
        for i in reversed(range(len(self.queue.objects))):
            if cv2.pointPolygonTest(np.asarray(self.queue.objects[i]['polygon'],np.float32),(float(x),float(y)),False)>=0:self.selected=i;break
        self.refresh()
    def finish_polygon(self):
        if not getattr(self,'drawing',False):return
        try:
            if self.replace_index is None:self.queue.add(self.points)
            else:self.queue.replace(self.replace_index,self.points)
        except ValueError as exc:self.status.set(str(exc));return
        self.revision+=1;self.points=[];self.drawing=False;self.replace_index=None;self.refresh()
    def cancel(self):self.points=[];self.drawing=False;self.replace_index=None;self.render()
    def delete(self):
        if self.selected is not None:self.queue.delete(self.selected);self.selected=None;self.revision+=1;self.refresh()
    def undo(self):
        if self.points:self.points.pop();self.render();return
        self.queue.undo();self.selected=None;self.revision+=1;self.refresh()
    def visibility(self,value):
        if self.selected is not None:self.queue.visibility(self.selected,value);self.revision+=1;self.refresh()
    def complete(self):
        if self.queue.current is None:return
        if self.points:self.status.set('描いている輪郭をEnterで確定するか、Escで中止してください。');return
        if self.worker and self.worker.busy:self.status.set('再解析が終わってから完了してください。');return
        try:self.queue.save(True)
        except OSError as exc:self.status.set('保存できません：'+str(exc));return
        self.queue.current=None;self.cancel();self.advance()
    def skip(self):self.cancel();self.advance()
    def previous(self):
        if self.queue.position<=0:return
        if self.queue.current is not None:self.queue.save(False)
        self.queue.current=None;self.queue.position-=2;self.cancel();self.advance()
    def analyze(self):
        if self.queue.current is None:return
        if self.work_thread is not None and self.work_thread.is_alive():
            self.status.set('学習準備・試験学習が終わってから再解析できます。');return
        from ai_measurement import AIWorker,AIJob
        from measurement import Settings
        if self.worker is None:self.worker=AIWorker()
        if self.worker.busy:self.pending_auto=True;return
        self.worker.submit(AIJob(self.queue.image.copy(),None,Settings(),None,(self.queue.current['sha256'],self.revision),backend='instance'))
        self.status.set('この画像の候補を自動解析しています。先に修正した場合、遅れて届く候補は適用しません。')
    def export(self):
        try:
            paths,count=self.queue.export();self.status.set(f'{count}枚の修正を{len(paths)}個のZIPへ保存しました：{paths[0].parent}')
            if __import__('os').name=='nt':__import__('os').startfile(str(paths[0].parent))
        except Exception as exc:messagebox.showerror('修正データ',str(exc))
    def background(self,task):
        if self.work_thread is not None and self.work_thread.is_alive():self.status.set('前の処理が動いています。');return
        if self.worker and self.worker.busy:self.status.set('画像の解析が終わってから実行してください。');return
        def run():
            try:self.work_result=task()
            except Exception as exc:self.work_result='処理できません：'+str(exc)
        self.status.set('修正データを処理しています。完了すると結果を表示します。')
        self.work_thread=threading.Thread(target=run,daemon=False);self.work_thread.start()
    def import_reviews(self):
        paths=filedialog.askopenfilenames(title='完了した修正ZIPを選択',filetypes=[('修正ZIP','*.zip')])
        if not paths:return
        def task():
            from review_learning import import_exports
            r=import_exports(self.queue.data,paths)
            return f'取り込み {r["imported"]}枚／重複 {r["duplicates"]}枚／競合 {r["conflicts"]}枚。既存の修正は上書きしません。'
        self.background(task)
    def prepare_reviews(self):
        def task():
            from review_learning import prepare
            folder,r=prepare(self.queue.data)
            return f'完了 {r["images"]}枚・蚕 {r["reviewed_objects"]}領域／追加 {r["added_objects"]}・削除 {r["deleted_objects"]}。学習 {r["train_images"]}枚、検証 {r["validation_images"]}枚、保留 {r["pending_images"]+r["quarantined_images"]}枚。結果：{folder}'
        self.background(task)
    def train_reviews(self):
        def task():
            from pathlib import Path
            from review_learning import prepare,train_candidate
            folder,r=prepare(self.queue.data)
            run,result=train_candidate(folder,Path(__file__).parent/'models/kaiko_texture.pt')
            before,after=result['before'],result['candidate']
            return f'試験学習完了。検証画素の蚕再現率 {before["foreground_recall"]:.1%}→{after["foreground_recall"]:.1%}、背景正解率 {before["background_specificity"]:.1%}→{after["background_specificity"]:.1%}。頭数の精度ではありません。本番未変更。結果：{run}'
        self.background(task)
    def close(self):
        if self.queue.current is not None:self.queue.save(False)
        self.closed=True
        if self.worker:self.worker.close()
        def finish():
            if self.worker:self.worker.thread.join()
            if self.work_thread:self.work_thread.join()
            self.lock.release()
        threading.Thread(target=finish,daemon=False).start()
        self.root.destroy()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args()
    root=tk.Tk()
    try:ImprovementWindow(root,a.data)
    except Exception as exc:messagebox.showerror('改善モード',str(exc));root.destroy()
    else:root.mainloop()
