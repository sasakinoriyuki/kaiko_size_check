"""Start with a local saved sample when available; never upload image data."""
import io
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from zipfile import ZipFile, BadZipFile

import cv2
import numpy as np
from PIL import Image

from app import App


def local_reference(base):
    base = Path(base)
    directory = base/'data/samples'
    candidates = sorted(directory.glob('sample_*.zip'),key=lambda p:p.stat().st_mtime,reverse=True) if directory.is_dir() else []
    for path in candidates[:20]:
        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
                pictures = [item for item in entries if item.filename.lower().endswith('.png')]
                if len(entries)>10 or len(pictures)!=1 or sum(i.file_size for i in entries)>32*1024*1024:
                    continue
                raw = archive.read(pictures[0])
            with Image.open(io.BytesIO(raw)) as probe:
                if probe.width*probe.height>20_000_000: continue
            image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is not None: return image,'保存済みの画像（輪郭確認）'
        except (OSError,ValueError,BadZipFile,RuntimeError,Image.DecompressionBombError,cv2.error):
            continue
    # Compatibility with older local packages; new distributions contain no photos.
    for name,label in [('instar5.png','参考画像（5齢・輪郭確認）'),('startup.png','参考画像（輪郭確認）')]:
        path=base/'examples'/name
        if path.is_file():
            image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is not None: return image,label
    return None,None


def main(on_ready=None):
    root=tk.Tk()
    try:
        app=App(root)
        base=Path(__file__).parent
        image,label=local_reference(base)
        if image is not None:
            app.ai_engine_choice.set('自作輪郭（比較）')
            app.reset_source(label);app.frame=image;app.render()
            root.after(150,app.start_ai_once)
        else:
            app.paused=True
            app.candidates=[];app.mask=None
            app.result.set('画像・録画またはカメラを選んでください。\n体長の測定値なし')
            app.tell('準備できました。「画像／録画を開く」または「カメラ接続」から使えます。')
            app.render()
    except (OSError,ValueError) as exc:
        messagebox.showerror('起動エラー',str(exc));root.destroy();return
    root.update_idletasks()
    if on_ready is not None:on_ready()
    root.mainloop()


if __name__=='__main__':main()
