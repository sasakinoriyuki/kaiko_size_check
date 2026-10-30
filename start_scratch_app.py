"""Open the ordinary app with the channel's own model and packaged sample."""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

from app import App


def main():
    root = tk.Tk()
    try:
        app = App(root)
        example = Path(__file__).parent/'examples/startup.png'
        if example.is_file():
            image = cv2.imdecode(np.fromfile(example, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('参考画像を読み込めませんでした。')
            app.reset_source('参考画像（2齢・自動解析は未確認）')
            app.frame = image
            app.render()
            root.after(150, app.start_ai_once)
    except (OSError, ValueError) as exc:
        messagebox.showerror('起動エラー', str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
