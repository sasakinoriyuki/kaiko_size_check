"""Prepare local, unlabelled images for reviewing and training a silkworm model."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
MAX_PAYLOAD = 24 * 1024 * 1024  # Leave room for ZIP headers and metadata below 32 MiB.


def sample_indices(frame_count, count=5):
    if not isinstance(frame_count, int) or frame_count < 1 or not 1 <= count <= 20:
        raise ValueError("録画のフレーム数と抽出枚数を確認してください。抽出枚数は1〜20です。")
    count = min(count, frame_count)
    fractions = [.5] if count == 1 else np.linspace(.1, .9, count)
    return sorted({round(float(f) * (frame_count - 1)) for f in fractions})


def image_samples(paths):
    for name in paths:
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"画像が見つかりません: {path.name}")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"画像を読み込めません: {path.name}")
        yield image, {"source_file": path.name, "source_kind": "image"}


def video_samples(path, count=5):
    path = Path(path)
    if not path.is_file():
        raise ValueError("録画ファイルが見つかりません。")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("録画を開けませんでした。MP4・MKV等の別の録画か、画像を選択してください。")
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(total) or total < 1:
            raise ValueError("録画の長さを取得できませんでした。画像からの準備を使ってください。")
        indices = sample_indices(int(total), count)
        fps = capture.get(cv2.CAP_PROP_FPS)
        seen = set()
        for index in indices:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError("この録画では時間の移動ができません。画像からの準備を使ってください。")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"録画のフレーム {index} を読み込めませんでした。")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not math.isfinite(position) or position < 1:
                raise ValueError("実際に読み込んだフレーム番号を取得できませんでした。")
            actual = int(round(position)) - 1
            if actual in seen:
                continue
            seen.add(actual)
            milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
            seconds = milliseconds / 1000 if math.isfinite(milliseconds) and milliseconds > 0 else (
                actual / fps if math.isfinite(fps) and fps > 0 else None)
            yield image, {"source_file": path.name, "source_kind": "video",
                          "requested_frame": index, "decoded_frame": actual,
                          "video_seconds": seconds}
    finally:
        capture.release()


def prepare_archives(samples, instar, output, progress=None, cancelled=None, max_payload=MAX_PAYLOAD):
    """Preserve source pixels; split archives and remove incomplete output on failure."""
    if not isinstance(instar, int) or not 1 <= instar <= 5:
        raise ValueError("齢は1〜5を選んでください。孵化直後は1齢です。")
    if max_payload < 1:
        raise ValueError("ZIPの容量上限を確認してください。")
    stamp = datetime.now().astimezone()
    token = uuid.uuid4().hex[:8]
    stem = f"training_instar{instar}_{stamp.strftime('%Y%m%d_%H%M%S')}_{token}"
    chunks, chunk, size = [], [], 0
    for index, (image, metadata) in enumerate(samples, 1):
        if cancelled is not None and cancelled.is_set():
            raise ValueError("画像の準備を中止しました。")
        if index > 20:
            raise ValueError("1回に選べる画像は20枚までです。齢ごとに分けて準備してください。")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("PNGへの変換に失敗しました。")
        payload = encoded.tobytes()
        if len(payload) > max_payload:
            raise ValueError("画像1枚がZIPの容量上限を超えました。別の画像を選んでください。")
        if chunk and size + len(payload) > max_payload:
            chunks.append(chunk)
            chunk, size = [], 0
        filename = f"instar{instar}_{token}_{index:03d}.png"
        record = {**metadata, "file": filename,
                  "source_resolution": [int(image.shape[1]), int(image.shape[0])]}
        # Source names are sufficient; never add the user's full local file path.
        record["source_file"] = Path(str(record["source_file"])).name
        chunk.append((filename, payload, record))
        size += len(payload)
        if progress:
            progress(index)
    if chunk:
        chunks.append(chunk)
    if not chunks:
        raise ValueError("準備できる画像がありませんでした。")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    created, temporary = [], []
    try:
        for part, items in enumerate(chunks, 1):
            if cancelled is not None and cancelled.is_set():
                raise ValueError("画像の準備を中止しました。")
            dest = output / f"{stem}_part{part:02d}.zip"
            temp = dest.with_suffix(".tmp")
            temporary.append(temp)
            manifest = {"schema_version": 1, "created_at": stamp.isoformat(timespec="seconds"),
                        "declared_instar": instar, "annotation_status": "unlabelled",
                        "part": part, "parts": len(chunks), "frames": [entry[2] for entry in items],
                        "note": "Original image pixels. Instar is user-declared; there are no body labels yet."}
            with ZipFile(temp, "w", compression=ZIP_DEFLATED) as archive:
                for filename, payload, _ in items:
                    archive.writestr(filename, payload)
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            os.replace(temp, dest)
            created.append(dest)
    except Exception:
        for path in [*temporary, *created]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return created


class PreparationWindow:
    def __init__(self, root):
        self.root, self.paths, self.kind = root, (), None
        self.messages = queue.SimpleQueue()
        self.cancelled = threading.Event()
        self.busy = False
        self.output = BASE / "data/training_exports"
        root.title("蚕CHANNEL 学習用画像の準備")
        root.geometry("800x420")
        root.minsize(680, 380)
        panel = ttk.Frame(root, padding=16)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="同じ齢の画像・録画を選びます。画像を切り抜いたり、縮小したりせずにZIPへ保存します。",
                  wraplength=690).pack(anchor="w")
        options = ttk.Frame(panel)
        options.pack(fill="x", pady=12)
        ttk.Label(options, text="撮影時の齢").pack(side="left")
        self.instar = tk.StringVar(value="1")
        ttk.Combobox(options, textvariable=self.instar, values=["1", "2", "3", "4", "5"],
                     width=4, state="readonly").pack(side="left", padx=6)
        ttk.Label(options, text="齢　（孵化直後は1齢）　録画から取り出す枚数").pack(side="left")
        self.count = tk.StringVar(value="5")
        ttk.Spinbox(options, textvariable=self.count, from_=1, to=20, width=4).pack(side="left", padx=6)
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x")
        self.video_button = ttk.Button(buttons, text="録画を1本選ぶ", command=lambda: self.choose("video"))
        self.video_button.pack(side="left")
        self.image_button = ttk.Button(buttons, text="画像を複数選ぶ", command=lambda: self.choose("image"))
        self.image_button.pack(side="left", padx=8)
        self.youtube_button = ttk.Button(buttons, text="YouTubeの録画URL", command=lambda: self.choose("youtube"))
        self.youtube_button.pack(side="left")
        self.selection = tk.StringVar(value="ファイルを選んでください。元のファイルは変更しません。")
        ttk.Label(panel, textvariable=self.selection, wraplength=690).pack(anchor="w", pady=10)
        self.make_button = ttk.Button(panel, text="学習用の画像ZIPを作る", command=self.start)
        self.make_button.pack(anchor="w")
        self.status = tk.StringVar(value="録画は再生時間の10%〜90%から均等に抽出します。別の日時の画像も役立ちます。")
        ttk.Label(panel, textvariable=self.status, wraplength=690).pack(anchor="w", pady=12)
        ttk.Button(panel, text="出力フォルダを開く", command=self.open_output).pack(anchor="w")
        ttk.Label(panel, text="ZIPはPC内に保存します。自動送信やAIの学習は行いません。", wraplength=690).pack(anchor="w", pady=10)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def choose(self, kind):
        if self.busy:
            return
        if kind == "youtube":
            url = simpledialog.askstring("YouTubeの録画URL", "完了済みの動画URLを1本入力してください。yt-dlpとffmpegが必要です。")
            paths = (url.strip(),) if url and url.strip() else ()
        elif kind == "video":
            name = filedialog.askopenfilename(filetypes=[("録画", "*.mp4 *.mkv *.avi *.mov"), ("すべて", "*.*")])
            paths = (name,) if name else ()
        else:
            paths = filedialog.askopenfilenames(filetypes=[("画像", "*.png *.jpg *.jpeg *.bmp"), ("すべて", "*.*")])
        if paths:
            self.paths, self.kind = tuple(paths), kind
            self.selection.set("YouTubeの動画を1本選択しました。必要な短い区間から画像を取得します。" if kind == "youtube"
                               else f"{'録画' if kind == 'video' else '画像'} {len(paths)} ファイルを選択しました: {Path(paths[0]).name}")

    def start(self):
        if self.busy:
            return
        if not self.paths:
            messagebox.showinfo("ファイル選択", "先に同じ齢の録画または画像を選んでください。")
            return
        try:
            instar, count = int(self.instar.get()), int(self.count.get())
            if not 1 <= instar <= 5 or not 1 <= count <= 20 or len(self.paths) > 20:
                raise ValueError()
        except ValueError:
            messagebox.showerror("設定", "齢は1〜5、枚数・選択画像は1〜20にしてください。")
            return
        self.busy = True
        for button in [self.video_button, self.image_button, self.youtube_button, self.make_button]:
            button.configure(state="disabled")
        self.status.set("画像を準備しています。長い録画では時間がかかる場合があります。")
        paths, kind = self.paths, self.kind

        def run():
            samples = None
            try:
                if kind == "youtube":
                    from collect_youtube_samples import youtube_samples
                    samples = youtube_samples(paths[0], count, self.cancelled)
                else:
                    samples = video_samples(paths[0], count) if kind == "video" else image_samples(paths)
                result = prepare_archives(samples, instar, self.output,
                                          progress=lambda n: self.messages.put(("progress", n)), cancelled=self.cancelled)
                self.messages.put(("done", result))
            except Exception as exc:
                self.messages.put(("error", str(exc) or type(exc).__name__))
            finally:
                if hasattr(samples, "close"):
                    samples.close()

        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.status.set(f"{value} 枚を準備しました。ZIPへの保存が終わるまでお待ちください。")
            else:
                self.busy = False
                for button in [self.video_button, self.image_button, self.youtube_button, self.make_button]:
                    button.configure(state="normal")
                if kind == "done":
                    self.status.set(f"ZIPを {len(value)} 個保存しました。「出力フォルダを開く」から確認できます。別の齢は齢を選び直して準備します。")
                else:
                    self.status.set("準備できませんでした: " + value)
        self.root.after(100, self.poll)

    def open_output(self):
        self.output.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(self.output)
        else:
            messagebox.showinfo("出力先", str(self.output))

    def close(self):
        self.cancelled.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PreparationWindow(root)
    root.mainloop()
