"""Windows 11 desktop prototype. No server, API key, or external upload."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid

import cv2
import numpy as np
from PIL import Image, ImageTk

from measurement import Settings, analyze, calibrate, channel_roi, demo_frame, polyline_length, summarize

BASE = Path(__file__).resolve().parent


class Capture:
    """Keep only the latest frame; avoid a growing video latency buffer."""
    def __init__(self, source, camera=False):
        self.source, self.camera = source, camera
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.terminal = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = None
        try:
            backend = cv2.CAP_DSHOW if self.camera and os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                self.terminal = "映像を開けません。カメラ番号・OBSの使用状況を確認してください。"
                return
            if self.camera:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 120 else 30
            due = time.monotonic()
            while not self.stop.is_set():
                if self.pause.is_set() and not self.camera:
                    self.stop.wait(0.03)
                    due = time.monotonic()
                    continue
                ok, frame = cap.read()
                if not ok:
                    self.terminal = "映像が途切れました。再接続してください。" if self.camera else "録画の終端です。"
                    break
                position = None if self.camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not self.pause.is_set():
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.frames.put_nowait((frame, position, time.monotonic()))
                due += 1 / fps
                self.stop.wait(max(0, due - time.monotonic()))
                due = max(due, time.monotonic() - 1 / fps)
        except Exception as exc:
            self.terminal = f"映像の取得エラー: {exc}"
        finally:
            if cap is not None:
                cap.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.5)


class App:
    def __init__(self, root):
        self.root = root
        root.title("蚕CHANNEL 体長チェック — 試作版")
        root.geometry("1380x930")
        root.minsize(1150, 800)
        self.capture = None
        self.frame = None
        self.source = "デモ（合成画像）"
        self.position = None
        self.ppmm = None
        self.calibration_points = []
        self.roi = None
        self.mode = "auto"
        self.points = []
        self.paused = False
        self.ended = False
        self.candidates = []
        self.mask = None
        self.settings = Settings()
        self.last_analysis = 0
        self.latest_acquired = None
        self.display_scale = 1.0
        self.data = BASE / "data"
        self.data.mkdir(exist_ok=True)
        self.obs_path = self.data / "obs_status.txt"
        self.csv_path = self.data / "measurements.csv"
        self.io_error = ""
        self.new_series()
        self.build_ui()
        self.set_static(demo_frame(), self.source)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.tick)

    def new_series(self):
        self.series = uuid.uuid4().hex[:12]
        self.history = []
        self.last_log = -1e20

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="デモ", command=lambda: self.set_static(demo_frame(), "デモ（合成画像）")).pack(side="left")
        ttk.Button(top, text="画像／録画を開く", command=self.open_file).pack(side="left", padx=5)
        ttk.Label(top, text="カメラ番号").pack(side="left")
        self.camera_index = tk.StringVar(value="0")
        ttk.Entry(top, textvariable=self.camera_index, width=4).pack(side="left")
        ttk.Button(top, text="カメラ接続", command=self.open_camera).pack(side="left", padx=5)
        ttk.Button(top, text="一時停止／再生・自動に戻る", command=self.toggle_pause).pack(side="left")
        ttk.Button(top, text="保存先を開く", command=self.open_data).pack(side="right")

        tools = ttk.Frame(self.root, padding=(8, 0))
        tools.pack(fill="x")
        ttk.Button(tools, text="① 基準線を2点指定", command=lambda: self.begin("calibrate")).pack(side="left")
        self.known = tk.StringVar(value="10")
        ttk.Entry(tools, textvariable=self.known, width=6).pack(side="left", padx=4)
        ttk.Label(tools, text="mm").pack(side="left")
        ttk.Button(tools, text="基準を解除（px）", command=self.clear_scale).pack(side="left", padx=6)
        ttk.Button(tools, text="② 測定範囲を2点指定", command=lambda: self.begin("roi")).pack(side="left")
        ttk.Button(tools, text="範囲を全体に戻す", command=self.clear_roi).pack(side="left", padx=5)
        layouts = ttk.Frame(self.root, padding=(8, 5))
        layouts.pack(fill="x")
        ttk.Label(layouts, text="OBS配置").pack(side="left", padx=(0, 5))
        ttk.Button(layouts, text="主映像：左77%", command=self.use_channel_layout).pack(side="left")
        ttk.Button(layouts, text="主映像：左2/3", command=lambda: self.use_channel_layout("two_thirds")).pack(side="left", padx=5)
        self.appearance = tk.StringVar(value="明るい蚕")
        ttk.Label(layouts, text="試験的な候補抽出").pack(side="left", padx=(12, 5))
        ttk.Combobox(layouts, textvariable=self.appearance, values=["明るい蚕", "暗い蚕"],
                     state="readonly", width=12).pack(side="left")
        ttk.Label(layouts, text="齢によらず選択し、検出設定を適用してください。").pack(side="left", padx=8)
        manual = ttk.Frame(self.root, padding=(8, 5))
        manual.pack(fill="x")
        ttk.Button(manual, text="③ 手動で体長をなぞる", command=lambda: self.begin("manual")).pack(side="left")
        ttk.Button(manual, text="手動測定を確定・保存", command=self.finish_manual).pack(side="left", padx=5)
        ttk.Button(manual, text="1点戻す", command=self.undo).pack(side="left")
        ttk.Label(manual, text="頭から尾まで体の中央をクリック → 確定。隠れた部分は測定できません。").pack(side="left", padx=10)
        ttk.Button(manual, text="検証画像を保存", command=self.save_sample).pack(side="left", padx=5)

        params = ttk.Frame(self.root, padding=8)
        params.pack(fill="x")
        self.params = {}
        for label, key in [("明るさ境界", "brightness"), ("彩度上限", "saturation"),
                           ("面積最小px²", "min_area"), ("面積最大px²", "max_area"),
                           ("細長さ下限", "min_aspect")]:
            ttk.Label(params, text=label).pack(side="left", padx=(5, 2))
            var = tk.StringVar(value=str(getattr(self.settings, key)))
            self.params[key] = var
            ttk.Entry(params, textvariable=var, width=7).pack(side="left")
        ttk.Button(params, text="検出設定を適用", command=self.apply_settings).pack(side="left", padx=8)
        self.show_mask = tk.BooleanVar()
        ttk.Checkbutton(params, text="検出マスク表示", variable=self.show_mask, command=self.render).pack(side="left")

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, padding=8, wraplength=1300).pack(fill="x")
        content = ttk.Frame(self.root, padding=8)
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, width=960, height=540, bg="#15201d", highlightthickness=0)
        self.canvas.pack(side="left", anchor="n", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Configure>", lambda event: self.render())
        side = ttk.Frame(content, width=300, padding=(12, 0))
        side.pack(side="right", fill="y")
        self.result = tk.StringVar()
        ttk.Label(side, textvariable=self.result, font=("Yu Gothic UI", 13), justify="left").pack(anchor="w")
        self.items = tk.Listbox(side, width=32, height=8)
        self.items.pack(fill="x", pady=10)
        ttk.Label(side, text="緑：未確認の候補線　橙：除外領域\n候補の個数は蚕の匹数ではありません。\n番号はこの画面内の仮番号です。", wraplength=280).pack(anchor="w")
        ttk.Label(side, text="候補線の平均長（未確認・この起動内）").pack(anchor="w", pady=(16, 3))
        self.chart = tk.Canvas(side, width=280, height=150, bg="white", highlightthickness=0)
        self.chart.pack()
        ttk.Label(side, text="蚕の認識精度は未確認です。\n葉・背景の誤検出も含みます。\n蚕の体長や成長の実測値としては使えません。", wraplength=280).pack(anchor="w", pady=6)
        ttk.Label(self.root, text="試作版：自動検出の実映像での精度は未検証です。身体が隠れている場合は全長を確定できません。", padding=8).pack(fill="x")

    def tell(self, text):
        self.status.set(text + (f"　保存エラー: {self.io_error}" if self.io_error else ""))

    def write_obs(self, text):
        try:
            temp = self.obs_path.with_suffix(".tmp")
            temp.write_text(text + "\n", encoding="utf-8-sig")
            os.replace(temp, self.obs_path)
            self.io_error = ""
        except OSError as exc:
            self.io_error = str(exc)
            self.status.set(f"OBS用テキストの保存に失敗しました: {exc}")

    def stop_capture(self):
        if self.capture:
            self.capture.close()
            self.capture = None

    def reset_source(self, name):
        self.stop_capture()
        self.source = name
        self.frame = None
        self.position = None
        self.ppmm = None
        self.calibration_points = []
        self.roi = None
        self.points = []
        self.candidates = []
        self.mask = None
        self.latest_acquired = None
        self.mode = "auto"
        self.paused = False
        self.ended = False
        self.new_series()
        self.items.delete(0, "end")
        self.result.set("映像待ち・基準未設定（px）")
        self.write_obs("蚕CHANNEL | 映像待ち | 測定値なし")
        self.tell("映像を読み込んでいます。入力の変更時はmmの基準を解除します。")
        self.render()

    def set_static(self, frame, name):
        self.reset_source(name)
        self.frame = frame
        self.process()
        self.tell("静止画です。基準線・測定範囲を指定できます。実映像の調整は画像や録画から始めてください。")

    def open_file(self):
        name = filedialog.askopenfilename(filetypes=[("画像・録画", "*.png *.jpg *.jpeg *.bmp *.mp4 *.mkv *.avi *.mov"), ("すべて", "*.*")])
        if not name:
            return
        if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            try:
                frame = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("画像を読み込めませんでした。")
            except (OSError, ValueError, cv2.error) as exc:
                messagebox.showerror("画像エラー", str(exc))
                return
            self.set_static(frame, name)
        else:
            self.reset_source(name)
            self.capture = Capture(name)

    def open_camera(self):
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("カメラ番号", "0以上の整数にしてください。通常は0または1です。")
            return
        self.reset_source(f"camera:{index}")
        self.capture = Capture(index, camera=True)

    def open_data(self):
        if os.name == "nt":
            os.startfile(self.data)
        else:
            messagebox.showinfo("保存先", str(self.data))

    def begin(self, mode):
        if self.frame is None:
            return
        self.paused = True
        if self.capture:
            self.capture.pause.set()
        self.mode = mode
        self.points = []
        self.show_mask.set(False)
        self.items.delete(0, "end")
        self.result.set("一時停止・操作中\n自動測定の更新を停止しています。")
        self.write_obs("蚕CHANNEL | 設定・手動測定中 | 自動測定値なし")
        descriptions = {"calibrate": "基準の長さをmmで入力し、映像内の基準線の両端をクリックしてください。",
                        "roi": "測定範囲の左上と右下をクリックしてください。定規や配信UIを範囲から外します。",
                        "manual": "一時停止しました。頭の先から尾の先まで、体の中央を曲がりに沿って順にクリックし、確定してください。隠れた部分は測れません。"}
        self.tell(descriptions[mode])
        self.render()

    def click(self, event):
        if self.frame is None or self.mode == "auto":
            return
        x, y = event.x / self.display_scale, event.y / self.display_scale
        h, w = self.frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.points.append((x, y))
        if self.mode == "calibrate" and len(self.points) == 2:
            try:
                ppmm = calibrate(self.points, float(self.known.get()))
            except ValueError as exc:
                self.points = []
                messagebox.showerror("基準線", str(exc))
                return
            self.ppmm = ppmm
            self.calibration_points = list(self.points)
            self.new_series()
            self.mode, self.points = "auto", []
            self.process()
            self.tell(f"基準設定: {ppmm:.3f} px/mm。一時停止中です。カメラや容器の高さ・ズームを変えたら再設定してください。")
        elif self.mode == "roi" and len(self.points) == 2:
            a, b = self.points
            roi = (int(min(a[0], b[0])), int(min(a[1], b[1])), int(max(a[0], b[0])), int(max(a[1], b[1])))
            if roi[2] - roi[0] < 20 or roi[3] - roi[1] < 20:
                self.points = []
                messagebox.showerror("範囲", "縦横20px以上の範囲を選んでください。")
                return
            self.roi = roi
            self.new_series()
            self.mode, self.points = "auto", []
            self.process()
            self.tell("測定範囲を設定しました。一時停止中です。再生ボタンで自動測定に戻ります。")
        self.render()

    def undo(self):
        if self.points:
            self.points.pop()
            self.render()

    def finish_manual(self):
        if self.mode != "manual" or len(self.points) < 2:
            messagebox.showinfo("手動測定", "手動でなぞるボタンを押し、体の中央を2点以上クリックしてください。")
            return
        length = polyline_length(self.points) / (self.ppmm or 1)
        unit = "mm" if self.ppmm is not None else "px"
        stats = {"count": 1, "excluded_regions": 0, "unit": unit,
                 "mean": length, "min": length, "max": length, "lengths": [length]}
        saved = self.log(stats, "manual", self.points)
        self.result.set(f"手動測定\n体長 {length:.1f} {unit}\n自動候補の集計には含めません。")
        self.write_obs(f"蚕CHANNEL | 手動測定・静止値 {length:.1f} {unit}")
        self.tell(f"手動体長 {length:.1f} {unit}。" + ("CSVに保存しました。" if saved else "保存に失敗しました。"))

    def clear_scale(self):
        self.ppmm, self.calibration_points = None, []
        self.mode, self.points = "auto", []
        self.new_series()
        self.process()
        self.tell("基準を解除しました。px単位です。")

    def clear_roi(self):
        self.roi = None
        self.mode, self.points = "auto", []
        self.new_series()
        self.process()

    def use_channel_layout(self, layout="wide"):
        if self.frame is None:
            self.tell("先にOBSの映像または画像・録画を開いてください。")
            return
        self.paused = True
        if self.capture:
            self.capture.pause.set()
        self.roi = channel_roi(self.frame.shape, layout)
        self.mode, self.points = "auto", []
        self.new_series()
        self.process()
        name = "左77%" if layout == "wide" else "左2/3"
        self.tell(f"{name}を測定範囲にしました。右のUIを除外しています。枠を確認し、再生してください。YouTubeのページや操作バーが映る場合は範囲を手動調整します。")

    def save_sample(self):
        if self.frame is None:
            self.tell("検証画像を保存するには、先に映像を開いてください。")
            return None
        h, w = self.frame.shape[:2]
        x0, y0, x1, y1 = self.roi or (0, 0, w, h)
        cropped = self.frame[y0:y1, x0:x1]
        stamp = datetime.now().astimezone()
        stem = f"sample_{stamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        directory = self.data / "samples"
        image_path = directory / f"{stem}.png"
        metadata_path = directory / f"{stem}.json"
        metadata = {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "source": self.source, "video_seconds": self.position,
            "source_resolution": [w, h], "crop_xyxy": [x0, y0, x1, y1],
            "pixels_per_mm": self.ppmm, "settings": self.settings.dictionary(),
            "calibration_points_source_xy": self.calibration_points,
            "manual_points_source_xy": self.points if self.mode == "manual" else [],
            "note": "Source pixels without overlays. Detection candidates are not ground-truth labels.",
        }
        try:
            directory.mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", cropped)
            if not ok:
                raise ValueError("PNGの変換に失敗しました。")
            image_path.write_bytes(encoded.tobytes())
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError, cv2.error) as exc:
            for partial in (image_path, metadata_path):
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
            self.tell(f"検証画像の保存に失敗しました: {exc}")
            return None
        self.tell(f"検証画像を保存しました: data/samples/{stem}.png（選択範囲・元の解像度）。外部送信はしていません。")
        return image_path

    def apply_settings(self):
        try:
            settings = Settings(**{k: (float(v.get()) if k == "min_aspect" else int(v.get())) for k, v in self.params.items()},
                                appearance="dark" if self.appearance.get() == "暗い蚕" else "light")
            settings.validate()
        except ValueError as exc:
            messagebox.showerror("検出設定", str(exc))
            return
        self.settings = settings
        self.new_series()
        self.mode, self.points = "auto", []
        self.process()
        self.tell("検出設定を適用しました。緑の線が実際の頭から尾までを表すか確認してください。")

    def toggle_pause(self):
        if self.frame is None or self.ended:
            self.tell("映像を開いてください。録画の終端・切断後は入力を開き直します。")
            return
        self.paused = not self.paused if self.mode == "auto" else False
        self.mode, self.points = "auto", []
        if self.capture:
            if self.paused:
                self.capture.pause.set()
            else:
                self.capture.pause.clear()
        self.process()
        self.tell("一時停止中です。" if self.paused else "試験的な候補抽出中です。蚕の認識は未確認です。")

    def log(self, stats, method="auto_candidate", points=None):
        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "series": self.series, "source": self.source, "video_seconds": self.position,
            "method": method, "unit": stats["unit"], "count": stats["count"],
            "excluded_regions": stats["excluded_regions"], "mean": stats["mean"],
            "min": stats["min"], "max": stats["max"],
            "lengths": json.dumps(stats["lengths"]), "pixels_per_mm": self.ppmm,
            "resolution": f"{self.frame.shape[1]}x{self.frame.shape[0]}",
            "roi": json.dumps(self.roi), "settings": json.dumps(self.settings.dictionary()),
            "calibration_points": json.dumps(self.calibration_points),
            "manual_points": json.dumps(points),
        }
        try:
            needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            self.io_error = ""
            return True
        except OSError as exc:
            self.io_error = str(exc)
            self.tell("CSVの保存に失敗しました。Excelで開いている場合は閉じてください。")
            return False

    def process(self):
        if self.frame is None:
            return
        self.candidates, self.mask = analyze(self.frame, self.settings, self.roi)
        stats = summarize(self.candidates, self.ppmm)
        unit = stats["unit"]
        mean = f"{stats['mean']:.1f} {unit}" if stats["mean"] is not None else "--"
        span = f"{stats['min']:.1f}〜{stats['max']:.1f} {unit}" if stats["count"] else "--"
        scale = "基準設定済み" if self.ppmm is not None else "基準未設定（px）"
        self.result.set(f"未確認候補 {stats['count']} 領域\n候補線の平均長 {mean}\n最小〜最大 {span}\n除外領域 {stats['excluded_regions']} 個\n{scale}\n※蚕の匹数・体長ではありません")
        self.items.delete(0, "end")
        n = 0
        for c in self.candidates:
            if c.length_px is not None:
                n += 1
                self.items.insert("end", f"候補 {n:02d}   {c.length_px / (self.ppmm or 1):.1f} {unit}")
        state = "停止・静止値" if self.paused or self.ended else ("候補抽出中" if self.capture else "静止画")
        self.write_obs(f"蚕CHANNEL | {state}\n自動検出を調整中・蚕の認識は未確認\n体長の測定値なし")
        now = time.monotonic()
        if not self.paused and not self.ended and now - self.last_log >= 10 and not self.source.startswith("デモ"):
            self.last_log = now
            self.log(stats)
            # Keep a fixed 600-sample window, including gaps when no body is measurable.
            self.history.append((now, stats["mean"]))
            self.history = self.history[-600:]
        self.render()

    def render(self):
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        if self.frame is None:
            self.draw_chart()
            return
        frame = self.frame.copy()
        if self.show_mask.get() and self.mask is not None:
            frame[:] = 0
            x0, y0, x1, y1 = self.roi or (0, 0, frame.shape[1], frame.shape[0])
            frame[y0:y1, x0:x1] = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        number = 0
        for c in self.candidates if self.mode == "auto" else []:
            color = (80, 220, 100) if c.length_px is not None else (0, 165, 255)
            cv2.drawContours(frame, [c.contour], -1, color, 1)
            if c.path is not None:
                number += 1
                pts = np.rint(c.path).astype(np.int32)
                cv2.polylines(frame, [pts], False, (40, 250, 90), 2)
                x, y = pts[len(pts) // 2]
                cv2.putText(frame, str(number), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3)
                cv2.putText(frame, str(number), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 255, 110), 1)
        if self.roi:
            cv2.rectangle(frame, self.roi[:2], self.roi[2:], (230, 180, 30), 2)
        if len(self.calibration_points) == 2:
            pts = np.rint(self.calibration_points).astype(np.int32)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255, 180, 30), 2)
        if self.points:
            pts = np.rint(self.points).astype(np.int32)
            for p in pts:
                cv2.circle(frame, tuple(p), 4, (240, 80, 220), -1)
            cv2.polylines(frame, [pts], False, (240, 80, 220), 2)
        h, w = frame.shape[:2]
        self.display_scale = min(max(320, self.canvas.winfo_width()) / w,
                                 max(240, self.canvas.winfo_height()) / h, 1.5)
        size = (max(1, int(w * self.display_scale)), max(1, int(h * self.display_scale)))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(size, Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.draw_chart()

    def draw_chart(self):
        self.chart.delete("all")
        values = [v for _, v in self.history if v is not None]
        if len(values) < 2:
            self.chart.create_text(140, 70, text="有効な記録が2点以上で表示", fill="#555")
            return
        lo, hi = min(values), max(values)
        padding = max((hi - lo) * .1, .5)
        lo, hi = lo - padding, hi + padding
        self.chart.create_text(5, 8, text=f"{hi:.1f} {'mm' if self.ppmm else 'px'}", anchor="nw")
        self.chart.create_text(5, 127, text=f"{lo:.1f}", anchor="nw")
        self.chart.create_line(45, 20, 45, 125, 270, 125, fill="#bbb")
        start, end = self.history[0][0], self.history[-1][0]
        previous = None
        for t, value in self.history:
            if value is None:
                previous = None
                continue
            p = (45 + 220 * (t - start) / max(end - start, 1), 125 - 100 * (value - lo) / (hi - lo))
            if previous is not None:
                self.chart.create_line(*previous, *p, fill="#16816b", width=2)
            self.chart.create_oval(p[0]-2, p[1]-2, p[0]+2, p[1]+2, fill="#16816b", outline="")
            previous = p
        self.chart.create_text(260, 143, text=f"{(end-start)/60:.1f}分間", anchor="e")

    def tick(self):
        if self.capture:
            try:
                frame, position, acquired = self.capture.frames.get_nowait()
            except queue.Empty:
                if self.capture.terminal and not self.ended:
                    self.ended = True
                    self.write_obs("蚕CHANNEL | 映像停止 | 現在の測定値なし")
                    self.tell(self.capture.terminal)
            else:
                if not self.paused and not self.ended:
                    self.latest_acquired = acquired
                    if self.frame is None or time.monotonic() - self.last_analysis >= .5:
                        if self.frame is not None and frame.shape[:2] != self.frame.shape[:2]:
                            self.ppmm, self.calibration_points, self.roi = None, [], None
                            self.new_series()
                            self.tell("映像解像度が変わったため、基準と範囲を解除しました。")
                        self.frame, self.position = frame, position
                        self.last_analysis = time.monotonic()
                        self.process()
            if self.capture.camera and not self.paused and not self.ended:
                # Drivers sometimes block without reporting an error. Never retain a live value forever.
                if self.latest_acquired is not None and time.monotonic() - self.latest_acquired > 5:
                    self.write_obs("蚕CHANNEL | 映像更新待ち | 現在の測定値なし")
        self.root.after(80, self.tick)

    def close(self):
        self.stop_capture()
        self.write_obs("蚕CHANNEL | アプリ終了 | 測定値なし")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        App(root)
    except OSError as exc:
        messagebox.showerror("保存先エラー", f"アプリのフォルダに書き込めません。書き込み可能な場所に展開してください。\n{exc}")
        root.destroy()
    else:
        root.mainloop()
