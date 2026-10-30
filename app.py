"""Windows 11 desktop prototype. No server, API key, or external upload."""
import csv
from dataclasses import replace
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
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image, ImageTk

from measurement import Settings, analyze, calibrate, channel_roi, demo_frame, polyline_length, summarize
from ai_measurement import AIJob, AIWorker

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
        root.title("蚕CHANNEL 体長チェック — 自作モデル試験版")
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
        self.ai_worker = None
        self.ai_mode = "off"
        self.ai_failed = False
        self.ai_pending = False
        self.new_series()
        self.build_ui()
        self.set_static(demo_frame(), self.source)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.tick)

    def new_series(self):
        self.series = uuid.uuid4().hex[:12]
        self.ai_generation = getattr(self, "ai_generation", 0) + 1
        self.history = []
        self.last_log = -1e20
        self.prompt_labels = []
        self.prompt_candidates = []
        self.prompt_selected = None
        self.prompt_saved = False

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
        engines = ttk.Frame(self.root, padding=(8, 2))
        engines.pack(fill="x")
        ttk.Label(engines, text="解析方式").pack(side="left")
        self.ai_engine_choice = tk.StringVar(value="自作モデル（試験）")
        engine_menu = ttk.Combobox(engines, textvariable=self.ai_engine_choice,
                                  values=["自作モデル（試験）", "MobileSAM（従来）"], state="readonly", width=24)
        engine_menu.pack(side="left", padx=5)
        engine_menu.bind("<<ComboboxSelected>>", self.change_ai_engine)
        ttk.Label(engines, text="自作モデルは明るい体・暗い体を一緒に解析します。結果は未確認の候補です。").pack(side="left", padx=8)
        ai = ttk.Frame(self.root, padding=(8, 5))
        ai.pack(fill="x")
        ttk.Button(ai, text="AIでこの画面を解析", command=self.start_ai_once).pack(side="left")
        ttk.Button(ai, text="小さい蚕を細かく探索（試験）", command=self.start_ai_fine).pack(side="left", padx=5)
        ttk.Button(ai, text="AIで映像を繰り返し解析", command=self.start_ai_live).pack(side="left", padx=5)
        ttk.Button(ai, text="色と形の抽出に戻る", command=self.stop_ai).pack(side="left")
        ttk.Label(ai, text="自作モデルは配布された重みを使います。詳細探索・クリック補助は従来モデル用です。", wraplength=300).pack(side="left", padx=8)
        guided = ttk.Frame(self.root, padding=(8, 2))
        guided.pack(fill="x")
        ttk.Button(guided, text="蚕をクリックして拾い直す", command=lambda: self.begin("ai_point")).pack(side="left")
        ttk.Button(guided, text="指定した点で再検出", command=self.request_prompt).pack(side="left", padx=5)
        ttk.Button(guided, text="全身を確認して保存", command=self.finish_prompt).pack(side="left")
        ttk.Label(guided, text="左クリック：体／右クリック：背景。右の候補一覧で切替。全自動解析とは別の補助測定です。").pack(side="left", padx=8)
        manual = ttk.Frame(self.root, padding=(8, 5))
        manual.pack(fill="x")
        ttk.Button(manual, text="③ 手動で体長をなぞる", command=lambda: self.begin("manual")).pack(side="left")
        ttk.Button(manual, text="手動測定を確定・保存", command=self.finish_manual).pack(side="left", padx=5)
        ttk.Button(manual, text="1点戻す", command=self.undo).pack(side="left")
        ttk.Label(manual, text="頭から尾まで体の中央をクリック → 確定。隠れた部分は測定できません。").pack(side="left", padx=10)
        ttk.Button(manual, text="検証用ZIPを保存", command=self.save_sample).pack(side="left", padx=5)

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
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0, minsize=300)
        content.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(content, width=960, height=540, bg="#15201d", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.click)
        self.canvas.bind("<Configure>", lambda event: self.render())
        side = ttk.Frame(content, width=300, padding=(12, 0))
        side.grid(row=0, column=1, sticky="ns")
        side.pack_propagate(False)
        self.result = tk.StringVar()
        ttk.Label(side, textvariable=self.result, font=("Yu Gothic UI", 13), justify="left", wraplength=280).pack(anchor="w")
        self.items = tk.Listbox(side, width=32, height=8)
        self.items.pack(fill="x", pady=10)
        self.items.bind("<<ListboxSelect>>", self.select_prompt)
        ttk.Label(side, text="緑：未確認の候補線　橙：除外領域\n候補の個数は蚕の匹数ではありません。\n番号はこの画面内の仮番号です。", wraplength=280).pack(anchor="w")
        ttk.Label(side, text="候補線の平均長（未確認・この起動内）", wraplength=280).pack(anchor="w", pady=(16, 3))
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
        self.prompt_labels, self.prompt_candidates = [], []
        self.prompt_selected, self.prompt_saved = None, False
        self.candidates = []
        self.mask = None
        self.latest_acquired = None
        self.mode = "auto"
        self.paused = False
        self.ended = False
        self.ai_mode, self.ai_failed, self.ai_pending = "off", False, False
        if self.settings.ai_search == "fine":
            self.show_settings(getattr(self, "standard_settings", replace(self.settings, ai_search="standard")))
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
        self.ai_generation += 1
        self.points = []
        self.prompt_labels, self.prompt_candidates = [], []
        self.prompt_selected, self.prompt_saved = None, False
        self.ai_pending = False
        if mode == "ai_point":
            self.mask = None
        self.show_mask.set(False)
        self.items.delete(0, "end")
        self.result.set("一時停止・操作中\n自動測定の更新を停止しています。")
        self.write_obs("蚕CHANNEL | 設定・手動測定中 | 自動測定値なし")
        descriptions = {"calibrate": "基準の長さをmmで入力し、映像内の基準線の両端をクリックしてください。",
                        "roi": "測定範囲の左上と右下をクリックしてください。定規や配信UIを範囲から外します。",
                        "manual": "一時停止しました。頭の先から尾の先まで、体の中央を曲がりに沿って順にクリックし、確定してください。隠れた部分は測れません。",
                        "ai_point": "一時停止しました。1匹の体の上を左クリックし、『指定した点で再検出』を押します。体に点を追加、背景を右クリックして絞り込めます。"}
        self.tell(descriptions[mode])
        self.render()

    def click(self, event):
        if self.frame is None or self.mode == "auto":
            return
        if getattr(event, "num", 1) == 3 and self.mode != "ai_point":
            return
        x, y = event.x / self.display_scale, event.y / self.display_scale
        h, w = self.frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.mode == "ai_point" and self.roi:
            x0, y0, x1, y1 = self.roi
            if not (x0 <= x < x1 and y0 <= y < y1):
                self.tell("蚕と背景の点は、青い測定範囲の枠内に置いてください。")
                return
        self.points.append((x, y))
        if self.mode == "ai_point":
            self.prompt_labels.append(0 if getattr(event, "num", 1) == 3 else 1)
            self.invalidate_prompt()
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
            if self.mode == "ai_point":
                self.prompt_labels.pop()
                self.invalidate_prompt()
            self.render()

    def invalidate_prompt(self):
        self.ai_generation += 1
        self.prompt_candidates = []
        self.prompt_selected, self.prompt_saved = None, False
        self.items.delete(0, "end")
        self.result.set("AI補助・点を指定中\n『指定した点で再検出』を押してください。")
        self.write_obs("蚕CHANNEL | AI補助の候補を指定中 | 体長の測定値なし")

    def request_prompt(self):
        if self.mode != "ai_point" or not any(self.prompt_labels):
            self.tell("『蚕をクリックして拾い直す』を押し、1匹の体の上を左クリックしてください。")
            return
        if self.ai_worker is None:
            self.ai_worker = AIWorker()
        if self.ai_worker.busy:
            self.tell("前のAI処理が動いています。完了後に『指定した点で再検出』を押してください。")
            return
        self.invalidate_prompt()
        job = AIJob(self.frame.copy(), self.position, self.settings, self.roi,
                    (self.series, self.ai_generation), tuple(self.points), tuple(self.prompt_labels))
        if self.ai_worker.submit(job):
            self.tell("指定した蚕の候補を解析中です。同じ静止画での再検出には画像の解析結果を再利用します。")

    def poll_prompt(self, result):
        job = result["job"]
        if self.mode != "ai_point" or job.token != (self.series, self.ai_generation):
            if self.mode == "ai_point":
                self.tell("前のAI処理が終わりました。現在の点で『指定した点で再検出』を押してください。")
            return
        if result["error"] is not None:
            self.tell(result["error"])
            self.result.set("AI補助を実行できませんでした。\n上の案内を確認してください。")
            return
        self.frame, self.position = job.frame, job.position
        self.prompt_candidates = result["candidates"]
        self.items.delete(0, "end")
        for i, c in enumerate(self.prompt_candidates):
            length = f"{c.length_px / (self.ppmm or 1):.1f} {'mm' if self.ppmm else 'px'}" if c.length_px is not None else "中心線を作れません"
            self.items.insert("end", f"拾い直し {i+1}   {length}")
        if self.prompt_candidates:
            self.items.selection_set(0)
            self.show_prompt(0)
            self.tell(f"拾い直し完了（{result['device']}・{result['seconds']:.1f}秒）。右の一覧で候補を切り替え、頭から尾までの輪郭と線を確認してください。")
        else:
            self.result.set("指定した点に合う候補がありません。\n体の別の場所に点を追加して再検出します。")
            self.tell("候補がありません。点を追加するか『1点戻す』で指定を変え、再検出してください。")
        self.render()

    def select_prompt(self, _event=None):
        if self.mode == "ai_point":
            selection = self.items.curselection()
            if selection:
                self.show_prompt(selection[0])

    def show_prompt(self, index):
        if not 0 <= index < len(self.prompt_candidates):
            return
        if self.prompt_selected != index:
            self.prompt_saved = False
        self.prompt_selected = index
        c = self.prompt_candidates[index]
        if c.length_px is None:
            self.result.set("AI補助・未確認の輪郭\n中心線を作れないため保存できません。\n体と背景の点を追加して再検出してください。")
        else:
            unit = "mm" if self.ppmm is not None else "px"
            state = "全身を手動確認・CSV保存済み" if self.prompt_saved else "未確認の候補"
            self.result.set(f"AI補助・{state}\n中心線長 {c.length_px / (self.ppmm or 1):.1f} {unit}\n頭から尾までの全身か確認してください。\n部分的な体は保存しないでください。")
        if not self.prompt_saved:
            self.write_obs("蚕CHANNEL | AI補助の候補を確認中 | 体長の測定値なし")
        self.render()

    def finish_prompt(self):
        if self.mode != "ai_point" or self.prompt_selected is None:
            self.tell("先に蚕をクリックして再検出し、右の候補一覧から全身を捉えた候補を選んでください。")
            return
        c = self.prompt_candidates[self.prompt_selected]
        if c.length_px is None:
            self.tell("中心線を作れない候補は保存できません。点を追加して再検出するか、手動でなぞってください。")
            return
        if self.prompt_saved:
            self.tell("この候補は既にCSVへ保存しました。別の蚕は『蚕をクリックして拾い直す』から始めます。")
            return
        length = c.length_px / (self.ppmm or 1)
        unit = "mm" if self.ppmm is not None else "px"
        stats = {"count": 1, "excluded_regions": 0, "unit": unit,
                 "mean": length, "min": length, "max": length, "lengths": [length]}
        if self.log(stats, "sam_assisted", c.path.tolist()):
            self.prompt_saved = True
            self.result.set(f"AI補助・全身を手動確認\n中心線長 {length:.1f} {unit}\nCSVに保存しました。\n自動候補の集計には含めません。")
            self.write_obs(f"蚕CHANNEL | AI補助・手動確認の静止値 {length:.1f} {unit}")
            self.tell("CSVに保存しました。『検証用ZIPを保存』で、画像・指定した点・選択した輪郭を1つのZIPに残せます。")

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
        archive_path = directory / f"{stem}.zip"
        temporary_path = directory / f"{stem}.tmp"
        metadata = {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "source": self.source, "video_seconds": self.position,
            "source_resolution": [w, h], "crop_xyxy": [x0, y0, x1, y1],
            "pixels_per_mm": self.ppmm, "settings": self.settings.dictionary(),
            "candidate_method": ("mobile_sam" if self.mode == "ai_point" else self.selected_ai_engine()) if self.mode == "ai_point" or getattr(self, "ai_mode", "off") != "off" else "threshold",
            "candidate_status": "unverified",
            "calibration_points_source_xy": self.calibration_points,
            "manual_points_source_xy": self.points if self.mode == "manual" else [],
            "note": "Source pixels without overlays. Detection candidates are not ground-truth labels.",
        }
        if self.mode == "ai_point" and self.prompt_selected is not None:
            c = self.prompt_candidates[self.prompt_selected]
            metadata["assisted_selection"] = {
                "status": "user_confirmed" if self.prompt_saved else "unverified",
                "prompt_points_source_xy": self.points, "prompt_labels": self.prompt_labels,
                "contour_source_xy": c.contour.reshape(-1, 2).tolist(),
                "centerline_source_xy": c.path.tolist() if c.path is not None else None,
                "length_px": c.length_px,
            }
        try:
            directory.mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", cropped)
            if not ok:
                raise ValueError("PNGの変換に失敗しました。")
            with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(f"{stem}.png", encoded.tobytes())
                archive.writestr(f"{stem}.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            os.replace(temporary_path, archive_path)
        except (OSError, ValueError, cv2.error) as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.tell(f"検証画像の保存に失敗しました: {exc}")
            return None
        self.tell(f"検証用ZIPを保存しました: data/samples/{stem}.zip（画像と設定入り）。『保存先を開く』→samples にあります。")
        return archive_path

    def apply_settings(self):
        try:
            settings = Settings(**{k: (float(v.get()) if k == "min_aspect" else int(v.get())) for k, v in self.params.items()},
                                appearance="dark" if self.appearance.get() == "暗い蚕" else "light",
                                ai_search=self.settings.ai_search)
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
        self.ai_generation += 1
        if getattr(self, "ai_mode", "off") == "single" and not self.paused and self.capture:
            self.ai_mode = "live"
        self.mode, self.points = "auto", []
        if self.capture:
            if self.paused:
                self.capture.pause.set()
            else:
                self.capture.pause.clear()
        self.process()
        self.tell("一時停止中です。" if self.paused else "試験的な候補抽出中です。蚕の認識は未確認です。")

    def start_ai_fine(self):
        if self.selected_ai_engine() == "scratch":
            self.tell("自作モデルは『AIでこの画面を解析』を使ってください。詳細探索とクリック補助は従来モデル用です。")
            return
        if self.frame is None:
            self.tell("先に画像または映像を開いてください。")
            return
        # A visible preset for the shared first-instar image, not an age classifier.
        if self.settings.ai_search != "fine":
            self.standard_settings = self.settings
        self.show_settings(Settings(appearance="dark", min_area=30, max_area=4000, ai_search="fine"))
        self.start_ai_once(fine=True)

    def show_settings(self, settings):
        self.settings = settings
        self.appearance.set("暗い蚕" if settings.appearance == "dark" else "明るい蚕")
        for key, var in self.params.items():
            var.set(str(getattr(settings, key)))

    def start_ai_once(self, fine=False):
        if self.frame is None:
            self.tell("先に画像または映像を開いてください。")
            return
        if not fine and self.settings.ai_search == "fine":
            self.show_settings(getattr(self, "standard_settings", replace(self.settings, ai_search="standard")))
        else:
            self.settings = replace(self.settings, ai_search="fine" if fine else "standard")
        self.paused = True
        if self.capture:
            self.capture.pause.set()
        self.ai_mode, self.ai_failed = "single", False
        self.mode, self.points = "auto", []
        self.new_series()
        self.request_ai(self.frame, self.position)

    def start_ai_live(self):
        if not self.capture or self.ended or self.frame is None:
            self.tell("カメラまたは録画を開いてから使います。静止画には『AIでこの画面を解析』を使ってください。")
            return
        self.ai_mode, self.ai_failed = "live", False
        self.mode, self.points, self.paused = "auto", [], False
        self.capture.pause.clear()
        self.new_series()
        self.request_ai(self.frame, self.position)

    def stop_ai(self):
        self.ai_mode, self.ai_failed, self.ai_pending = "off", False, False
        self.mode, self.points = "auto", []
        self.new_series()
        self.process()
        self.tell("色と形による試験的な抽出に戻りました。蚕の認識は未確認です。")

    def request_ai(self, frame, position):
        if self.ai_failed or self.mode != "auto":
            return
        if self.ai_worker is None:
            self.ai_worker = AIWorker()
        if self.ai_worker.busy:
            self.ai_pending = True
            return
        job = AIJob(frame.copy(), position, self.settings, self.roi,
                    (self.series, self.ai_generation), backend=self.selected_ai_engine())
        if self.ai_worker.submit(job):
            self.ai_pending = False
            timing = "自作モデルで解析中です。" if job.backend == "scratch" else ("詳細探索中です。CPUでは数分かかる場合があります。" if self.settings.ai_search == "fine" else "AI解析中です。CPUでは数十秒かかる場合があります。")
            self.tell(timing + "表示中の輪郭は前回の静止結果です。")
            self.write_obs("蚕CHANNEL | AI候補を解析中\n蚕の認識は未確認・体長の測定値なし")

    def poll_ai(self):
        if self.ai_worker is None:
            return
        result = self.ai_worker.receive()
        if result is None:
            return
        job = result["job"]
        if job.prompt_points is not None:
            self.poll_prompt(result)
            return
        current = (self.series, self.ai_generation)
        if (job.token != current or self.mode != "auto" or self.ai_mode == "off"
                or (self.ai_mode == "live" and self.ended)):
            if self.mode == "ai_point":
                self.tell("前のAI処理が終わりました。現在の点で『指定した点で再検出』を押してください。")
            elif self.ai_pending and self.mode == "auto" and self.ai_mode != "off":
                self.ai_pending = False
                self.process()
            return
        if result["error"] is not None:
            self.ai_failed = True
            self.tell(result["error"])
            self.result.set("AI解析を実行できませんでした。\n上の案内を確認してください。")
            self.write_obs("蚕CHANNEL | AI解析を実行できません\n体長の測定値なし")
            return
        # Draw masks only on the exact frame sent to the worker.
        self.frame, self.position = job.frame, job.position
        self.candidates, self.mask = result["candidates"], None
        self.show_mask.set(False)
        fine = job.settings.ai_search == "fine"
        name = "自作モデル・静止結果" if job.backend == "scratch" else ("AI詳細探索・静止結果" if fine else "AI静止結果")
        prefix = f"{name}（{result['device']}・{result['seconds']:.1f}秒）\n"
        method = "scratch_auto_candidate" if job.backend == "scratch" else ("sam_fine_candidate" if fine else "sam_auto_candidate")
        self.publish_candidates(method, prefix, self.ai_mode == "single")
        self.tell(f"AI解析が完了しました（{result['seconds']:.1f}秒）。緑の線が蚕の全身をたどるか確認してください。部分的な体や誤検出も残ります。")

    def selected_ai_engine(self):
        choice = getattr(self, "ai_engine_choice", None)
        return "scratch" if choice is not None and choice.get().startswith("自作") else "mobile_sam"

    def change_ai_engine(self, _event=None):
        self.ai_mode, self.ai_failed, self.ai_pending = "off", False, False
        self.paused = True
        if self.capture:
            self.capture.pause.set()
        self.mode, self.points, self.candidates, self.mask = "auto", [], [], None
        self.new_series()
        self.items.delete(0, "end")
        self.show_mask.set(False)
        self.result.set("解析方式を変更しました。\nAI解析はまだ実行していません。\n体長の測定値なし")
        self.render()
        self.write_obs("蚕CHANNEL | 解析方式を変更\n体長の測定値なし")
        self.tell("解析方式を変更しました。画像・範囲を確認してAI解析ボタンを押してください。")

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
        if getattr(self, "ai_mode", "off") != "off":
            if self.ai_mode == "live" and self.paused:
                self.result.set("AIの繰り返し解析を一時停止中\n再生すると現在の設定で解析します。")
                self.write_obs("蚕CHANNEL | AI解析を一時停止中\n体長の測定値なし")
                return
            self.request_ai(self.frame, self.position)
            return
        self.candidates, self.mask = analyze(self.frame, self.settings, self.roi)
        self.publish_candidates()

    def publish_candidates(self, method="auto_candidate", prefix="", force_record=False):
        stats = summarize(self.candidates, self.ppmm)
        unit = stats["unit"]
        mean = f"{stats['mean']:.1f} {unit}" if stats["mean"] is not None else "--"
        span = f"{stats['min']:.1f}〜{stats['max']:.1f} {unit}" if stats["count"] else "--"
        scale = "基準設定済み" if self.ppmm is not None else "基準未設定（px）"
        self.result.set(prefix + f"未確認候補 {stats['count']} 領域\n候補線の平均長 {mean}\n最小〜最大 {span}\n除外領域 {stats['excluded_regions']} 個\n{scale}\n※蚕の匹数・体長ではありません")
        self.items.delete(0, "end")
        n = 0
        for c in self.candidates:
            if c.length_px is not None:
                n += 1
                self.items.insert("end", f"候補 {n:02d}   {c.length_px / (self.ppmm or 1):.1f} {unit}")
        state = "停止・静止値" if self.paused or self.ended else ("候補抽出中" if self.capture else "静止画")
        self.write_obs(f"蚕CHANNEL | {state}\n自動検出を調整中・蚕の認識は未確認\n体長の測定値なし")
        now = time.monotonic()
        automatic_due = not self.paused and not self.ended and now - self.last_log >= 10
        if (automatic_due or force_record) and not self.source.startswith("デモ"):
            self.last_log = now
            self.log(stats, method)
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
        shown = self.candidates if self.mode == "auto" else []
        if self.mode == "ai_point" and self.prompt_selected is not None:
            shown = [self.prompt_candidates[self.prompt_selected]]
        for c in shown:
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
            for i, p in enumerate(pts):
                color = ((60, 220, 60) if self.prompt_labels[i] else (40, 60, 255)) if self.mode == "ai_point" else (240, 80, 220)
                cv2.circle(frame, tuple(p), 5, (0, 0, 0), -1)
                cv2.circle(frame, tuple(p), 3, color, -1)
            if self.mode != "ai_point":
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
        self.poll_ai()
        if self.capture:
            try:
                frame, position, acquired = self.capture.frames.get_nowait()
            except queue.Empty:
                if self.capture.terminal and not self.ended:
                    self.ended = True
                    self.ai_generation += 1
                    self.write_obs("蚕CHANNEL | 映像停止 | 現在の測定値なし")
                    self.tell(self.capture.terminal)
            else:
                if not self.paused and not self.ended:
                    self.latest_acquired = acquired
                    if self.frame is None or time.monotonic() - self.last_analysis >= .5:
                        if self.frame is not None and frame.shape[:2] != self.frame.shape[:2]:
                            self.ppmm, self.calibration_points, self.roi = None, [], None
                            self.frame, self.candidates, self.mask = None, [], None
                            self.new_series()
                            self.tell("映像解像度が変わったため、基準と範囲を解除しました。")
                        self.last_analysis = time.monotonic()
                        if self.ai_mode == "live":
                            if not self.ai_worker or not self.ai_worker.busy:
                                self.request_ai(frame, position)
                        else:
                            self.frame, self.position = frame, position
                            self.process()
            if self.capture.camera and not self.paused and not self.ended:
                # Drivers sometimes block without reporting an error. Never retain a live value forever.
                if self.latest_acquired is not None and time.monotonic() - self.latest_acquired > 5:
                    self.write_obs("蚕CHANNEL | 映像更新待ち | 現在の測定値なし")
        self.root.after(80, self.tick)

    def close(self):
        self.stop_capture()
        if self.ai_worker:
            self.ai_worker.close()
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
