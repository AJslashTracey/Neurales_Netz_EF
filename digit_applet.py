import tkinter as tk
from tkinter import ttk

import numpy as np

from network import NeuralNetwork


CANVAS_SIZE = 280
MODEL_SIZE = 28
DOWNSAMPLE_FACTOR = CANVAS_SIZE // MODEL_SIZE
BRUSH_RADIUS = 12
AUTO_PREDICT_DEBOUNCE_MS = 220
BUFFER_STROKE_STEP = 3
GRAPH_WIDTH = 360
GRAPH_HEIGHT = 170
TREND_WIDTH = 360
TREND_HEIGHT = 95
PREVIEW_SCALE = 4
HISTORY_LENGTH = 40

THEME = {
    "background": "#d1d1d1",
    "card": "#d7e4ff",
    "card_border": "#ba5aff",
    "accent": "#fcb065",
    "accent_strong": "#ffb000",
    "alert": "#e6583e",
    "violet": "#ba5aff",
    "blue": "#5ac8ff",
    "grid": "#ffb000",
}


class DigitApplet:
    def __init__(self, model: NeuralNetwork, debug: bool = False) -> None:
        self.model = model
        self.debug = debug
        self.root = tk.Tk()
        self.root.title("Digit Predictor")
        self.root.resizable(False, False)

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("App.TFrame", background=THEME["background"])
        self.style.configure(
            "Card.TFrame",
            background=THEME["card"],
            relief="flat",
            borderwidth=2,
            highlightbackground=THEME["card_border"],
            highlightcolor=THEME["card_border"],
        )
        self.style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=THEME["card"],
            background=THEME["blue"],
            bordercolor=THEME["card_border"],
        )
        self.style.configure(
            "Accent.TButton",
            background=THEME["accent"],
            foreground="#1b1b1b",
            borderwidth=0,
            padding=6,
            relief="flat",
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", THEME["accent_strong"])],
            relief=[("pressed", "groove")],
        )
        self.style.configure(
            "Accent.TCheckbutton",
            background=THEME["card"],
            foreground=THEME["violet"],
        )

        self.root.configure(bg=THEME["background"])

        self.buffer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.auto_predict_job: str | None = None

        self.auto_predict_var = tk.BooleanVar(value=False)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.snap_to_grid_var = tk.BooleanVar(value=True)
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.status_var = tk.StringVar(value="Draw a digit, then click Predict.")
        self.confidence_var = tk.StringVar(value="Top confidence: -")
        self.margin_var = tk.StringVar(value="Top-2 margin: -")
        self.certainty_var = tk.StringVar(value="Certainty score: -")

        self.last_processed = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        self.recent_confidences: list[float] = []
        self.recent_margins: list[float] = []

        self._build_ui()
        # Bring the window to front on launch.
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self.root.geometry("+220+120")

        if self.debug:
            print("Tk window created and focused.", flush=True)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10, style="App.TFrame")
        outer.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(outer, style="Card.TFrame")
        left.grid(row=0, column=0, rowspan=8, padx=(0, 14), sticky="n")

        right = ttk.Frame(outer, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nw")

        ttk.Label(
            left,
            text="Drawing Grid (28x28)",
            font=("Helvetica", 12, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground=THEME["violet"],
        )
        self.canvas.grid(row=1, column=0, sticky="w")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        ttk.Checkbutton(
            left,
            text="Show grid",
            variable=self.show_grid_var,
            command=self._redraw_grid,
            style="Accent.TCheckbutton",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        ttk.Label(
            right,
            textvariable=self.prediction_var,
            font=("Helvetica", 16, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.top_labels: list[ttk.Label] = []
        self.top_bars: list[ttk.Progressbar] = []

        ttk.Label(
            right,
            text="Top predictions",
            font=("Helvetica", 11, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=1, column=0, sticky="w", pady=(2, 4))

        top3_frame = tk.Frame(right, bg=THEME["card"])
        top3_frame.grid(row=2, column=0, sticky="w")
        for idx in range(3):
            lbl = ttk.Label(
                top3_frame,
                text=f"#{idx + 1}: -",
                width=12,
                background=THEME["card"],
                foreground=THEME["violet"],
            )
            bar = ttk.Progressbar(
                top3_frame,
                orient="horizontal",
                length=210,
                maximum=1.0,
                style="Accent.Horizontal.TProgressbar",
            )
            lbl.grid(row=idx, column=0, sticky="w", pady=3)
            bar.grid(row=idx, column=1, padx=(8, 0), pady=3, sticky="w")
            self.top_labels.append(lbl)
            self.top_bars.append(bar)

        ttk.Label(
            right,
            text="Probability graph (0-9)",
            font=("Helvetica", 11, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=3, column=0, sticky="w", pady=(10, 4))
        self.graph_canvas = tk.Canvas(
            right,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            bg=THEME["card"],
            highlightthickness=1,
            highlightbackground=THEME["violet"],
        )
        self.graph_canvas.grid(row=4, column=0, sticky="w")

        ttk.Label(
            right,
            text="Input + confidence diagnostics",
            font=("Helvetica", 11, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=5, column=0, sticky="w", pady=(10, 4))

        diagnostics = tk.Frame(right, bg=THEME["card"])
        diagnostics.grid(row=6, column=0, sticky="w")

        left_diag = tk.Frame(diagnostics, bg=THEME["card"])
        left_diag.grid(row=0, column=0, sticky="nw", padx=(0, 14))
        right_diag = tk.Frame(diagnostics, bg=THEME["card"])
        right_diag.grid(row=0, column=1, sticky="nw")

        ttk.Label(
            left_diag,
            text="Processed 28x28 preview",
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=0, column=0, sticky="w")
        self.preview_canvas = tk.Canvas(
            left_diag,
            width=MODEL_SIZE * PREVIEW_SCALE,
            height=MODEL_SIZE * PREVIEW_SCALE,
            bg=THEME["card"],
            highlightthickness=1,
            highlightbackground=THEME["violet"],
        )
        self.preview_canvas.grid(row=1, column=0, pady=(4, 0))

        ttk.Label(
            right_diag,
            textvariable=self.confidence_var,
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=0, column=0, sticky="w")
        self.confidence_bar = ttk.Progressbar(
            right_diag,
            orient="horizontal",
            length=190,
            maximum=1.0,
            style="Accent.Horizontal.TProgressbar",
        )
        self.confidence_bar.grid(row=1, column=0, sticky="w", pady=(2, 6))

        ttk.Label(
            right_diag,
            textvariable=self.margin_var,
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=2, column=0, sticky="w")
        self.margin_bar = ttk.Progressbar(
            right_diag,
            orient="horizontal",
            length=190,
            maximum=1.0,
            style="Accent.Horizontal.TProgressbar",
        )
        self.margin_bar.grid(row=3, column=0, sticky="w", pady=(2, 6))

        ttk.Label(
            right_diag,
            textvariable=self.certainty_var,
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=4, column=0, sticky="w")
        self.certainty_bar = ttk.Progressbar(
            right_diag,
            orient="horizontal",
            length=190,
            maximum=1.0,
            style="Accent.Horizontal.TProgressbar",
        )
        self.certainty_bar.grid(row=5, column=0, sticky="w", pady=(2, 0))

        ttk.Label(
            right,
            text="Confidence trend (recent predictions)",
            font=("Helvetica", 10, "bold"),
            background=THEME["card"],
            foreground=THEME["violet"],
        ).grid(row=7, column=0, sticky="w", pady=(10, 3))
        self.trend_canvas = tk.Canvas(
            right,
            width=TREND_WIDTH,
            height=TREND_HEIGHT,
            bg=THEME["card"],
            highlightthickness=1,
            highlightbackground=THEME["violet"],
        )
        self.trend_canvas.grid(row=8, column=0, sticky="w")

        controls = tk.Frame(right, bg=THEME["card"])
        controls.grid(row=9, column=0, sticky="w", pady=(10, 0))

        ttk.Button(
            controls,
            text="Predict",
            command=self.predict,
            style="Accent.TButton",
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            controls,
            text="Clear",
            command=self.clear,
            style="Accent.TButton",
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Checkbutton(
            controls,
            text="Auto-predict",
            variable=self.auto_predict_var,
            style="Accent.TCheckbutton",
        ).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Snap to cells",
            variable=self.snap_to_grid_var,
            style="Accent.TCheckbutton",
        ).grid(row=0, column=3, padx=(12, 0), sticky="w")

        self.status_label = ttk.Label(
            right,
            textvariable=self.status_var,
            background=THEME["card"],
            foreground=THEME["accent"],
        )
        self.status_label.grid(row=10, column=0, sticky="w", pady=(8, 0))

        self._redraw_grid()
        self._draw_probability_graph(np.zeros(10, dtype=np.float32))
        self._draw_processed_preview(self.last_processed)
        self._draw_trend_graph()

    def _redraw_grid(self) -> None:
        self.canvas.delete("grid")
        if not self.show_grid_var.get():
            return

        step = DOWNSAMPLE_FACTOR
        color = THEME["violet"]
        for x in range(0, CANVAS_SIZE + 1, step):
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, fill=color, tags="grid")
        for y in range(0, CANVAS_SIZE + 1, step):
            self.canvas.create_line(0, y, CANVAS_SIZE, y, fill=color, tags="grid")

    def _draw_probability_graph(self, probs: np.ndarray) -> None:
        self.graph_canvas.delete("all")

        margin_left = 18
        margin_bottom = 20
        max_h = GRAPH_HEIGHT - 34
        bar_width = (GRAPH_WIDTH - margin_left - 10) / 10.0

        self.graph_canvas.create_line(
            margin_left,
            GRAPH_HEIGHT - margin_bottom,
            GRAPH_WIDTH - 6,
            GRAPH_HEIGHT - margin_bottom,
            fill=THEME["violet"],
        )

        for digit in range(10):
            p = float(probs[digit]) if probs.size else 0.0
            x0 = margin_left + digit * bar_width + 3
            x1 = margin_left + (digit + 1) * bar_width - 3
            y1 = GRAPH_HEIGHT - margin_bottom
            y0 = y1 - p * max_h
            self.graph_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=THEME["blue"],
                outline="",
            )
            self.graph_canvas.create_text(
                (x0 + x1) / 2,
                GRAPH_HEIGHT - 8,
                text=str(digit),
                fill=THEME["violet"],
                font=("Helvetica", 9),
            )

    def _draw_processed_preview(self, img28: np.ndarray) -> None:
        self.preview_canvas.delete("all")
        for y in range(MODEL_SIZE):
            for x in range(MODEL_SIZE):
                v = float(np.clip(img28[y, x], 0.0, 1.0))
                gray = int(v * 255)
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0 = x * PREVIEW_SCALE
                y0 = y * PREVIEW_SCALE
                self.preview_canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + PREVIEW_SCALE,
                    y0 + PREVIEW_SCALE,
                    fill=color,
                    outline="",
                )
        self.preview_canvas.create_rectangle(
            0,
            0,
            MODEL_SIZE * PREVIEW_SCALE,
            MODEL_SIZE * PREVIEW_SCALE,
            outline="#6c6c6c",
        )

    def _update_confidence_metrics(self, probs: np.ndarray) -> None:
        sorted_probs = np.sort(probs)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
        margin = max(0.0, top1 - top2)
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        entropy_norm = entropy / np.log(10.0)
        certainty = float(np.clip(1.0 - entropy_norm, 0.0, 1.0))

        self.confidence_var.set(f"Top confidence: {top1 * 100:.1f}%")
        self.margin_var.set(f"Top-2 margin: {margin * 100:.1f}%")
        self.certainty_var.set(f"Certainty score: {certainty * 100:.1f}%")
        self.confidence_bar["value"] = top1
        self.margin_bar["value"] = margin
        self.certainty_bar["value"] = certainty

        self.recent_confidences.append(top1)
        self.recent_margins.append(margin)
        self.recent_confidences = self.recent_confidences[-HISTORY_LENGTH:]
        self.recent_margins = self.recent_margins[-HISTORY_LENGTH:]
        self._draw_trend_graph()

    def _draw_trend_graph(self) -> None:
        self.trend_canvas.delete("all")
        w = TREND_WIDTH
        h = TREND_HEIGHT
        left, right = 10, w - 10
        top, bottom = 10, h - 18

        self.trend_canvas.create_line(left, bottom, right, bottom, fill=THEME["violet"])
        self.trend_canvas.create_line(left, top, left, bottom, fill=THEME["violet"])

        if not self.recent_confidences:
            self.trend_canvas.create_text(
                w / 2,
                h / 2,
                text="No predictions yet",
                fill=THEME["accent"],
                font=("Helvetica", 10),
            )
            return

        def _points(values: list[float]) -> list[float]:
            n = len(values)
            points: list[float] = []
            for i, v in enumerate(values):
                x = left + (i / max(n - 1, 1)) * (right - left)
                y = bottom - float(np.clip(v, 0.0, 1.0)) * (bottom - top)
                points.extend([x, y])
            return points

        conf_pts = _points(self.recent_confidences)
        margin_pts = _points(self.recent_margins)
        if len(conf_pts) >= 4:
            self.trend_canvas.create_line(*margin_pts, fill=THEME["accent"], width=2, smooth=True)
            self.trend_canvas.create_line(*conf_pts, fill=THEME["blue"], width=2, smooth=True)

        self.trend_canvas.create_text(
            right - 42,
            top + 8,
            text="conf",
            fill=THEME["blue"],
            font=("Helvetica", 9, "bold"),
        )
        self.trend_canvas.create_text(
            right - 42,
            top + 22,
            text="margin",
            fill=THEME["accent"],
            font=("Helvetica", 9),
        )

    def on_press(self, event: tk.Event) -> None:
        self.last_x = int(event.x)
        self.last_y = int(event.y)
        if self.snap_to_grid_var.get():
            self._paint_cell_from_canvas(self.last_x, self.last_y)
        else:
            self._stamp(self.last_x, self.last_y)
        if self.auto_predict_var.get():
            self._schedule_auto_predict()

    def on_drag(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y

        if self.snap_to_grid_var.get():
            self._draw_cell_line(self.last_x, self.last_y, x, y)
        else:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                x,
                y,
                fill="white",
                width=BRUSH_RADIUS * 2,
                capstyle=tk.ROUND,
                smooth=True,
                tags="stroke",
            )
            self._draw_line_on_buffer(self.last_x, self.last_y, x, y)
        self.last_x, self.last_y = x, y

        if self.auto_predict_var.get():
            self._schedule_auto_predict()

    def on_release(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None
        if self.auto_predict_var.get():
            self.predict()

    def _schedule_auto_predict(self) -> None:
        if self.auto_predict_job is not None:
            self.root.after_cancel(self.auto_predict_job)
        self.auto_predict_job = self.root.after(AUTO_PREDICT_DEBOUNCE_MS, self.predict)

    def _stamp(self, x: int, y: int) -> None:
        x = min(max(x, 0), CANVAS_SIZE - 1)
        y = min(max(y, 0), CANVAS_SIZE - 1)

        x_start = max(0, x - BRUSH_RADIUS)
        x_end = min(CANVAS_SIZE, x + BRUSH_RADIUS + 1)
        y_start = max(0, y - BRUSH_RADIUS)
        y_end = min(CANVAS_SIZE, y + BRUSH_RADIUS + 1)

        yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= BRUSH_RADIUS**2
        patch = self.buffer[y_start:y_end, x_start:x_end]
        patch[mask] = 1.0

    def _paint_cell_from_canvas(self, x: int, y: int) -> None:
        cell = DOWNSAMPLE_FACTOR
        cell_x = min(max(x // cell, 0), MODEL_SIZE - 1)
        cell_y = min(max(y // cell, 0), MODEL_SIZE - 1)

        px0 = cell_x * cell
        py0 = cell_y * cell
        px1 = px0 + cell
        py1 = py0 + cell

        self.buffer[py0:py1, px0:px1] = 1.0
        self.canvas.create_rectangle(
            px0,
            py0,
            px1,
            py1,
            fill="white",
            outline="",
            tags="stroke",
        )

    def _draw_cell_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        sampled_steps = max(1, steps // BUFFER_STROKE_STEP)
        xs = np.linspace(x0, x1, sampled_steps + 1)
        ys = np.linspace(y0, y1, sampled_steps + 1)
        for x, y in zip(xs, ys):
            self._paint_cell_from_canvas(int(x), int(y))

    def _draw_line_on_buffer(self, x0: int, y0: int, x1: int, y1: int) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        sampled_steps = max(1, steps // BUFFER_STROKE_STEP)
        xs = np.linspace(x0, x1, sampled_steps + 1)
        ys = np.linspace(y0, y1, sampled_steps + 1)
        for x, y in zip(xs, ys):
            self._stamp(int(x), int(y))

    def _prepare_input(self) -> np.ndarray:
        small = (
            self.buffer.reshape(
                MODEL_SIZE,
                DOWNSAMPLE_FACTOR,
                MODEL_SIZE,
                DOWNSAMPLE_FACTOR,
            ).mean(axis=(1, 3))
        ).astype(np.float32)
        processed = self._mnist_style_preprocess(small)
        self.last_processed = processed
        return processed.reshape(1, MODEL_SIZE * MODEL_SIZE)

    def _mnist_style_preprocess(self, img28: np.ndarray) -> np.ndarray:
        img = img28.copy().astype(np.float32)
        if float(img.max()) <= 1e-6:
            return img

        img /= float(img.max())
        mask = img > 0.05
        if not np.any(mask):
            return img

        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = img[y0:y1, x0:x1]

        h, w = crop.shape
        target = 20
        scale = min(target / max(h, 1), target / max(w, 1))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        yy = np.linspace(0, h - 1, new_h).astype(np.int32)
        xx = np.linspace(0, w - 1, new_w).astype(np.int32)
        resized = crop[yy][:, xx]

        canvas = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        py = (MODEL_SIZE - new_h) // 2
        px = (MODEL_SIZE - new_w) // 2
        canvas[py : py + new_h, px : px + new_w] = resized

        mass = float(canvas.sum())
        if mass > 1e-6:
            ys2, xs2 = np.indices(canvas.shape, dtype=np.float32)
            cy = float((ys2 * canvas).sum() / mass)
            cx = float((xs2 * canvas).sum() / mass)
            shift_y = int(round((MODEL_SIZE - 1) / 2 - cy))
            shift_x = int(round((MODEL_SIZE - 1) / 2 - cx))
            canvas = np.roll(canvas, shift=(shift_y, shift_x), axis=(0, 1))
            if shift_y > 0:
                canvas[:shift_y, :] = 0.0
            elif shift_y < 0:
                canvas[shift_y:, :] = 0.0
            if shift_x > 0:
                canvas[:, :shift_x] = 0.0
            elif shift_x < 0:
                canvas[:, shift_x:] = 0.0

        return np.clip(canvas, 0.0, 1.0)

    def predict(self) -> None:
        self.auto_predict_job = None
        x = self._prepare_input()
        probs = self.model.predict_proba(x)[0]
        top_idx = np.argsort(probs)[::-1][:3]

        best_digit = int(top_idx[0])
        best_prob = float(probs[best_digit])
        self.prediction_var.set(f"Prediction: {best_digit} ({best_prob * 100:.1f}%)")

        for rank, digit in enumerate(top_idx):
            p = float(probs[digit])
            self.top_labels[rank].configure(text=f"#{rank + 1}: {int(digit)} ({p * 100:.1f}%)")
            self.top_bars[rank]["value"] = p

        self._draw_probability_graph(probs)
        self._draw_processed_preview(self.last_processed)
        self._update_confidence_metrics(probs)
        self.status_var.set("Prediction updated.")

    def clear(self) -> None:
        self.canvas.delete("all")
        self.buffer.fill(0.0)
        self.prediction_var.set("Prediction: -")
        for rank in range(3):
            self.top_labels[rank].configure(text=f"#{rank + 1}: -")
            self.top_bars[rank]["value"] = 0.0
        self._draw_probability_graph(np.zeros(10, dtype=np.float32))
        self.last_processed = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        self.recent_confidences.clear()
        self.recent_margins.clear()
        self._draw_processed_preview(self.last_processed)
        self._draw_trend_graph()
        self.confidence_var.set("Top confidence: -")
        self.margin_var.set("Top-2 margin: -")
        self.certainty_var.set("Certainty score: -")
        self.confidence_bar["value"] = 0.0
        self.margin_bar["value"] = 0.0
        self.certainty_bar["value"] = 0.0
        self.status_var.set("Canvas cleared.")
        self._redraw_grid()


def run_applet(model: NeuralNetwork, debug: bool = False) -> None:
    try:
        app = DigitApplet(model, debug=debug)
        if debug:
            print("Entering Tk mainloop...", flush=True)
            print("If no window is visible: check other Spaces and Cmd+Tab.", flush=True)
        app.root.mainloop()
    except tk.TclError as exc:
        print(f"Tkinter failed to start GUI: {exc}", flush=True)
        print("Try: python3 -m tkinter", flush=True)
