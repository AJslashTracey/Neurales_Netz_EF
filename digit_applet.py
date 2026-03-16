import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

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
QUALITY_WIDTH = 360
LEARNING_HEIGHT = 150
CONFUSION_HEIGHT = 250
SIGNAL_HEIGHT = 170
SPARKLINE_HEIGHT = 66
PREVIEW_SCALE = 4
HISTORY_LENGTH = 40
METRICS_SUFFIX = ".metrics.npz"


class DigitApplet:
    def __init__(
        self,
        model: NeuralNetwork | None,
        debug: bool = False,
        training_dashboard: dict[str, np.ndarray] | None = None,
        model_path: str | None = None,
    ) -> None:
        self.model = model
        self.debug = debug
        self.model_path = model_path
        self.training_dashboard = training_dashboard or {}
        self.root = tk.Tk()
        self.model_label_var = tk.StringVar(value="Model: unknown")
        self.root.title("Digit Predictor")
        self.root.resizable(False, False)

        self.buffer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.auto_predict_job: str | None = None
        self.training_in_progress = False

        self.auto_predict_var = tk.BooleanVar(value=False)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.snap_to_grid_var = tk.BooleanVar(value=True)
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.status_var = tk.StringVar(value="Draw a digit, then click Predict.")
        self.confidence_var = tk.StringVar(value="Top confidence: -")
        self.margin_var = tk.StringVar(value="Top-2 margin: -")
        self.certainty_var = tk.StringVar(value="Certainty score: -")
        self.quality_summary_var = tk.StringVar(value="Model quality vs dataset quality: unavailable")
        self.ceiling_progress_var = tk.StringVar(value="Benchmark progress: unavailable")
        self.dataset_quality_var = tk.StringVar(value="Dataset quality: unavailable")
        self.limits_var = tk.StringVar(value="Likely bottlenecks: dataset metrics unavailable")
        if self.model_path:
            self.model_label_var.set(f"Model: {os.path.basename(self.model_path)}")

        self.last_processed = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        self.recent_confidences: list[float] = []
        self.recent_margins: list[float] = []
        self._refresh_dashboard_metrics()

        self._build_ui()
        # Bring the window to front on launch.
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self.root.geometry("+220+120")
        if self.model is None:
            self.status_var.set("No model loaded. Click 'Load model...' to begin.")
            self.model_label_var.set("Model: not loaded")
            self.root.after(150, self._choose_and_load_model)

        if self.debug:
            print("Tk window created and focused.", flush=True)

    def _metric_series(self, key: str, size: int | None = None) -> np.ndarray:
        value = self.training_dashboard.get(key)
        if value is None:
            return np.zeros((0,), dtype=np.float32) if size is None else np.zeros((size,), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if size is not None:
            if arr.size < size:
                out = np.zeros((size,), dtype=np.float32)
                out[: arr.size] = arr
                return out
            return arr[:size]
        return arr

    def _metric_matrix_series(self, key: str, rows: int, cols: int) -> np.ndarray:
        value = self.training_dashboard.get(key)
        if value is None:
            return np.zeros((0, rows, cols), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (rows, cols):
            return arr.reshape(1, rows, cols)
        if arr.ndim == 3 and arr.shape[1:] == (rows, cols):
            return arr
        if arr.ndim == 2 and rows == 1 and arr.shape[1] == cols:
            return arr.reshape(arr.shape[0], 1, cols)
        return np.zeros((0, rows, cols), dtype=np.float32)

    def _metric_scalar(self, key: str, default: float) -> float:
        value = self.training_dashboard.get(key)
        if value is None:
            return default
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[-1])

    def _refresh_dashboard_metrics(self) -> None:
        self.train_loss_hist = self._metric_series("train_loss")
        self.val_loss_hist = self._metric_series("val_loss")
        self.train_acc_hist = self._metric_series("train_acc")
        self.val_acc_hist = self._metric_series("val_acc")
        self.test_acc_hist = self._metric_series("test_acc")
        self.confusion_hist = self._metric_matrix_series("test_confusion_norm", rows=10, cols=10)
        self.per_class_acc_hist = self._metric_matrix_series("test_per_class_acc", rows=1, cols=10)
        self.train_class_counts = self._metric_series("train_class_counts", size=10)
        self.test_class_counts = self._metric_series("test_class_counts", size=10)
        self.label_noise_estimate = self._metric_scalar("label_noise_estimate", default=0.02)
        self.dataset_ceiling = self._metric_scalar(
            "dataset_ceiling", default=max(0.0, 1.0 - self.label_noise_estimate)
        )
        self.augmentation_enabled = self._metric_scalar("augmentation_flag", default=1.0) >= 0.5
        self.architecture_depth = int(round(self._metric_scalar("architecture_depth", default=0.0)))
        self.parameter_count = int(round(self._metric_scalar("parameter_count", default=0.0)))

    def _load_metrics_for_model(self, model_path: str) -> dict[str, np.ndarray]:
        metrics_path = f"{model_path}{METRICS_SUFFIX}"
        if not os.path.exists(metrics_path):
            return {}
        data = np.load(metrics_path)
        return {key: data[key] for key in data.files}

    def _to_numpy_payload(
        self, metrics: dict[str, list[float] | list[np.ndarray] | np.ndarray | float]
    ) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.astype(np.float32)
                continue
            if isinstance(value, list):
                if not value:
                    payload[key] = np.array([], dtype=np.float32)
                elif isinstance(value[0], np.ndarray):
                    payload[key] = np.stack(value).astype(np.float32)
                else:
                    payload[key] = np.array(value, dtype=np.float32)
                continue
            payload[key] = np.array([value], dtype=np.float32)
        return payload

    def _save_metrics(
        self, path: str, metrics: dict[str, list[float] | list[np.ndarray] | np.ndarray | float]
    ) -> None:
        np.savez(path, **self._to_numpy_payload(metrics))

    def _refresh_optional_training_views(self) -> None:
        if hasattr(self, "learning_canvas"):
            self._draw_learning_curves()
        if hasattr(self, "training_sparkline_canvas"):
            self._draw_training_sparkline()
        if hasattr(self, "confusion_canvas"):
            self._draw_confusion_matrix()
        if hasattr(self, "signal_canvas"):
            self._draw_signal_noise_chart()
        if hasattr(self, "compare_canvas"):
            self._draw_benchmark_panel()

    def _choose_and_load_model(self) -> None:
        initial_dir = "."
        if self.model_path:
            initial_dir = os.path.dirname(self.model_path) or "."
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select model file",
            initialdir=initial_dir,
            filetypes=[
                ("NumPy model files", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self._load_model_from_path(selected)

    def _load_model_from_path(self, selected_path: str) -> None:
        model_path = selected_path
        if selected_path.endswith(METRICS_SUFFIX):
            candidate = selected_path[: -len(METRICS_SUFFIX)]
            if os.path.exists(candidate):
                model_path = candidate
            else:
                self.status_var.set("Selected metrics file has no matching model file.")
                return

        try:
            self.model = NeuralNetwork.load_model(model_path)
        except Exception as exc:
            self.status_var.set(f"Failed to load model: {exc}")
            return

        self.model_path = model_path
        self.model_label_var.set(f"Model: {os.path.basename(model_path)}")
        self.training_dashboard = self._load_metrics_for_model(model_path)
        self._refresh_dashboard_metrics()
        self._refresh_optional_training_views()
        self.status_var.set(f"Loaded model: {os.path.basename(model_path)}")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, rowspan=8, padx=(0, 14), sticky="n")

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nw")

        ttk.Label(left, text="Drawing Grid (28x28)", font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#888888",
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
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        ttk.Label(right, textvariable=self.prediction_var, font=("Helvetica", 16, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.top_labels: list[ttk.Label] = []
        self.top_bars: list[ttk.Progressbar] = []
        for _ in range(3):
            self.top_labels.append(ttk.Label(right, text="-"))
            self.top_bars.append(ttk.Progressbar(right, orient="horizontal", length=210, maximum=1.0))

        ttk.Label(
            right, text="Top predictions", font=("Helvetica", 11, "bold")
        ).grid(row=1, column=0, sticky="w", pady=(2, 4))

        top3_frame = ttk.Frame(right)
        top3_frame.grid(row=2, column=0, sticky="w")
        for idx in range(3):
            lbl = ttk.Label(top3_frame, text=f"#{idx + 1}: -", width=12)
            bar = ttk.Progressbar(top3_frame, orient="horizontal", length=210, maximum=1.0)
            lbl.grid(row=idx, column=0, sticky="w", pady=3)
            bar.grid(row=idx, column=1, padx=(8, 0), pady=3, sticky="w")
            self.top_labels[idx] = lbl
            self.top_bars[idx] = bar

        ttk.Label(
            right, text="Probability graph (0-9)", font=("Helvetica", 11, "bold")
        ).grid(row=3, column=0, sticky="w", pady=(10, 4))
        self.graph_canvas = tk.Canvas(
            right,
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            bg="#f3f3f3",
            highlightthickness=1,
            highlightbackground="#c9c9c9",
        )
        self.graph_canvas.grid(row=4, column=0, sticky="w")

        ttk.Label(
            right, text="Input + confidence diagnostics", font=("Helvetica", 11, "bold")
        ).grid(row=5, column=0, sticky="w", pady=(10, 4))

        diagnostics = ttk.Frame(right)
        diagnostics.grid(row=6, column=0, sticky="w")

        left_diag = ttk.Frame(diagnostics)
        left_diag.grid(row=0, column=0, sticky="nw", padx=(0, 14))
        right_diag = ttk.Frame(diagnostics)
        right_diag.grid(row=0, column=1, sticky="nw")

        ttk.Label(left_diag, text="Processed 28x28 preview").grid(row=0, column=0, sticky="w")
        self.preview_canvas = tk.Canvas(
            left_diag,
            width=MODEL_SIZE * PREVIEW_SCALE,
            height=MODEL_SIZE * PREVIEW_SCALE,
            bg="#101010",
            highlightthickness=1,
            highlightbackground="#c9c9c9",
        )
        self.preview_canvas.grid(row=1, column=0, pady=(4, 0))

        ttk.Label(right_diag, textvariable=self.confidence_var).grid(row=0, column=0, sticky="w")
        self.confidence_bar = ttk.Progressbar(right_diag, orient="horizontal", length=190, maximum=1.0)
        self.confidence_bar.grid(row=1, column=0, sticky="w", pady=(2, 6))

        ttk.Label(right_diag, textvariable=self.margin_var).grid(row=2, column=0, sticky="w")
        self.margin_bar = ttk.Progressbar(right_diag, orient="horizontal", length=190, maximum=1.0)
        self.margin_bar.grid(row=3, column=0, sticky="w", pady=(2, 6))

        ttk.Label(right_diag, textvariable=self.certainty_var).grid(row=4, column=0, sticky="w")
        self.certainty_bar = ttk.Progressbar(right_diag, orient="horizontal", length=190, maximum=1.0)
        self.certainty_bar.grid(row=5, column=0, sticky="w", pady=(2, 0))

        ttk.Label(
            right, text="Confidence trend (recent predictions)", font=("Helvetica", 10, "bold")
        ).grid(row=7, column=0, sticky="w", pady=(10, 3))
        self.trend_canvas = tk.Canvas(
            right,
            width=TREND_WIDTH,
            height=TREND_HEIGHT,
            bg="#f3f3f3",
            highlightthickness=1,
            highlightbackground="#c9c9c9",
        )
        self.trend_canvas.grid(row=8, column=0, sticky="w")

        controls = ttk.Frame(right)
        controls.grid(row=9, column=0, sticky="w", pady=(10, 0))

        ttk.Button(controls, text="Predict", command=self.predict).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Clear", command=self.clear).grid(row=0, column=1, padx=(0, 8))
        ttk.Checkbutton(
            controls,
            text="Auto-predict",
            variable=self.auto_predict_var,
        ).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Snap to cells",
            variable=self.snap_to_grid_var,
        ).grid(row=0, column=3, padx=(12, 0), sticky="w")
        ttk.Button(controls, text="Load model...", command=self._choose_and_load_model).grid(
            row=0, column=4, padx=(12, 0)
        )
        self.train_button = ttk.Button(controls, text="Train model...", command=self._open_training_dialog)
        self.train_button.grid(row=0, column=5, padx=(8, 0))

        ttk.Label(right, textvariable=self.status_var).grid(row=10, column=0, sticky="w", pady=(8, 0))
        ttk.Label(right, textvariable=self.model_label_var).grid(row=11, column=0, sticky="w", pady=(4, 0))

        self._redraw_grid()
        self._draw_probability_graph(np.zeros(10, dtype=np.float32))
        self._draw_processed_preview(self.last_processed)
        self._draw_trend_graph()

    def _redraw_grid(self) -> None:
        self.canvas.delete("grid")
        if not self.show_grid_var.get():
            return

        step = DOWNSAMPLE_FACTOR
        color = "#1f1f1f"
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
            fill="#888888",
        )

        for digit in range(10):
            p = float(probs[digit]) if probs.size else 0.0
            x0 = margin_left + digit * bar_width + 3
            x1 = margin_left + (digit + 1) * bar_width - 3
            y1 = GRAPH_HEIGHT - margin_bottom
            y0 = y1 - p * max_h
            self.graph_canvas.create_rectangle(x0, y0, x1, y1, fill="#1f8cff", outline="")
            self.graph_canvas.create_text(
                (x0 + x1) / 2,
                GRAPH_HEIGHT - 8,
                text=str(digit),
                fill="#333333",
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

        self.trend_canvas.create_line(left, bottom, right, bottom, fill="#9a9a9a")
        self.trend_canvas.create_line(left, top, left, bottom, fill="#9a9a9a")

        if not self.recent_confidences:
            self.trend_canvas.create_text(
                w / 2,
                h / 2,
                text="No predictions yet",
                fill="#777777",
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
            self.trend_canvas.create_line(*margin_pts, fill="#9cc8ff", width=2, smooth=True)
            self.trend_canvas.create_line(*conf_pts, fill="#1f8cff", width=2, smooth=True)

        self.trend_canvas.create_text(
            right - 42,
            top + 8,
            text="conf",
            fill="#1f8cff",
            font=("Helvetica", 9, "bold"),
        )
        self.trend_canvas.create_text(
            right - 42,
            top + 22,
            text="margin",
            fill="#6faef6",
            font=("Helvetica", 9),
        )

    def _draw_learning_curves(self) -> None:
        c = self.learning_canvas
        c.delete("all")
        w = QUALITY_WIDTH
        h = LEARNING_HEIGHT
        left, right = 28, w - 10
        top, bottom = 12, h - 24
        c.create_line(left, bottom, right, bottom, fill="#9a9a9a")
        c.create_line(left, top, left, bottom, fill="#9a9a9a")
        c.create_text(7, top + 4, text="1.0", anchor="w", fill="#777777", font=("Helvetica", 8))
        c.create_text(7, bottom - 2, text="0.0", anchor="w", fill="#777777", font=("Helvetica", 8))

        n = int(
            max(
                self.train_loss_hist.size,
                self.val_loss_hist.size,
                self.train_acc_hist.size,
                self.val_acc_hist.size,
            )
        )
        if n < 2:
            c.create_text(
                w / 2,
                h / 2,
                text="No training history found (.metrics.npz missing).",
                fill="#777777",
                font=("Helvetica", 10),
            )
            return

        max_loss = float(
            max(
                np.max(self.train_loss_hist) if self.train_loss_hist.size else 1.0,
                np.max(self.val_loss_hist) if self.val_loss_hist.size else 1.0,
                1e-5,
            )
        )

        def _line(values: np.ndarray, color: str, normalize_by: float = 1.0) -> None:
            if values.size < 2:
                return
            points: list[float] = []
            for i, val in enumerate(values):
                x = left + (i / max(values.size - 1, 1)) * (right - left)
                y_norm = float(np.clip(val / normalize_by, 0.0, 1.0))
                y = bottom - y_norm * (bottom - top)
                points.extend([x, y])
            c.create_line(*points, fill=color, width=2, smooth=True)

        _line(self.train_loss_hist, "#f39a3f", normalize_by=max_loss)
        _line(self.val_loss_hist, "#d45e2e", normalize_by=max_loss)
        _line(self.train_acc_hist, "#51b45a")
        _line(self.val_acc_hist, "#2a8f3c")

        ceiling_y = bottom - float(np.clip(self.dataset_ceiling, 0.0, 1.0)) * (bottom - top)
        c.create_line(left, ceiling_y, right, ceiling_y, fill="#6e66d9", width=1, dash=(4, 3))

        legend = (
            "loss(train/val): orange/red   "
            "acc(train/val): light/dark green   "
            "ceiling: purple dashed"
        )
        c.create_text(left, h - 8, anchor="w", text=legend, fill="#585858", font=("Helvetica", 8))

    def _draw_training_sparkline(self) -> None:
        c = self.training_sparkline_canvas
        c.delete("all")
        w = QUALITY_WIDTH
        h = SPARKLINE_HEIGHT
        left, right = 10, w - 10
        top, bottom = 8, h - 14
        c.create_line(left, bottom, right, bottom, fill="#9a9a9a")
        if self.test_acc_hist.size < 2:
            c.create_text(w / 2, h / 2, text="No test trend yet", fill="#777777", font=("Helvetica", 9))
            return

        pts: list[float] = []
        for i, val in enumerate(self.test_acc_hist):
            x = left + (i / max(self.test_acc_hist.size - 1, 1)) * (right - left)
            y = bottom - float(np.clip(val, 0.0, 1.0)) * (bottom - top)
            pts.extend([x, y])
        c.create_line(*pts, fill="#1f8cff", width=2, smooth=True)
        ceiling_y = bottom - float(np.clip(self.dataset_ceiling, 0.0, 1.0)) * (bottom - top)
        c.create_line(left, ceiling_y, right, ceiling_y, fill="#6e66d9", width=1, dash=(3, 3))

        current_test = float(self.test_acc_hist[-1])
        self.quality_summary_var.set(
            f"Current test acc {current_test * 100:.1f}% / Estimated dataset ceiling "
            f"~{self.dataset_ceiling * 100:.1f}% (label noise ~{self.label_noise_estimate * 100:.1f}%)"
        )

    def _draw_confusion_matrix(self) -> None:
        c = self.confusion_canvas
        c.delete("all")
        w = QUALITY_WIDTH
        h = CONFUSION_HEIGHT
        c.create_text(10, 10, anchor="nw", text="Normalized confusion matrix (test split)", font=("Helvetica", 10, "bold"))
        if self.confusion_hist.shape[0] == 0:
            c.create_text(w / 2, h / 2, text="No evaluation confusion matrix available", fill="#777777")
            return

        matrix = self.confusion_hist[-1]
        grid_left = 34
        grid_top = 30
        cell = 24
        for i in range(10):
            c.create_text(grid_left - 14, grid_top + i * cell + cell / 2, text=str(i), fill="#555555", font=("Helvetica", 8))
            c.create_text(grid_left + i * cell + cell / 2, grid_top - 12, text=str(i), fill="#555555", font=("Helvetica", 8))

        for row in range(10):
            for col in range(10):
                val = float(np.clip(matrix[row, col], 0.0, 1.0))
                shade = int(245 - val * 180)
                color = f"#{shade:02x}{shade:02x}ff"
                x0 = grid_left + col * cell
                y0 = grid_top + row * cell
                c.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="#d4d4d4")
                if val >= 0.08:
                    c.create_text(
                        x0 + cell / 2,
                        y0 + cell / 2,
                        text=f"{val:.2f}",
                        fill="#1e1e1e",
                        font=("Helvetica", 7),
                    )

        total_counts = self.train_class_counts + self.test_class_counts
        count_min = int(np.min(total_counts)) if total_counts.size else 0
        count_max = int(np.max(total_counts)) if total_counts.size else 0
        balance_ratio = float(count_min / max(count_max, 1))
        class_text = ", ".join(f"{i}:{int(v)}" for i, v in enumerate(total_counts.astype(int)))
        self.dataset_quality_var.set(
            f"Dataset quality: class balance ratio={balance_ratio:.3f}, "
            f"label noise~{self.label_noise_estimate * 100:.1f}%\nClass frequencies: {class_text}"
        )

    def _draw_signal_noise_chart(self) -> None:
        c = self.signal_canvas
        c.delete("all")
        w = QUALITY_WIDTH
        h = SIGNAL_HEIGHT
        left, right = 24, w - 20
        top, bottom = 14, h - 24
        c.create_line(left, bottom, right, bottom, fill="#9a9a9a")
        c.create_line(left, top, left, bottom, fill="#9a9a9a")
        c.create_text(4, top, text="acc", anchor="nw", fill="#4b8f35", font=("Helvetica", 8))
        c.create_text(right - 4, top, text="count", anchor="ne", fill="#4369d7", font=("Helvetica", 8))

        if self.per_class_acc_hist.shape[0] == 0:
            c.create_text(w / 2, h / 2, text="No per-class evaluation history", fill="#777777")
            return

        acc = self.per_class_acc_hist[-1, 0]
        counts = self.train_class_counts + self.test_class_counts
        max_count = float(np.max(counts)) if counts.size else 1.0
        bar_w = (right - left) / 10.0
        line_pts: list[float] = []
        for digit in range(10):
            x0 = left + digit * bar_w + 2
            x1 = left + (digit + 1) * bar_w - 2
            y_acc = bottom - float(np.clip(acc[digit], 0.0, 1.0)) * (bottom - top)
            c.create_rectangle(x0, y_acc, x1, bottom, fill="#58b75f", outline="")
            count_norm = float(counts[digit] / max(max_count, 1.0)) if counts.size else 0.0
            y_count = bottom - count_norm * (bottom - top)
            line_pts.extend([(x0 + x1) / 2, y_count])
            c.create_text((x0 + x1) / 2, bottom + 9, text=str(digit), fill="#444444", font=("Helvetica", 8))
        if len(line_pts) >= 4:
            c.create_line(*line_pts, fill="#406bd8", width=2, smooth=True)
        c.create_text(left + 4, top + 8, text="bars: class accuracy", anchor="nw", fill="#4b8f35", font=("Helvetica", 8))
        c.create_text(left + 4, top + 20, text="line: sample coverage", anchor="nw", fill="#406bd8", font=("Helvetica", 8))

    def _draw_benchmark_panel(self) -> None:
        self.compare_canvas.delete("all")
        current_test = float(self.test_acc_hist[-1]) if self.test_acc_hist.size else 0.0
        gap = max(0.0, self.dataset_ceiling - current_test)
        self.ceiling_progress_var.set(
            f"Benchmark progress: model {current_test * 100:.1f}% vs best-case "
            f"{self.dataset_ceiling * 100:.1f}% (gap {gap * 100:.1f} pts)"
        )
        factors = [
            f"dataset size={int(np.sum(self.train_class_counts + self.test_class_counts))}",
            f"augmentation={'on' if self.augmentation_enabled else 'off'}",
            f"architecture depth={self.architecture_depth} layers",
        ]
        if self.parameter_count > 0:
            factors.append(f"params~{self.parameter_count}")
        self.limits_var.set("Likely bottlenecks: " + " | ".join(factors))

        c = self.compare_canvas
        w = QUALITY_WIDTH
        h = 70
        left = 90
        right = w - 12
        y1 = 20
        y2 = 48
        c.create_text(10, y1, text="Model", anchor="w", font=("Helvetica", 9, "bold"))
        c.create_text(10, y2, text="Dataset ceiling", anchor="w", font=("Helvetica", 9, "bold"))

        model_x = left + current_test * (right - left)
        ceil_x = left + self.dataset_ceiling * (right - left)
        c.create_rectangle(left, y1 - 7, right, y1 + 7, fill="#ececec", outline="")
        c.create_rectangle(left, y2 - 7, right, y2 + 7, fill="#ececec", outline="")
        c.create_rectangle(left, y1 - 7, model_x, y1 + 7, fill="#1f8cff", outline="", tags=("model_bar",))
        c.create_rectangle(left, y2 - 7, ceil_x, y2 + 7, fill="#6e66d9", outline="", tags=("ceiling_bar",))
        c.create_text(right, y1, text=f"{current_test * 100:.1f}%", anchor="e", fill="#1f8cff", font=("Helvetica", 9, "bold"))
        c.create_text(
            right,
            y2,
            text=f"{self.dataset_ceiling * 100:.1f}%",
            anchor="e",
            fill="#6e66d9",
            font=("Helvetica", 9, "bold"),
        )

        model_tip = (
            f"Model: current test accuracy {current_test * 100:.1f}%.\n"
            f"Coverage {int(np.sum(self.train_class_counts + self.test_class_counts))} samples across 10 classes."
        )
        ceiling_tip = (
            f"Dataset ceiling assumes 1 - label_noise.\n"
            f"label noise estimate ~{self.label_noise_estimate * 100:.1f}% -> ceiling {self.dataset_ceiling * 100:.1f}%."
        )
        c.tag_bind("model_bar", "<Enter>", lambda _e: self.status_var.set(model_tip))
        c.tag_bind("ceiling_bar", "<Enter>", lambda _e: self.status_var.set(ceiling_tip))
        c.tag_bind("model_bar", "<Leave>", lambda _e: self.status_var.set("Prediction updated."))
        c.tag_bind("ceiling_bar", "<Leave>", lambda _e: self.status_var.set("Prediction updated."))

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
        if self.model is None:
            self.status_var.set("No model loaded. Click 'Load model...' first.")
            return
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

    def _open_training_dialog(self) -> None:
        if self.training_in_progress:
            self.status_var.set("Training already running. Please wait for completion.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Train model")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        content = ttk.Frame(dialog, padding=10)
        content.grid(row=0, column=0, sticky="nsew")

        default_model_path = self.model_path or "model.npz"
        train_dir_var = tk.StringVar(value="data/Reduced_MNIST_Data/Reduced_Trainging_data")
        test_dir_var = tk.StringVar(value="data/Reduced_MNIST_Data/Reduced_Testing_data")
        model_path_var = tk.StringVar(value=default_model_path)
        epochs_var = tk.StringVar(value="80")
        batch_size_var = tk.StringVar(value="64")
        learning_rate_var = tk.StringVar(value="0.005")
        hidden_dims_var = tk.StringVar(value="256,128,64")
        weight_decay_var = tk.StringVar(value="0.0001")
        val_split_var = tk.StringVar(value="0.1")
        lr_decay_step_var = tk.StringVar(value="20")
        lr_decay_factor_var = tk.StringVar(value="0.5")
        patience_var = tk.StringVar(value="12")
        augment_var = tk.BooleanVar(value=True)

        if self.model is not None and self.model.weights:
            inferred_hidden = ",".join(str(w.shape[1]) for w in self.model.weights[:-1])
            if inferred_hidden:
                hidden_dims_var.set(inferred_hidden)

        def _browse_train_dir() -> None:
            selected = filedialog.askdirectory(parent=dialog, title="Select training data folder")
            if selected:
                train_dir_var.set(selected)

        def _browse_test_dir() -> None:
            selected = filedialog.askdirectory(parent=dialog, title="Select testing data folder")
            if selected:
                test_dir_var.set(selected)

        def _browse_model_file() -> None:
            selected = filedialog.asksaveasfilename(
                parent=dialog,
                title="Choose model output file",
                defaultextension=".npz",
                filetypes=[("NumPy model files", "*.npz"), ("All files", "*.*")],
                initialfile=os.path.basename(model_path_var.get()) or "model.npz",
            )
            if selected:
                model_path_var.set(selected)

        row = 0
        ttk.Label(content, text="Train data folder").grid(row=row, column=0, sticky="w")
        ttk.Entry(content, textvariable=train_dir_var, width=46).grid(row=row, column=1, padx=(6, 6))
        ttk.Button(content, text="Browse...", command=_browse_train_dir).grid(row=row, column=2)
        row += 1

        ttk.Label(content, text="Test data folder").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(content, textvariable=test_dir_var, width=46).grid(
            row=row, column=1, padx=(6, 6), pady=(6, 0)
        )
        ttk.Button(content, text="Browse...", command=_browse_test_dir).grid(row=row, column=2, pady=(6, 0))
        row += 1

        ttk.Label(content, text="Model output path").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(content, textvariable=model_path_var, width=46).grid(
            row=row, column=1, padx=(6, 6), pady=(6, 0)
        )
        ttk.Button(content, text="Browse...", command=_browse_model_file).grid(row=row, column=2, pady=(6, 0))
        row += 1

        ttk.Label(content, text="Epochs").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(content, textvariable=epochs_var, width=14).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        row += 1

        ttk.Label(content, text="Batch size").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=batch_size_var, width=14).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0))
        row += 1

        ttk.Label(content, text="Learning rate").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=learning_rate_var, width=14).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Label(content, text="Hidden dims (csv)").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=hidden_dims_var, width=24).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Label(content, text="Weight decay").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=weight_decay_var, width=14).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Label(content, text="Validation split").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=val_split_var, width=14).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0))
        row += 1

        ttk.Label(content, text="LR decay step").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=lr_decay_step_var, width=14).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Label(content, text="LR decay factor").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=lr_decay_factor_var, width=14).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Label(content, text="Early stopping patience").grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(content, textvariable=patience_var, width=14).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        row += 1

        ttk.Checkbutton(content, text="Use data augmentation", variable=augment_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        row += 1

        button_row = ttk.Frame(content)
        button_row.grid(row=row, column=0, columnspan=3, sticky="e", pady=(10, 0))

        def _start_training() -> None:
            try:
                hidden_dims = tuple(
                    int(x.strip()) for x in hidden_dims_var.get().split(",") if x.strip()
                )
                if not hidden_dims:
                    raise ValueError("Hidden dims must contain at least one layer size.")

                train_params = {
                    "train_dir": train_dir_var.get().strip(),
                    "test_dir": test_dir_var.get().strip(),
                    "model_path": model_path_var.get().strip(),
                    "epochs": int(epochs_var.get()),
                    "batch_size": int(batch_size_var.get()),
                    "learning_rate": float(learning_rate_var.get()),
                    "hidden_dims": hidden_dims,
                    "weight_decay": float(weight_decay_var.get()),
                    "val_split": float(val_split_var.get()),
                    "lr_decay_step": int(lr_decay_step_var.get()),
                    "lr_decay_factor": float(lr_decay_factor_var.get()),
                    "patience": int(patience_var.get()),
                    "use_augmentation": bool(augment_var.get()),
                }
            except Exception as exc:
                self.status_var.set(f"Invalid training settings: {exc}")
                return

            if not train_params["train_dir"] or not train_params["test_dir"]:
                self.status_var.set("Training/test data paths are required.")
                return
            if not train_params["model_path"]:
                self.status_var.set("Model output path is required.")
                return
            if self.training_in_progress:
                self.status_var.set("Training already in progress.")
                return

            self.training_in_progress = True
            self.train_button.state(["disabled"])
            self.status_var.set("Training started... progress is printed in terminal.")
            dialog.destroy()
            thread = threading.Thread(
                target=self._train_in_background,
                args=(train_params,),
                daemon=True,
            )
            thread.start()

        ttk.Button(button_row, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Start training", command=_start_training).grid(row=0, column=1)

    def _train_in_background(self, params: dict[str, object]) -> None:
        try:
            from data_loader import load_data
            from train import train_model

            X_train, y_train, X_test, y_test = load_data(
                train_dir=str(params["train_dir"]),
                test_dir=str(params["test_dir"]),
            )

            model = NeuralNetwork(
                input_dim=X_train.shape[1],
                hidden_dims=tuple(int(v) for v in params["hidden_dims"]),  # type: ignore[arg-type]
                output_dim=y_train.shape[1],
                learning_rate=float(params["learning_rate"]),
                weight_decay=float(params["weight_decay"]),
                seed=42,
            )

            metrics = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                epochs=int(params["epochs"]),
                batch_size=int(params["batch_size"]),
                val_split=float(params["val_split"]),
                use_augmentation=bool(params["use_augmentation"]),
                lr_decay_step=int(params["lr_decay_step"]),
                lr_decay_factor=float(params["lr_decay_factor"]),
                early_stopping_patience=int(params["patience"]),
            )

            model_path = str(params["model_path"])
            model.save_model(model_path)
            metrics_path = f"{model_path}{METRICS_SUFFIX}"
            self._save_metrics(metrics_path, metrics)
            dashboard = self._load_metrics_for_model(model_path)
            self.root.after(0, lambda: self._on_training_success(model, model_path, dashboard))
        except Exception as exc:
            self.root.after(0, lambda: self._on_training_failure(exc))

    def _on_training_success(
        self,
        model: NeuralNetwork,
        model_path: str,
        dashboard: dict[str, np.ndarray],
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.training_dashboard = dashboard
        self._refresh_dashboard_metrics()
        self._refresh_optional_training_views()
        self.model_label_var.set(f"Model: {os.path.basename(model_path)}")
        self.status_var.set(
            f"Training finished. Model saved to {os.path.basename(model_path)} and loaded."
        )
        self.training_in_progress = False
        self.train_button.state(["!disabled"])

    def _on_training_failure(self, exc: Exception) -> None:
        self.status_var.set(f"Training failed: {exc}")
        self.training_in_progress = False
        self.train_button.state(["!disabled"])

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


def run_applet(
    model: NeuralNetwork | None,
    debug: bool = False,
    training_dashboard: dict[str, np.ndarray] | None = None,
    model_path: str | None = None,
) -> None:
    try:
        app = DigitApplet(
            model,
            debug=debug,
            training_dashboard=training_dashboard,
            model_path=model_path,
        )
        if debug:
            print("Entering Tk mainloop...", flush=True)
            print("If no window is visible: check other Spaces and Cmd+Tab.", flush=True)
        app.root.mainloop()
    except tk.TclError as exc:
        print(f"Tkinter failed to start GUI: {exc}", flush=True)
        print("Try: python3 -m tkinter", flush=True)
