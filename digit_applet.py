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


class DigitApplet:
    def __init__(self, model: NeuralNetwork, debug: bool = False) -> None:
        self.model = model
        self.debug = debug
        self.root = tk.Tk()
        self.root.title("Digit Predictor")
        self.root.resizable(False, False)

        self.buffer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.auto_predict_job: str | None = None

        self.auto_predict_var = tk.BooleanVar(value=False)
        self.prediction_var = tk.StringVar(value="Prediction: -")

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
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            outer,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        self.canvas.grid(row=0, column=0, rowspan=5, padx=(0, 12))

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        ttk.Label(outer, textvariable=self.prediction_var, font=("Helvetica", 16, "bold")).grid(
            row=0, column=1, sticky="w", pady=(0, 8)
        )

        self.top_labels: list[ttk.Label] = []
        self.top_bars: list[ttk.Progressbar] = []
        for idx in range(3):
            lbl = ttk.Label(outer, text=f"#{idx + 1}: -")
            bar = ttk.Progressbar(outer, orient="horizontal", length=180, maximum=1.0)
            lbl.grid(row=1 + idx, column=1, sticky="w")
            bar.grid(row=1 + idx, column=2, padx=(8, 0), pady=2, sticky="w")
            self.top_labels.append(lbl)
            self.top_bars.append(bar)

        controls = ttk.Frame(outer)
        controls.grid(row=4, column=1, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Button(controls, text="Predict", command=self.predict).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Clear", command=self.clear).grid(row=0, column=1, padx=(0, 8))
        ttk.Checkbutton(
            controls,
            text="Auto-predict",
            variable=self.auto_predict_var,
        ).grid(row=0, column=2, sticky="w")

    def on_press(self, event: tk.Event) -> None:
        self.last_x = int(event.x)
        self.last_y = int(event.y)
        self._stamp(self.last_x, self.last_y)
        if self.auto_predict_var.get():
            self._schedule_auto_predict()

    def on_drag(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            x,
            y,
            fill="white",
            width=BRUSH_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=True,
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
        return small.reshape(1, MODEL_SIZE * MODEL_SIZE)

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

    def clear(self) -> None:
        self.canvas.delete("all")
        self.buffer.fill(0.0)
        self.prediction_var.set("Prediction: -")
        for rank in range(3):
            self.top_labels[rank].configure(text=f"#{rank + 1}: -")
            self.top_bars[rank]["value"] = 0.0


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
