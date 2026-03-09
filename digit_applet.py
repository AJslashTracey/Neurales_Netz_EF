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
        self.show_grid_var = tk.BooleanVar(value=True)
        self.snap_to_grid_var = tk.BooleanVar(value=True)
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.status_var = tk.StringVar(value="Draw a digit, then click Predict.")

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

        controls = ttk.Frame(right)
        controls.grid(row=5, column=0, sticky="w", pady=(10, 0))

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

        ttk.Label(right, textvariable=self.status_var).grid(row=6, column=0, sticky="w", pady=(8, 0))

        self._redraw_grid()
        self._draw_probability_graph(np.zeros(10, dtype=np.float32))

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

        self._draw_probability_graph(probs)
        self.status_var.set("Prediction updated.")

    def clear(self) -> None:
        self.canvas.delete("all")
        self.buffer.fill(0.0)
        self.prediction_var.set("Prediction: -")
        for rank in range(3):
            self.top_labels[rank].configure(text=f"#{rank + 1}: -")
            self.top_bars[rank]["value"] = 0.0
        self._draw_probability_graph(np.zeros(10, dtype=np.float32))
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
