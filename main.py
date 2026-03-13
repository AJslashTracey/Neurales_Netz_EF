import argparse
import os

import numpy as np

from network import NeuralNetwork

METRICS_SUFFIX = ".metrics.npz"


def _to_numpy_payload(
    metrics: dict[str, list[float] | list[np.ndarray] | np.ndarray | float]
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


def _save_metrics(path: str, metrics: dict[str, list[float] | list[np.ndarray] | np.ndarray | float]) -> None:
    np.savez(path, **_to_numpy_payload(metrics))


def _load_metrics(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _discover_model_path(preferred_path: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_dirs = [".", "models"]
    discovered: list[str] = []
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for name in sorted(os.listdir(base_dir)):
            if not name.endswith(".npz") or name.endswith(METRICS_SUFFIX):
                continue
            path = os.path.join(base_dir, name)
            if os.path.isfile(path):
                discovered.append(path)

    if not discovered:
        return None

    for candidate in discovered:
        if os.path.basename(candidate) == "model.npz":
            return candidate
    return discovered[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run a digit classifier applet.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--app", action="store_true", help="Launch the drawing applet.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--model-path", type=str, default="model.npz", help="Model file path.")
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/Reduced_MNIST_Data/Reduced_Trainging_data",
        help="Path to the folder holding training digit JPGs.",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="data/Reduced_MNIST_Data/Reduced_Testing_data",
        help="Path to the folder holding validation digit JPGs.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden layer sizes, e.g. 256,128,64",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--lr-decay-step",
        type=int,
        default=20,
        help="Decay learning rate every N epochs (0 to disable).",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="Factor applied at each LR decay step.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation during training.",
    )
    parser.add_argument(
        "--debug-app",
        action="store_true",
        help="Print detailed app startup logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    should_train = args.train or (not args.train and not args.app)
    model: NeuralNetwork | None = None
    training_dashboard: dict[str, np.ndarray] = {}
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(",") if x.strip())

    if should_train:
        from data_loader import load_data
        from train import train_model

        X_train, y_train, X_test, y_test = load_data(
            train_dir=args.train_data_path,
            test_dir=args.test_data_path,
        )
        print(
            f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )

        model = NeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            output_dim=y_train.shape[1],
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=42,
        )

        metrics = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            use_augmentation=not args.no_augment,
            lr_decay_step=args.lr_decay_step,
            lr_decay_factor=args.lr_decay_factor,
            early_stopping_patience=args.patience,
        )
        model.save_model(args.model_path)
        print(f"Saved model to: {args.model_path}")
        metrics_path = f"{args.model_path}{METRICS_SUFFIX}"
        _save_metrics(metrics_path, metrics)
        training_dashboard = _load_metrics(metrics_path)
        print(f"Saved training metrics to: {metrics_path}")

    if args.app:
        print("Starting digit applet...", flush=True)
        app_model_path: str | None = args.model_path
        if model is None:
            resolved_model_path = _discover_model_path(args.model_path)
            app_model_path = resolved_model_path
            if resolved_model_path is None:
                print("No startup model found. App will open and let you choose one.", flush=True)
            else:
                if args.debug_app:
                    print(f"Loading model from: {resolved_model_path}", flush=True)
                model = NeuralNetwork.load_model(resolved_model_path)
                training_dashboard = _load_metrics(f"{resolved_model_path}{METRICS_SUFFIX}")
                if args.debug_app:
                    print("Model loaded successfully.", flush=True)
        try:
            from digit_applet import run_applet
        except Exception as exc:
            print(f"Could not import Tkinter applet: {exc}")
            print("If needed on macOS, install a Python build with Tk support.")
            return
        run_applet(
            model,
            debug=args.debug_app,
            training_dashboard=training_dashboard,
            model_path=app_model_path,
        )


if __name__ == "__main__":
    main()
