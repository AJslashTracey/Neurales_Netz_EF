import argparse
import os

from network import NeuralNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run a digit classifier applet.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--app", action="store_true", help="Launch the drawing applet.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--model-path", type=str, default="model.npz", help="Model file path.")
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

    if should_train:
        from data_loader import load_data
        from train import train_model

        X_train, y_train, X_test, y_test = load_data()
        print(
            f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )

        model = NeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dims=(128, 64),
            output_dim=y_train.shape[1],
            learning_rate=args.learning_rate,
            seed=42,
        )

        train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model.save_model(args.model_path)
        print(f"Saved model to: {args.model_path}")

    if args.app:
        print("Starting digit applet...", flush=True)
        if model is None:
            if not os.path.exists(args.model_path):
                print(
                    f"Model file not found: {args.model_path}\n"
                    "Run training first, e.g.: python main.py --train"
                )
                return
            if args.debug_app:
                print(f"Loading model from: {args.model_path}", flush=True)
            model = NeuralNetwork.load_model(args.model_path)
            if args.debug_app:
                print("Model loaded successfully.", flush=True)
        try:
            from digit_applet import run_applet
        except Exception as exc:
            print(f"Could not import Tkinter applet: {exc}")
            print("If needed on macOS, install a Python build with Tk support.")
            return
        run_applet(model, debug=args.debug_app)


if __name__ == "__main__":
    main()
