import numpy as np
from data_loader import load_data, shuffle_data
from network import NeuralNetwork

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def accuracy_from_probs(y_probs, y_true_one_hot):
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(y_true_one_hot, axis=1)
    return np.mean(y_pred == y_true)


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def make_batches(X, y, batch_size):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


def train_model(
    epochs=20,
    learning_rate=0.01,
    batch_size=64,
    data_path="data/Reduced_MNIST_Data",
):
    X_train, y_train, X_test, y_test = load_data(data_path, shuffle_train=True)

    model = NeuralNetwork(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        X_train, y_train = shuffle_data(X_train, y_train)
        epoch_loss = 0.0
        steps = 0

        for X_batch, y_batch in make_batches(X_train, y_train, batch_size):
            y_hat = model.forward(X_batch)
            loss = model.compute_loss(y_hat, y_batch)
            model.backward(y_batch)
            model.update_params(learning_rate)

            epoch_loss += loss
            steps += 1

        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)
        train_acc = accuracy_from_probs(train_probs, y_train)
        test_acc = accuracy_from_probs(test_probs, y_test)
        mean_loss = epoch_loss / max(steps, 1)

        history["train_loss"].append(mean_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={mean_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    return model, history, (X_test, y_test)


def plot_training_curves(history, output_html="training_curves.html"):
    if go is None:
        print("Plotly not installed: skipping training curve visualization.")
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_acc"],
            mode="lines+markers",
            name="Train Accuracy",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["test_acc"],
            mode="lines+markers",
            name="Test Accuracy",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Training Curves",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")


def plot_network_architecture(output_html="network_architecture.html"):
    if go is None:
        print("Plotly not installed: skipping network architecture visualization.")
        return

    layers = [784, 128, 64, 10]
    layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
    max_nodes_to_draw = [16, 16, 12, 10]

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    layer_nodes = []

    for layer_index, (layer_size, max_draw, name) in enumerate(zip(layers, max_nodes_to_draw, layer_names)):
        draw_count = min(layer_size, max_draw)
        y_positions = np.linspace(1, -1, draw_count)
        current_nodes = []
        for i in range(draw_count):
            node_x.append(layer_index)
            node_y.append(y_positions[i])
            node_text.append(f"{name} neuron {i + 1}<br>Layer size: {layer_size}")
            node_color.append(layer_index)
            current_nodes.append((layer_index, y_positions[i]))
        layer_nodes.append(current_nodes)

    edge_x = []
    edge_y = []
    for li in range(len(layer_nodes) - 1):
        for x0, y0 in layer_nodes[li]:
            for x1, y1 in layer_nodes[li + 1]:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(200,200,200,0.2)", width=1),
            hoverinfo="skip",
            name="Connections",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=12, color=node_color, colorscale="Turbo", showscale=False),
            text=node_text,
            hoverinfo="text",
            name="Neurons",
        )
    )
    fig.update_layout(
        title="Neural Network Architecture (784-128-64-10)",
        template="plotly_dark",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(layer_names))),
            ticktext=[f"{name}<br>({size})" for name, size in zip(layer_names, layers)],
            title="Layers",
        ),
        yaxis=dict(showticklabels=False, title=""),
        showlegend=False,
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")


def plot_confusion_matrix(model, X_test, y_test, output_html="confusion_matrix.html"):
    if go is None:
        print("Plotly not installed: skipping confusion matrix visualization.")
        return

    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            colorscale="Viridis",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        xaxis_title="Predicted",
        yaxis_title="True",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")







