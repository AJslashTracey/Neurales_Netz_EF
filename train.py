import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def plot_training_animation(history, output_html="training_animation.html"):
    if go is None:
        print("Plotly not installed: skipping training animation.")
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    test_acc = history["test_acc"]

    frames = []
    for i in range(len(epochs)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=epochs[: i + 1],
                        y=train_loss[: i + 1],
                        mode="lines+markers",
                        name="Train Loss",
                        line=dict(color="#00CC96"),
                    ),
                    go.Scatter(
                        x=epochs[: i + 1],
                        y=train_acc[: i + 1],
                        mode="lines+markers",
                        name="Train Accuracy",
                        yaxis="y2",
                        line=dict(color="#636EFA"),
                    ),
                    go.Scatter(
                        x=epochs[: i + 1],
                        y=test_acc[: i + 1],
                        mode="lines+markers",
                        name="Test Accuracy",
                        yaxis="y2",
                        line=dict(color="#EF553B"),
                    ),
                ],
                name=str(i + 1),
            )
        )

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
    )

    fig.update_layout(
        title="Live Training Animation (Epoch by Epoch)",
        template="plotly_dark",
        xaxis=dict(title="Epoch", range=[1, len(epochs)]),
        yaxis=dict(title="Loss", range=[0, max(train_loss) * 1.1]),
        yaxis2=dict(
            title="Accuracy",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "x": 0.05,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "y": -0.08,
                "len": 0.8,
                "currentvalue": {"prefix": "Epoch: "},
                "steps": [
                    {
                        "label": str(i + 1),
                        "method": "animate",
                        "args": [[str(i + 1)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(len(epochs))
                ],
            }
        ],
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

    num_layers = len(layer_names)
    num_frames = 48

    frames = []
    for frame_idx in range(num_frames):
        phase = 2 * np.pi * frame_idx / num_frames
        active_layer = frame_idx % num_layers

        dynamic_sizes = []
        for x in node_x:
            layer_boost = 7 if x == active_layer else 0
            pulse = 2 * np.sin(phase + x * 0.6)
            dynamic_sizes.append(10 + layer_boost + pulse)

        edge_alpha = 0.06 + 0.22 * (0.5 + 0.5 * np.sin(phase))
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color=f"rgba(200,200,200,{edge_alpha:.3f})", width=1),
                    hoverinfo="skip",
                    name="Connections",
                ),
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker=dict(
                        size=dynamic_sizes,
                        color=node_color,
                        colorscale="Turbo",
                        showscale=False,
                        line=dict(width=0.4, color="rgba(255,255,255,0.25)"),
                    ),
                    text=node_text,
                    hoverinfo="text",
                    name="Neurons",
                ),
            ],
            name=str(frame_idx),
        )
        frames.append(frame)

    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
    fig.update_layout(
        title="Animated Neural Network Architecture (784-128-64-10)",
        template="plotly_dark",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(layer_names))),
            ticktext=[f"{name}<br>({size})" for name, size in zip(layer_names, layers)],
            title="Layers",
        ),
        yaxis=dict(showticklabels=False, title=""),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "y": -0.08,
                "len": 0.8,
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for i in range(num_frames)
                ],
            }
        ],
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


def _build_digit_route_sankey(model, x_sample, target_digit, output_html, sample_index):
    if go is None:
        return

    x = x_sample.reshape(1, -1)
    probs = model.predict_proba(x)[0]
    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])

    input_vec = model.X[0]
    a1 = model.a1[0]
    a2 = model.a2[0]

    top_h2 = 8
    top_h1 = 12
    top_inputs = 20

    h2_scores = a2 * np.maximum(model.W3[:, pred_digit], 0.0)
    h2_idx = np.argsort(h2_scores)[-top_h2:][::-1]

    h1_scores = np.zeros_like(a1)
    for j in range(len(a1)):
        score = 0.0
        for i in h2_idx:
            score += a1[j] * max(model.W2[j, i], 0.0)
        h1_scores[j] = score
    h1_idx = np.argsort(h1_scores)[-top_h1:][::-1]

    input_scores = np.zeros_like(input_vec)
    for k in range(len(input_vec)):
        score = 0.0
        for j in h1_idx:
            score += input_vec[k] * max(model.W1[k, j], 0.0)
        input_scores[k] = score
    input_idx = np.argsort(input_scores)[-top_inputs:][::-1]

    node_labels = []
    node_colors = []
    node_map = {}

    for idx in input_idx:
        r, c = divmod(int(idx), 28)
        label = f"px({r},{c})"
        node_map[("in", int(idx))] = len(node_labels)
        node_labels.append(label)
        node_colors.append("rgba(99,110,250,0.85)")

    for j in h1_idx:
        node_map[("h1", int(j))] = len(node_labels)
        node_labels.append(f"H1-{int(j)}")
        node_colors.append("rgba(0,204,150,0.85)")

    for i in h2_idx:
        node_map[("h2", int(i))] = len(node_labels)
        node_labels.append(f"H2-{int(i)}")
        node_colors.append("rgba(255,161,90,0.9)")

    out_key = ("out", pred_digit)
    node_map[out_key] = len(node_labels)
    node_labels.append(f"OUT-{pred_digit}")
    node_colors.append("rgba(239,85,59,0.95)")

    sources = []
    targets = []
    values = []

    for k in input_idx:
        for j in h1_idx:
            val = input_vec[k] * max(model.W1[k, j], 0.0)
            if val > 1e-8:
                sources.append(node_map[("in", int(k))])
                targets.append(node_map[("h1", int(j))])
                values.append(float(val))

    for j in h1_idx:
        for i in h2_idx:
            val = a1[j] * max(model.W2[j, i], 0.0)
            if val > 1e-8:
                sources.append(node_map[("h1", int(j))])
                targets.append(node_map[("h2", int(i))])
                values.append(float(val))

    for i in h2_idx:
        val = a2[i] * max(model.W3[i, pred_digit], 0.0)
        if val > 1e-8:
            sources.append(node_map[("h2", int(i))])
            targets.append(node_map[out_key])
            values.append(float(val))

    if not values:
        values = [1.0]
        sources = [node_map[("h2", int(h2_idx[0]))]]
        targets = [node_map[out_key]]

    scale = max(values)
    values = [v / scale for v in values]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=14,
                    thickness=14,
                    label=node_labels,
                    color=node_colors,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(180,180,220,0.28)",
                ),
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        title=(
            f"Digit Route Visualization | true={target_digit} pred={pred_digit} "
            f"(conf={confidence:.3f}, sample={sample_index})"
        ),
        font=dict(size=12),
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")


def plot_digit_routes(model, X_test, y_test, output_dir="plots/digit_routes"):
    if go is None:
        print("Plotly not installed: skipping digit route visualizations.")
        return

    os.makedirs(output_dir, exist_ok=True)
    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)

    for digit in range(10):
        true_mask = np.where(y_true == digit)[0]
        if len(true_mask) == 0:
            continue

        correct_mask = [idx for idx in true_mask if y_pred[idx] == digit]
        chosen_idx = int(correct_mask[0] if correct_mask else true_mask[0])

        output_html = os.path.join(output_dir, f"digit_{digit}_route.html")
        _build_digit_route_sankey(
            model=model,
            x_sample=X_test[chosen_idx],
            target_digit=digit,
            output_html=output_html,
            sample_index=chosen_idx,
        )


def _prepare_digit_route_data(model, x_sample, target_digit):
    x = x_sample.reshape(1, -1)
    probs = model.predict_proba(x)[0]
    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])

    input_vec = model.X[0]
    a1 = model.a1[0]
    a2 = model.a2[0]

    top_h2 = 10
    top_h1 = 16
    top_inputs = 28

    h2_scores = a2 * np.maximum(model.W3[:, pred_digit], 0.0)
    h2_idx = np.argsort(h2_scores)[-top_h2:][::-1]

    h1_scores = np.zeros_like(a1)
    for j in range(len(a1)):
        score = 0.0
        for i in h2_idx:
            score += a1[j] * max(model.W2[j, i], 0.0)
        h1_scores[j] = score
    h1_idx = np.argsort(h1_scores)[-top_h1:][::-1]

    input_scores = np.zeros_like(input_vec)
    for k in range(len(input_vec)):
        score = 0.0
        for j in h1_idx:
            score += input_vec[k] * max(model.W1[k, j], 0.0)
        input_scores[k] = score
    input_idx = np.argsort(input_scores)[-top_inputs:][::-1]

    c_in_h1 = []
    for k in input_idx:
        for j in h1_idx:
            val = input_vec[k] * max(model.W1[k, j], 0.0)
            if val > 1e-9:
                c_in_h1.append((int(k), int(j), float(val)))

    c_h1_h2 = []
    for j in h1_idx:
        for i in h2_idx:
            val = a1[j] * max(model.W2[j, i], 0.0)
            if val > 1e-9:
                c_h1_h2.append((int(j), int(i), float(val)))

    c_h2_out = []
    for i in h2_idx:
        val = a2[i] * max(model.W3[i, pred_digit], 0.0)
        if val > 1e-9:
            c_h2_out.append((int(i), int(pred_digit), float(val)))

    all_vals = [v for _, _, v in c_in_h1 + c_h1_h2 + c_h2_out]
    max_val = max(all_vals) if all_vals else 1.0

    for idx in range(len(c_in_h1)):
        k, j, v = c_in_h1[idx]
        c_in_h1[idx] = (k, j, v / max_val)
    for idx in range(len(c_h1_h2)):
        j, i, v = c_h1_h2[idx]
        c_h1_h2[idx] = (j, i, v / max_val)
    for idx in range(len(c_h2_out)):
        i, o, v = c_h2_out[idx]
        c_h2_out[idx] = (i, o, v / max_val)

    return {
        "target_digit": int(target_digit),
        "pred_digit": pred_digit,
        "confidence": confidence,
        "input_idx": input_idx,
        "h1_idx": h1_idx,
        "h2_idx": h2_idx,
        "c_in_h1": c_in_h1,
        "c_h1_h2": c_h1_h2,
        "c_h2_out": c_h2_out,
    }


def _animate_digit_route_matplotlib(route_data, output_gif, title_prefix="", fps=14, total_frames=90):
    input_idx = route_data["input_idx"]
    h1_idx = route_data["h1_idx"]
    h2_idx = route_data["h2_idx"]
    pred_digit = route_data["pred_digit"]
    target_digit = route_data["target_digit"]
    confidence = route_data["confidence"]

    x_positions = [0.0, 1.0, 2.0, 3.0]
    y_input = np.linspace(1.0, -1.0, len(input_idx))
    y_h1 = np.linspace(1.0, -1.0, len(h1_idx))
    y_h2 = np.linspace(1.0, -1.0, len(h2_idx))
    y_out = np.array([0.0])

    pos_in = {int(k): (x_positions[0], y_input[ii]) for ii, k in enumerate(input_idx)}
    pos_h1 = {int(j): (x_positions[1], y_h1[ii]) for ii, j in enumerate(h1_idx)}
    pos_h2 = {int(i): (x_positions[2], y_h2[ii]) for ii, i in enumerate(h2_idx)}
    pos_out = {pred_digit: (x_positions[3], y_out[0])}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.set_facecolor("#0b0d12")
    fig.patch.set_facecolor("#0b0d12")
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")

    ax.text(0.0, 1.09, f"Input ({len(input_idx)})", color="white", ha="center", fontsize=10)
    ax.text(1.0, 1.09, f"Hidden1 ({len(h1_idx)})", color="white", ha="center", fontsize=10)
    ax.text(2.0, 1.09, f"Hidden2 ({len(h2_idx)})", color="white", ha="center", fontsize=10)
    ax.text(3.0, 1.09, "Output (1)", color="white", ha="center", fontsize=10)

    title = ax.set_title("", color="white", fontsize=12, pad=16)

    node_in = ax.scatter([p[0] for p in pos_in.values()], [p[1] for p in pos_in.values()], s=36, c="#636efa")
    node_h1 = ax.scatter([p[0] for p in pos_h1.values()], [p[1] for p in pos_h1.values()], s=44, c="#00cc96")
    node_h2 = ax.scatter([p[0] for p in pos_h2.values()], [p[1] for p in pos_h2.values()], s=52, c="#ffa15a")
    node_out = ax.scatter([3.0], [0.0], s=95, c="#ef553b")

    lines_1 = []
    for k, j, v in route_data["c_in_h1"]:
        x0, y0 = pos_in[k]
        x1, y1 = pos_h1[j]
        line, = ax.plot([x0, x1], [y0, y1], color="#9bb0ff", alpha=0.03, linewidth=0.4 + 2.8 * v)
        lines_1.append((line, v))

    lines_2 = []
    for j, i, v in route_data["c_h1_h2"]:
        x0, y0 = pos_h1[j]
        x1, y1 = pos_h2[i]
        line, = ax.plot([x0, x1], [y0, y1], color="#7fffd4", alpha=0.03, linewidth=0.4 + 2.8 * v)
        lines_2.append((line, v))

    lines_3 = []
    for i, _, v in route_data["c_h2_out"]:
        x0, y0 = pos_h2[i]
        x1, y1 = pos_out[pred_digit]
        line, = ax.plot([x0, x1], [y0, y1], color="#ff9f9a", alpha=0.03, linewidth=0.6 + 3.2 * v)
        lines_3.append((line, v))

    def stage_progress(frame, start, end):
        if frame < start:
            return 0.0
        if frame > end:
            return 1.0
        return (frame - start) / max(1, (end - start))

    s1_end = total_frames // 3
    s2_end = 2 * total_frames // 3
    s3_end = total_frames - 1

    def update(frame):
        p1 = stage_progress(frame, 0, s1_end)
        p2 = stage_progress(frame, s1_end, s2_end)
        p3 = stage_progress(frame, s2_end, s3_end)

        pulse = 0.75 + 0.25 * np.sin(frame * 0.35)

        for ln, v in lines_1:
            ln.set_alpha(0.03 + 0.85 * v * p1 * pulse)
        for ln, v in lines_2:
            ln.set_alpha(0.03 + 0.85 * v * p2 * pulse)
        for ln, v in lines_3:
            ln.set_alpha(0.03 + 0.95 * v * p3 * pulse)

        node_in.set_sizes(np.full(len(input_idx), 22 + 24 * p1))
        node_h1.set_sizes(np.full(len(h1_idx), 28 + 34 * p2))
        node_h2.set_sizes(np.full(len(h2_idx), 34 + 42 * p3))
        node_out.set_sizes(np.full(1, 80 + 90 * p3))

        title.set_text(
            f"{title_prefix} true={target_digit} pred={pred_digit} conf={confidence:.3f} | frame {frame + 1}/{total_frames}"
        )
        return [ln for ln, _ in lines_1 + lines_2 + lines_3] + [node_in, node_h1, node_h2, node_out, title]

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(output_gif, writer=writer)
    plt.close(fig)
    print(f"Saved: {output_gif}")


def plot_digit_routes_live(model, X_test, y_test, output_dir="plots/digit_routes_live", fps=14):
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)

    for digit in range(10):
        true_mask = np.where(y_true == digit)[0]
        if len(true_mask) == 0:
            continue

        correct_mask = [idx for idx in true_mask if y_pred[idx] == digit]
        chosen_idx = int(correct_mask[0] if correct_mask else true_mask[0])

        route_data = _prepare_digit_route_data(model, X_test[chosen_idx], digit)
        output_gif = os.path.join(output_dir, f"digit_{digit}_live.gif")
        _animate_digit_route_matplotlib(
            route_data,
            output_gif=output_gif,
            title_prefix="Live route",
            fps=fps,
            total_frames=84,
        )







