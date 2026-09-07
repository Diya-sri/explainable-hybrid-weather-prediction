"""Small NumPy LSTM and hybrid inference utilities used by training and Flask."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    shifted = x - x.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


@dataclass
class NumpyLSTM:
    input_size: int
    hidden_size: int
    output_size: int
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.params = {
            "W": rng.normal(0, scale, (self.input_size + self.hidden_size, 4 * self.hidden_size)),
            "b": np.zeros(4 * self.hidden_size),
            "Wy": rng.normal(0, np.sqrt(2.0 / self.hidden_size), (self.hidden_size, self.output_size)),
            "by": np.zeros(self.output_size),
        }

    def forward(self, x, cache=False):
        p, hsize = self.params, self.hidden_size
        h = np.zeros((x.shape[0], hsize))
        c = np.zeros_like(h)
        saved = []
        for t in range(x.shape[1]):
            previous_h, previous_c = h, c
            joined = np.concatenate([x[:, t], previous_h], axis=1)
            gates = joined @ p["W"] + p["b"]
            i = _sigmoid(gates[:, :hsize])
            f = _sigmoid(gates[:, hsize:2*hsize])
            g = np.tanh(gates[:, 2*hsize:3*hsize])
            o = _sigmoid(gates[:, 3*hsize:])
            c = f * previous_c + i * g
            h = o * np.tanh(c)
            if cache:
                saved.append((joined, previous_c, i, f, g, o, c))
        logits = h @ p["Wy"] + p["by"]
        return (logits, saved, h) if cache else logits

    def predict_proba(self, x):
        return softmax(self.forward(np.asarray(x, dtype=np.float64)))

    def loss_and_gradients(self, x, y, class_weights):
        logits, saved, last_h = self.forward(x, cache=True)
        probabilities = softmax(logits)
        weights = class_weights[y]
        loss = -np.sum(weights * np.log(probabilities[np.arange(len(y)), y] + 1e-12)) / weights.sum()
        dlogits = probabilities
        dlogits[np.arange(len(y)), y] -= 1
        dlogits *= (weights / weights.sum())[:, None]

        p = self.params
        grads = {name: np.zeros_like(value) for name, value in p.items()}
        grads["Wy"] = last_h.T @ dlogits
        grads["by"] = dlogits.sum(axis=0)
        dh_next = dlogits @ p["Wy"].T
        dc_next = np.zeros_like(dh_next)
        for joined, previous_c, i, f, g, o, c in reversed(saved):
            tanh_c = np.tanh(c)
            do = dh_next * tanh_c
            dc = dc_next + dh_next * o * (1 - tanh_c * tanh_c)
            df, di, dg = dc * previous_c, dc * g, dc * i
            dc_next = dc * f
            da = np.concatenate([
                di * i * (1 - i), df * f * (1 - f), dg * (1 - g * g), do * o * (1 - o)
            ], axis=1)
            grads["W"] += joined.T @ da
            grads["b"] += da.sum(axis=0)
            dh_next = (da @ p["W"].T)[:, self.input_size:]
        total_norm = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        if total_norm > 5.0:
            grads = {k: v * (5.0 / total_norm) for k, v in grads.items()}
        return float(loss), grads

    def fit(self, x, y, x_val, y_val, epochs=12, batch_size=256, learning_rate=0.003):
        counts = np.bincount(y, minlength=self.output_size).astype(float)
        class_weights = np.sqrt(len(y) / (self.output_size * np.maximum(counts, 1)))
        moments = {k: np.zeros_like(v) for k, v in self.params.items()}
        velocities = {k: np.zeros_like(v) for k, v in self.params.items()}
        rng, step, best = np.random.default_rng(self.seed), 0, None
        best_val = np.inf
        for epoch in range(epochs):
            order = rng.permutation(len(y))
            losses = []
            for start in range(0, len(y), batch_size):
                idx = order[start:start + batch_size]
                loss, grads = self.loss_and_gradients(x[idx], y[idx], class_weights)
                losses.append(loss)
                step += 1
                for name, grad in grads.items():
                    moments[name] = .9 * moments[name] + .1 * grad
                    velocities[name] = .999 * velocities[name] + .001 * grad * grad
                    mhat = moments[name] / (1 - .9 ** step)
                    vhat = velocities[name] / (1 - .999 ** step)
                    self.params[name] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            val_p = self.predict_proba(x_val)
            val_loss = -np.mean(np.log(val_p[np.arange(len(y_val)), y_val] + 1e-12))
            val_acc = np.mean(val_p.argmax(1) == y_val)
            print(f"epoch {epoch+1:02d}: loss={np.mean(losses):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best = {k: v.copy() for k, v in self.params.items()}
        self.params = best

    def save(self, path):
        np.savez_compressed(path, input_size=self.input_size, hidden_size=self.hidden_size,
                            output_size=self.output_size, **self.params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(int(data["input_size"]), int(data["hidden_size"]), int(data["output_size"]))
        model.params = {k: data[k] for k in ("W", "b", "Wy", "by")}
        return model


def hybrid_probabilities(tree, lstm, sequences, blend_weight):
    tree_p = tree.predict_proba(sequences[:, -1, :])
    lstm_p = lstm.predict_proba(sequences)
    return blend_weight * tree_p + (1.0 - blend_weight) * lstm_p
