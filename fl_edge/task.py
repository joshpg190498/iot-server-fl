"""fl_edge.task — piezas de aprendizaje automático, independientes de Flower.

Contiene el modelo Keras, la carga y partición de datos, el entrenamiento local,
la evaluación centralizada y el export a .tflite. NO importa Flower: así puede
probarse de forma aislada y reutilizarse tanto en simulación como en el SBC real.

TensorFlow se importa de forma perezosa dentro de cada función que lo necesita,
para que las utilidades de partición (numpy puro) puedan testearse sin cargar TF.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

Dataset = Tuple[np.ndarray, np.ndarray]

# Metadatos por dataset. Fashion-MNIST/MNIST son 28x28x1; CIFAR-10 es 32x32x3.
DATASET_META: Dict[str, Dict] = {
    "fashion_mnist": {"shape": (28, 28, 1), "classes": 10},
    "mnist": {"shape": (28, 28, 1), "classes": 10},
    "cifar10": {"shape": (32, 32, 3), "classes": 10},
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def _load_raw(dataset: str) -> Tuple[Dataset, Dataset]:
    """Carga y normaliza un dataset de keras. Devuelve (train, test)."""
    import tensorflow as tf

    if dataset == "fashion_mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == "mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    elif dataset == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    x_tr = x_tr.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0
    # asegurar dimensión de canal (grises -> ...,1)
    if x_tr.ndim == 3:
        x_tr = x_tr[..., np.newaxis]
        x_te = x_te[..., np.newaxis]
    y_tr = y_tr.reshape(-1).astype("int64")
    y_te = y_te.reshape(-1).astype("int64")
    return (x_tr, y_tr), (x_te, y_te)


def partition_iid(x: np.ndarray, y: np.ndarray, num_partitions: int, seed: int = 42) -> List[Dataset]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    shards = np.array_split(idx, num_partitions)
    return [(x[s], y[s]) for s in shards]


def partition_dirichlet(
    x: np.ndarray,
    y: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int = 42,
    num_classes: int = 10,
) -> List[Dataset]:
    """Partición no-IID: Dirichlet(alpha) por clase. alpha bajo => más heterogéneo."""
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    for c in class_indices:
        rng.shuffle(c)

    client_indices: List[List[int]] = [[] for _ in range(num_partitions)]
    for c in range(num_classes):
        proportions = rng.dirichlet(alpha * np.ones(num_partitions))
        counts = (proportions * len(class_indices[c])).astype(int)
        while counts.sum() < len(class_indices[c]):
            counts[rng.integers(num_partitions)] += 1
        splits = np.split(class_indices[c], np.cumsum(counts)[:-1])
        for i in range(num_partitions):
            client_indices[i].extend(splits[i].tolist())

    parts: List[Dataset] = []
    for i in range(num_partitions):
        ci = np.array(client_indices[i], dtype=int)
        rng.shuffle(ci)
        parts.append((x[ci], y[ci]))
    return parts


def load_partition(
    partition_id: int,
    num_partitions: int,
    dataset: str = "fashion_mnist",
    alpha: float | None = None,
    seed: int = 42,
) -> Dataset:
    """Datos de entrenamiento de un cliente. alpha=None => IID; alpha>0 => Dirichlet."""
    (x_tr, y_tr), _ = _load_raw(dataset)
    meta = DATASET_META[dataset]
    if alpha is None:
        parts = partition_iid(x_tr, y_tr, num_partitions, seed)
    else:
        parts = partition_dirichlet(x_tr, y_tr, num_partitions, alpha, seed, meta["classes"])
    return parts[partition_id]


def load_centralized_testset(dataset: str = "fashion_mnist") -> Dataset:
    """Conjunto de test completo para la evaluación centralizada en el servidor."""
    _, (x_te, y_te) = _load_raw(dataset)
    return x_te, y_te


def load_centralized_trainset(
    dataset: str = "fashion_mnist", max_samples: int = 5000, seed: int = 42
) -> Dataset:
    """Subconjunto del set de entrenamiento para evaluar el modelo global sobre
    datos de train (curva de entrenamiento). Comparar con el test da la brecha de
    sobreajuste. Se usa un subconjunto por eficiencia."""
    (x_tr, y_tr), _ = _load_raw(dataset)
    if max_samples and len(x_tr) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_tr), size=max_samples, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    return x_tr, y_tr


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_model(dataset: str = "fashion_mnist", lr: float = 0.01, momentum: float = 0.9):
    """CNN pequeña, adecuada para correr en un SBC. Se compila con SGD."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    meta = DATASET_META[dataset]
    model = models.Sequential(
        [
            layers.Input(shape=meta["shape"]),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(meta["classes"], activation="softmax"),
        ],
        name=f"{dataset}_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_weights(model) -> List[np.ndarray]:
    return model.get_weights()


def set_weights(model, weights: List[np.ndarray]) -> None:
    model.set_weights(weights)


# ---------------------------------------------------------------------------
# Entrenamiento local y evaluación
# ---------------------------------------------------------------------------
def planned_steps(num_examples: int, epochs: int, batch_size: int) -> int:
    """Pasos de optimización que haría el entrenamiento completo (sin presupuesto)."""
    if num_examples <= 0:
        return 0
    return max(1, int(epochs)) * math.ceil(num_examples / max(1, int(batch_size)))


def _budget_callback(time_budget_s: float):
    """Callback Keras que cuenta pasos y corta el entrenamiento al agotar el presupuesto."""
    import tensorflow as tf

    class _TimeBudget(tf.keras.callbacks.Callback):
        def __init__(self, budget_s: float):
            super().__init__()
            self.budget_s = float(budget_s or 0.0)
            self.steps = 0
            self.last_logs: Dict[str, float] = {}
            self._t0 = 0.0

        def on_train_begin(self, logs=None):
            self._t0 = time.monotonic()

        def on_train_batch_end(self, batch, logs=None):
            self.steps += 1
            self.last_logs = dict(logs or {})
            if self.budget_s > 0 and time.monotonic() - self._t0 >= self.budget_s:
                self.model.stop_training = True

    return _TimeBudget(time_budget_s)


def train_local(
    model,
    data: Dataset,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.9,
    time_budget_s: float = 0.0,
) -> Tuple[int, Dict[str, float]]:
    """Entrena el modelo con los datos locales. Recompila para resetear el momentum.

    Presupuesto de tiempo: hasta `epochs` épocas o `time_budget_s` segundos, lo que
    ocurra primero (0 = sin presupuesto). Un nodo degradado completa menos pasos en la
    misma ventana: ese déficit de trabajo es el EFECTO que pondera la estrategia.

    Devuelve (num_examples, metrics) con metrics incluyendo steps_done y steps_planned.
    """
    import tensorflow as tf

    x, y = data
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if len(x) == 0:
        return 0, {"train_loss": 0.0, "train_accuracy": 0.0, "steps_done": 0, "steps_planned": 0}

    budget = _budget_callback(time_budget_s)
    hist = model.fit(x, y, epochs=max(1, int(epochs)), batch_size=batch_size, verbose=0,
                     callbacks=[budget])
    history = hist.history or {}
    loss = history.get("loss") or [budget.last_logs.get("loss", 0.0)]
    acc = history.get("accuracy") or [budget.last_logs.get("accuracy", 0.0)]
    metrics = {
        "train_loss": float(loss[-1]),
        "train_accuracy": float(acc[-1]),
        "steps_done": int(budget.steps),
        "steps_planned": planned_steps(len(x), epochs, batch_size),
    }
    return len(x), metrics


def evaluate(model, testset: Dataset) -> Tuple[float, float]:
    """Evaluación centralizada. Devuelve (loss, accuracy)."""
    x, y = testset
    loss, acc = model.evaluate(x, y, verbose=0)
    return float(loss), float(acc)


# ---------------------------------------------------------------------------
# Persistencia del modelo global
# ---------------------------------------------------------------------------
def export_tflite(model, server_round: int, out_dir: str = "checkpoints") -> str:
    """Convierte el modelo global a .tflite (formato objetivo para el SBC)."""
    import tensorflow as tf

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"global_r{server_round}.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(path, "wb") as f:
        f.write(converter.convert())
    return path


def save_keras(model, server_round: int, out_dir: str = "checkpoints") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"global_r{server_round}.keras")
    model.save(path)
    return path


# ---------------------------------------------------------------------------
# Demo standalone (paso 2): probar el pipeline local sin Flower.
#   docker compose run --rm fl-sim uv run python -m fl_edge.task
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATASET = "fashion_mnist"
    print(f"[task] cargando partición 0 de 2 ({DATASET}, IID)...")
    data = load_partition(partition_id=0, num_partitions=2, dataset=DATASET)
    x, y = data
    print(f"[task] muestras locales: {len(x)}  | forma: {x.shape}  | clases: {np.unique(y)}")

    print("[task] construyendo modelo...")
    model = build_model(DATASET)
    model.summary()

    print("[task] entrenando 1 época local...")
    n, m = train_local(model, data, epochs=1, batch_size=32, lr=0.01)
    print(f"[task] entrenado con {n} muestras | train_acc={m['train_accuracy']:.4f}")

    print("[task] evaluación centralizada...")
    testset = load_centralized_testset(DATASET)
    loss, acc = evaluate(model, testset)
    print(f"[task] central_loss={loss:.4f}  central_acc={acc:.4f}")

    tfl = export_tflite(model, server_round=0)
    print(f"[task] exportado: {tfl}")
    print("[task] OK")