"""E53 model helpers: cached embeddings and the unchanged v6.3b head recipe."""
from __future__ import annotations

import json

import e53_common as C


def load_embeddings():
    import numpy as np
    prov = json.loads((C.CACHE / "embedding_provenance.json").read_text(encoding="utf-8"))
    got = C.sha256_file(C.CACHE / "embeddings.npy")
    if got != prov["embeddings_npy_sha256"]:
        raise SystemExit("STOP: cached embeddings do not match their recorded hash")
    E = np.load(C.CACHE / "embeddings.npy")
    order = json.loads((C.CACHE / "embedding_index.json").read_text(encoding="utf-8"))
    return E, {h: k for k, h in enumerate(order)}


def tr_rows_in_recipe_order():
    """
    Training rows in the order the v6.3b script fits them: the order train_test_split
    returns, NOT corpus order. CalibratedClassifierCV(cv=3) folds depend on row order,
    so this order is part of the recipe (deviation D9). The order is recomputed and
    checked against manifests/train_split_order.json (created once).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    corpus = C.read_jsonl(C.MANIFESTS / "train_corpus.jsonl")
    y = np.array([r["label"] for r in corpus])
    tr, _ = train_test_split(np.arange(len(corpus)), test_size=0.10, random_state=C.SEED, stratify=y)
    order = [int(i) for i in tr]
    if {r["i"] for r in corpus if r["split"] == "tr"} != set(order):
        raise SystemExit("STOP: recomputed tr split differs from the manifest")
    path = C.MANIFESTS / "train_split_order.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != order:
            raise SystemExit("STOP: tr order differs from manifests/train_split_order.json")
    else:
        C.write_json_new(path, order)
    return [corpus[i] for i in order]


def fit_recipe(E, row_of, rows, weights=None):
    """The v6.3b recipe, unchanged: LinearSVC(C=0.8) + sigmoid calibration, cv=3."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    X = E[[row_of[r["text_sha256"]] for r in rows]]
    y = np.array([r["label"] for r in rows])
    w = np.array(weights if weights is not None else [r["weight"] for r in rows], dtype=float)
    model = CalibratedClassifierCV(LinearSVC(C=0.8, max_iter=5000, class_weight="balanced"),
                                   cv=3, method="sigmoid")
    model.fit(X, y, sample_weight=w)
    return model


def predict(model, X):
    return model.predict_proba(X)[:, 1].astype("float64")
