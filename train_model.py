from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from needle_hook_features import FEATURE_NAMES, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train needle-hook failure classifier from valid/invalid CSV files."
    )
    parser.add_argument("--valid-dir", type=Path, default=Path("valid"))
    parser.add_argument("--invalid-dir", type=Path, default=Path("invalid"))
    parser.add_argument(
        "--model-out", type=Path, default=Path("model") / "needle_hook_model.joblib"
    )
    parser.add_argument(
        "--metrics-out", type=Path, default=Path("model") / "metrics.json"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x, y, paths = build_dataset(args.valid_dir, args.invalid_dir)
    x_train, x_test, y_train, y_test, p_train, p_test = train_test_split(
        x,
        y,
        paths,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "num_samples_total": int(len(y)),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
        "class_distribution_total": {
            "valid": int((y == 0).sum()),
            "invalid": int((y == 1).sum()),
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_invalid": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_invalid": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_invalid": float(f1_score(y_test, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["valid", "invalid"], output_dict=True
        ),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_map": {0: "valid", 1: "invalid"},
    }
    joblib.dump(artifact, args.model_out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved model: {args.model_out}")
    print(f"Saved metrics: {args.metrics_out}")
    print(
        "Test metrics: "
        f"acc={metrics['accuracy']:.4f}, "
        f"f1_invalid={metrics['f1_invalid']:.4f}, "
        f"recall_invalid={metrics['recall_invalid']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])

    top_importance = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    print("Top feature importances:")
    for name, score in top_importance:
        print(f"  {name}: {score:.6f}")


if __name__ == "__main__":
    main()
