from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from needle_hook_features import extract_features_from_csv, iter_csv_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict needle-hook state from CSV file(s)."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("model") / "needle_hook_model.joblib"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one CSV file or a folder containing CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path to save prediction table.",
    )
    return parser.parse_args()


def gather_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = iter_csv_files(input_path)
        if not files:
            raise ValueError(f"No CSV files found in folder: {input_path}")
        return files
    raise ValueError(f"Input path not found: {input_path}")


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]

    files = gather_inputs(args.input)
    rows = []
    for csv_path in files:
        x = extract_features_from_csv(csv_path).reshape(1, -1)
        pred = int(model.predict(x)[0])
        prob_invalid = float(model.predict_proba(x)[0, 1])
        rows.append(
            {
                "file": str(csv_path),
                "pred_label": "invalid" if pred == 1 else "valid",
                "prob_invalid": prob_invalid,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="prob_invalid", ascending=False)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved predictions: {args.output_csv}")

    with pd.option_context("display.max_rows", 50, "display.max_colwidth", 120):
        print(df)


if __name__ == "__main__":
    main()

