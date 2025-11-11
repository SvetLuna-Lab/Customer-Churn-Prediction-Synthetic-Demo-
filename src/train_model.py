from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .eval_utils import print_classification_metrics, plot_roc_curve


def main() -> None:
    """
    Train a churn prediction model on the synthetic dataset.

    Steps:
    - load data from data/churn_synthetic.csv,
    - split into train/validation,
    - train a RandomForestClassifier,
    - print metrics and feature importances,
    - save ROC curve to figures/roc_curve.png.
    """
    path = Path("data/churn_synthetic.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Data file not found. Run src/generate_data.py (or `python -m src.generate_data`) first."
        )

    df = pd.read_csv(path)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print_classification_metrics(y_val, y_pred, y_prob)
    plot_roc_curve(y_val, y_prob)

    print("\nTop feature importances:")
    for name, imp in sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"{name}: {imp:.3f}")


if __name__ == "__main__":
    main()
