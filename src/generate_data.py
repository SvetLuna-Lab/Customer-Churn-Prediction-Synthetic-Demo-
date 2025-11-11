from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_churn(n: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer churn dataset.

    Features
    --------
    age : int
        Customer age.
    monthly_fee : float
        Monthly subscription fee.
    tenure_months : int
        How many months the customer has been with the company.
    support_tickets : int
        Number of support tickets opened.
    late_payments : int
        Number of late payments.

    Target
    ------
    churn : {0, 1}
        1 means the customer is predicted to churn.

    The logits are crafted so that:
    - cheaper plans churn more often,
    - longer tenure reduces churn,
    - more support tickets and late payments increase churn.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n)
    monthly_fee = rng.uniform(5.0, 150.0, size=n)
    tenure_months = rng.integers(1, 120, size=n)
    support_tickets = rng.poisson(1.5, size=n)
    late_payments = rng.poisson(0.7, size=n)

    # Logistic function for churn probability
    z = (
        -2.0
        + 0.015 * (150.0 - monthly_fee)  # cheaper plans churn more
        - 0.02 * tenure_months           # longer tenure reduces churn
        + 0.4 * support_tickets
        + 0.6 * late_payments
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    churn = rng.binomial(1, prob, size=n)

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_fee": monthly_fee,
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "churn": churn,
        }
    )
    return df


def main() -> None:
    df = generate_synthetic_churn()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "churn_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
