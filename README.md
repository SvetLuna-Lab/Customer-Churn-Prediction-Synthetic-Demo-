# Customer Churn Prediction (Synthetic Demo)

This repository contains a small, self-contained **customer churn prediction** demo:

- synthetic tabular dataset with basic customer features,
- binary churn label (`0` = stays, `1` = churns),
- a simple ML pipeline using `RandomForestClassifier`,
- evaluation metrics: accuracy, precision, recall, ROC-AUC,
- feature importance analysis and ROC curve plot.

The goal is to demonstrate a clean, readable ML workflow that could be extended
to real business data in telecom, SaaS subscriptions, banking, etc.

---

## Project structure

```text
customer-churn-prediction/
├─ data/
│  └─ .gitkeep                # folder for generated churn_synthetic.csv
├─ figures/
│  └─ .gitkeep                # ROC curve images will be saved here
├─ src/
│  ├─ __init__.py
│  ├─ generate_data.py        # synthetic data generator
│  ├─ train_model.py          # training + evaluation script
│  └─ eval_utils.py           # metrics printing and ROC plotting
├─ README.md
├─ requirements.txt
└─ .gitignore


Installation

Create and activate a virtual environment (optional but recommended), then:

pip install -r requirements.txt



Usage
1. Generate synthetic data

python -m src.generate_data
# or
python src/generate_data.py


This will create:

data/churn_synthetic.csv


2. Train the model and evaluate

python -m src.train_model
# or
python src/train_model.py


The script will:

split the data into train/validation,

train a RandomForestClassifier,

print accuracy, precision, recall, ROC-AUC,

print feature importances,

save figures/roc_curve.png.

Example console output (will vary slightly due to randomness):

Accuracy:  0.875
Precision: 0.842
Recall:    0.801
ROC-AUC:   0.912

Top feature importances:
late_payments:    0.410
support_tickets:  0.270
tenure_months:    0.180
monthly_fee:      0.100
age:              0.040



How the synthetic data is constructed

We simulate a very simple business logic:

Cheaper plans churn more often
(monthly_fee is inversely related to churn risk),

Longer tenure reduces churn
(tenure_months has a negative effect on the logit),

More support tickets and late payments increase churn.

The target churn is generated via a logistic function:

z = -2.0
    + 0.015 * (150 - monthly_fee)
    - 0.02 * tenure_months
    + 0.4  * support_tickets
    + 0.6  * late_payments
prob  = 1 / (1 + exp(-z))
churn ~ Bernoulli(prob)



This is, of course, a simplified toy model, but it is useful for:

demonstrating end-to-end ML flow,

checking that metrics and feature importances behave as expected,

practicing code structure without dealing with messy real-world data.


Possible extensions

If you want to evolve this project further, natural next steps could be:

add categorical features (tariff type, region, channel),

compare more models (Logistic Regression, XGBoost, etc.),

introduce class imbalance and experiment with weighting,

log metrics to a file or experiment tracker,

wrap the model into a small prediction API.

Even in this small synthetic setting, the core idea is the same as in real
commercial projects: understand which behaviors drive churn and how well
your model can detect them.
