# Dataset: UCI German Credit

Source: UCI Machine Learning Repository, Statlog (German Credit Data).

- `german_credit.data` is the original whitespace-separated dataset.
- It contains 1,000 historical credit applicants and 20 input features.
- The original last-column label is `1` for good credit risk and `2` for bad credit risk.
- The training script converts the label to `0` for good risk and `1` for bad risk so the positive class is easy to evaluate.

The source data uses compact category codes such as `A11`. The script treats these as categories and encodes them automatically; it does not invent a numeric ranking for them.
