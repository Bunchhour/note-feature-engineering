# Outlier Handling: Complete Learning Guide

> **TL;DR Quick Decision:** Need to handle outliers?
> - **Fast/Simple but lose data:** Trimming (great when outliers are few)
> - **Small dataset/Can't lose rows:** Capping / Winsorization (replaces extreme values)
> - **Mathematical Boundaries:** Normal (Gaussian), IQR (Skewed), Quantiles
> - **Domain Knowledge:** Arbitrary Capping (Hardcoded limits)

## Quick Decision Guide

| Your Data | Goal | Best Method | Why | Library |
|-----------|------|---|---|---|
| Large dataset, few outliers | Remove noise | Trimming | Fast, simple | pandas / Feature-engine |
| Small dataset, can't trim | Keep rows, reduce impact | Capping (Winsorizer) | Modifies value, keeps row | pandas / Feature-engine |
| Normal Distribution | Find natural edges | Gaussian ($\mu \pm 3\sigma$) | Mathematically sound | Feature-engine |
| Skewed Distribution | Find natural edges | IQR Proximity Rule | Standard for skewed data | Feature-engine |
| Domain-specific limits | Apply business logic | Arbitrary Capping | Hardcoded exact limits | Feature-engine |

---

## Part 1: What is an Outlier?

An outlier is a data point that is so significantly different from the rest of your dataset that you suspect it might not belong to the same underlying population.

Handling outliers is a critical step in building robust machine learning models, as algorithms like Linear Regression and AdaBoost are highly sensitive to extreme values.

---

## Part 2: Techniques for Handling Outliers

* **Trimming:** Removing the outlier from the dataset entirely.
    * *Advantage:* Extremely fast and simple.
    * *Disadvantage:* If your dataset has outliers across multiple variables, trimming them all might result in deleting a massive portion of your overall data.
* **Censoring (Capping / Winsorization):** Setting a maximum and minimum limit for your variable. Any value above or below these limits is replaced by the limit itself.
    * *Advantage:* No data points are removed; fast to implement.
    * *Disadvantage:* Distorts the statistical distribution of the variable and can alter its relationship with other variables.
* **Missing Value Imputation:** Treating the outlier as a missing value (e.g., `NaN`) and using standard imputation techniques to replace it.
* **Discretization:** Sorting continuous data into bins, meaning extreme outliers naturally get grouped into the highest or lowest bins alongside non-outlier edge cases.

### Domain Knowledge Disclaimer (Crucial)
**Context matters.** In medical diagnostics, anomaly detection, or fraud detection, the outliers are rarely "errors" or "noise." They are often the exact signals you are trying to detect (e.g., a malignant tumor or a fraudulent transaction). 
* **Rule of Thumb:** Never blindly cap or trim outliers without understanding the business or scientific context of the data. If the outlier represents the target condition, capping it destroys the most valuable information in your dataset.

---

## Part 3: Detecting Outliers (Theoretical Boundaries)

The most challenging part of outlier engineering is identifying them accurately. The method you choose depends on the distribution of your variable.

**Key Rule:** Always calculate limits on the Training Set! The mathematical limits must be calculated exclusively from the training set to prevent data leakage. You calculate the limits on the train set, and apply those limits to both the train and test sets.

### 3.1 Normal Distribution (Gaussian)
If your data follows a bell curve, roughly 99% of observations fall within three standard deviations of the mean.
* **Boundaries:** $\mu \pm 3\sigma$
* **Trimming/Capping:** Remove or replace points > $\mu + 3\sigma$ and < $\mu - 3\sigma$.

### 3.2 Interquantile Range (IQR) Proximity Rule
If your data is skewed or not normally distributed, the IQR rule is the standard approach.
* **IQR Calculation:** $IQR = Q_3 - Q_1$
* **Lower Boundary:** $Q_1 - 1.5 \times IQR$
* **Upper Boundary:** $Q_3 + 1.5 \times IQR$

### 3.3 Quantiles Rule
A simpler approach based purely on percentiles.
* **Boundaries:** 5th percentile (bottom 5%) and 95th percentile (top 5%).
* **Implementation:** Cap or trim below the 5th percentile and above the 95th percentile.

---

## Part 4: Arbitrary Capping

Sometimes, mathematical formulas don't make sense for a specific dataset. If you have strong **domain knowledge** about your data, you can manually define the strict minimum or maximum limits. For example, capping ticket fare at $200 or age at 50 based on business rules.

---

## Part 5: Implementation by Library

### 5.1 Pandas (Trimming & Capping)

```python
import pandas as pd

# Method A: Calculate Limits (IQR Example)
def find_limits_iqr(df, variable, multiplier=1.5):
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    return q1 - (multiplier * iqr), q3 + (multiplier * iqr)

lower_limit, upper_limit = find_limits_iqr(X_train, 'median_income', multiplier=3)

# 1. Trimming using .ge() and .le()
outliers_removed_train = X_train[
    X_train['median_income'].ge(lower_limit) & 
    X_train['median_income'].le(upper_limit)
]

# 2. Capping using .clip()
X_train['median_income'] = X_train['median_income'].clip(
    lower=lower_limit, 
    upper=upper_limit
)
```

### 5.2 Feature-engine (Trimming)

```python
from feature_engine.outliers import OutlierTrimmer

trimmer_iqr = OutlierTrimmer(
    capping_method='iqr',
    tail='both',
    fold=1.5, 
    variables=['median_income', 'population']
)

# Learn boundaries and trim
trimmer_iqr.fit(X_train)
X_train_trimmed = trimmer_iqr.transform(X_train)
X_test_trimmed = trimmer_iqr.transform(X_test)
```

### 5.3 Feature-engine (Winsorizer / Capping)

```python
from feature_engine.outliers import Winsorizer

winsorizer_gaussian = Winsorizer(
    capping_method='gaussian',
    tail='both',
    fold=3,
    variables=['worst smoothness']
)

# Learn boundaries and cap
winsorizer_gaussian.fit(X_train)
X_train_capped = winsorizer_gaussian.transform(X_train)
X_test_capped = winsorizer_gaussian.transform(X_test)
```

### 5.4 Feature-engine (Arbitrary Capping)

```python
from feature_engine.outliers import ArbitraryOutlierCapper

capper_both = ArbitraryOutlierCapper(
    max_capping_dict={'age': 50, 'fare': 200},
    min_capping_dict={'age': 10, 'fare': 100}
)

capper_both.fit(X_train)
X_train_capped = capper_both.transform(X_train)
X_test_capped = capper_both.transform(X_test)
```
 