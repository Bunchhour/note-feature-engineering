# Discretization Methods: Complete Learning Guide

> **TL;DR Quick Decision:** Need to convert continuous to categorical?
> - **Speed/Simple:** Equal-Width or Equal-Frequency
> - **Better Quality:** Supervised (Decision Tree, Chi-Merge)
> - **Domain Knowledge:** Arbitrary intervals  
> - **Clustering-based:** K-Means

## Quick Decision Guide

| Your Data | Goal | Best Method | Why | Library |
|-----------|------|---|---|---|
| Age (0-100) | Interpretable bins | Equal-Width or Arbitrary | Simple, business-friendly | pandas |
| Income (skewed) | Balanced representation | Equal-Frequency | Equal distribution | pandas |
| Any continuous + Target | Predictive power | Decision Tree / Chi-Merge | Supervised, optimal splits | Scikit-learn |
| Clustering task | Find natural groups | K-Means | Data-driven grouping | Scikit-learn |
| Any continuous | Simple threshold | Binarization | Just two categories | Scikit-learn |
| Messy data, mixed types | Production pipeline | Feature-engine | Auto-handling | Feature-engine |

---


## Part 1: Why Discretize Variables?

### 1.1 Core Objectives

Discretization converts continuous variables into discrete categorical variables. Key goals:

- **Interpretability:** More understandable results
- **Handling Non-linearity:** Captures non-linear relationships
- **Speed:** Faster processing  
- **Outlier Robustness:** Extreme values impact reduced
- **Business Requirements:** Some domains require categories

### 1.2 Advantages

| Advantage | Explanation |
|-----------|---|
| **Interpretability** | Business can understand results immediately |
| **Non-linearity Capture** | Creates step functions automatically |
| **Robustness to Outliers** | Extreme values treated like boundary values |
| **Reduced Variance** | Smoother predictions, less overfitting |

---


## Part 2: When to Discretize

### 2.1 By Model Type

- **Linear Models:** YES (if relationship is non-linear)
- **Tree-Based:** NO (trees partition naturally)  
- **Neural Networks:** NO (learn better from continuous)

### 2.2 By Data Characteristics

| Scenario | Should Discretize? |
|----------|---|
| Highly skewed data | YES |
| Few observations | NO |
| Categorical target | MAYBE |
| Continuous target | NO |

---


## Part 3: Theoretical Foundation

### 3.1 Unsupervised vs Supervised

**Unsupervised:** Equal-Width, Equal-Frequency, K-Means
- Ignore target variable, simple and interpretable

**Supervised:** Decision Tree, Chi-Merge
- Use target, optimize for prediction

### 3.2 Strategy Comparison

`
Data: [1, 2, 5, 10, 15, 20, 50, 100, 500]

EQUAL-WIDTH: width = 166
- Bin 1: 1-167 [many values]
- Bin 2: 167-334 [empty]
- Bin 3: 334-500 [1 value]
- Problem: Imbalanced

EQUAL-FREQUENCY: 3 values per bin
- Bin 1: [1, 2, 5]
- Bin 2: [10, 15, 20]
- Bin 3: [50, 100, 500]
- Benefit: Balanced
`

---


## Part 4: All Discretization Methods

### 4.1 Equal-Width Binning

**Formula:** width = (max - min) / n_bins
**Pros:** Simple, fast, interpretable
**Cons:** Imbalanced bins, outlier-sensitive

### 4.2 Equal-Frequency Binning

**Each bin has approximately n_samples / n_bins values**
**Pros:** Balanced, handles skewed data, outlier-robust
**Cons:** Arbitrary boundaries

### 4.3 Arbitrary Binning

**User-defined intervals**

\\python
pd.cut(age, bins=[0, 18, 35, 50, 65, 100],
       labels=['Child', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
\
### 4.4 K-Means Binning

**Use K-Means clustering to find optimal bin centers**
**Pros:** Data-driven, finds natural groupings
**Cons:** Expensive, requires choosing k

### 4.5 Decision Tree Binning (Supervised)

**Extract split thresholds from a decision tree**

\\python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)
\
### 4.6 Binarization

**Create only 2 bins (x > threshold: 1, else 0)**
**Pros:** Simplest, fastest
**Cons:** Loses all nuance

### 4.7 Chi-Merge

**Recursively merge adjacent bins using chi-squared test**
**Pros:** Statistically grounded
**Cons:** Complex, expensive

---


## Part 5: Implementation by Library

### 5.1 Pandas

\\python
import pandas as pd
data = pd.Series([1, 5, 10, 15, 20, 25, 30, 50, 100])

# Equal-Width
binned = pd.cut(data, bins=3)

# Equal-Frequency  
binned = pd.qcut(data, q=3)

# Custom with labels
binned = pd.cut(data, bins=[0, 20, 50, 100],
                labels=['Low', 'Mid', 'High'])
\
### 5.2 Scikit-learn

\\python
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

X = np.array([[1], [5], [10], [50], [100]])

kbd = KBinsDiscretizer(n_bins=3, strategy='quantile')
X_binned = kbd.fit_transform(X)

# Decision Tree supervised
from sklearn.tree import DecisionTreeClassifier
y = np.array([0, 0, 0, 1, 1])
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)
\
### 5.3 Feature-engine

\\python
from feature_engine.discretisation import EqualFrequencyDiscretiser
import pandas as pd

X = pd.DataFrame({'age': [1, 5, 10, 50, 100]})

efd = EqualFrequencyDiscretiser(variables=['age'], q=3)
X_disc = efd.fit_transform(X)
\
---


## Part 6: Practical Examples

### 6.1 Age Binning

\\python
df['age_cat'] = pd.cut(df['age'],
                       bins=[0, 25, 35, 50, 65, 100],
                       labels=['Gen Z', 'Millennials', 'Gen X', 'Boomers', 'Silent'])

df['age_freq'] = pd.qcut(df['age'], q=5)
\
### 6.2 Income (Skewed Data)

\\python
# Equal-Width: BAD for skewed
df['income_width'] = pd.cut(df['income'], bins=4)

# Equal-Frequency: GOOD for skewed
df['income_freq'] = pd.qcut(df['income'], q=4)
\
### 6.3 Supervised Binning

\\python
from sklearn.tree import DecisionTreeClassifier

X = df[['tenure_months', 'monthly_charges']].values
y = df['churn'].values

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)
\
---


## Part 7: Library Comparison

| Feature | pandas | Scikit-learn | Feature-engine |
|---------|--------|---|---|
| Equal-Width | ? pd.cut() | ? | ? |
| Equal-Frequency | ? pd.qcut() | ? | ? |
| K-Means | ? | ? | ? |
| Decision Tree | ? | ? | ? |
| Chi-Merge | ? | ? | ? |
| Returns DataFrame | ? | ? | ? |
| Multiple columns | ? | ? | ? |
| Production-ready | ? | ? | ?? |

---


## Part 8: Cheat Sheets & Quick Reference

### 8.1 Decision Tree

\1. Tree-based model? ? Skip
2. No target? ? Equal-Frequency
3. Have target? ? Decision Tree or Chi-Merge
4. Highly skewed? ? Equal-Frequency
5. Business rules? ? Arbitrary
\
### 8.2 One-Liners

\\python
# Equal-Width
pd.cut(data, bins=5)

# Equal-Frequency
pd.qcut(data, q=4)

# Custom
pd.cut(data, bins=[0, 25, 50, 100], labels=['Low', 'Mid', 'High'])
\
### 8.3 Troubleshooting

| Problem | Solution |
|---------|----------|
| Imbalanced bins | Use Equal-Frequency |
| Ugly boundaries | Use Arbitrary |
| Hurts performance | Use fewer bins |

---


## Part 9: Professional Workflow

### 9.1 Step-by-Step

1. Analyze distribution
2. Choose method (business or supervised)
3. Set parameters (number of bins)
4. Validate (visuals, performance)
5. Deploy (document boundaries!)
6. Monitor

### 9.2 Code Template

\\python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('discretizer', KBinsDiscretizer(n_bins=3, strategy='quantile')),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
\
### 9.3 Common Pitfalls

? Equal-Width on skewed data
? Fitting on test data
? Too many bins
? Forgetting edge cases

---


## Part 10: Summary & Checklist

### 10.1 Key Takeaways

1. Discretization: continuous ? categorical
2. Unsupervised: Equal-Width, Equal-Frequency, K-Means
3. Supervised: Decision Tree, Chi-Merge
4. Always validate visually
5. Skip for tree-based models

### 10.2 Method Comparison

| Method | Speed | Interpretability | Robustness | Power |
|--------|-------|---|---|---|
| Equal-Width | ??? | ??? | ? | ? |
| Equal-Frequency | ?? | ?? | ??? | ?? |
| Decision Tree | ? | ??? | ??? | ??? |

### 10.3 Checklist

- [ ] Do I need to discretize?
- [ ] Will it improve my model?
- [ ] Skewed data? Use Equal-Frequency
- [ ] Have target? Use supervised method
- [ ] Documented boundaries?
- [ ] Fitted only on training data?
- [ ] Validated on test data?

---

## References

- Scikit-learn: KBinsDiscretizer
- Feature-engine: Discretisation transformers
- Witten & Frank: Discretization and Permutation of Continuous Variables

