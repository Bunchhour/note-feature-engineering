from pathlib import Path

file_path = Path("9_ Variable Transformation.md")

sections = []

# HEADER
sections.append("""# Variable Transformation: Complete Student Learning Guide

> Quick Decision Guide: Need to transform your data? Use this flow chart.

## Quick Decision Flowchart

- Do you have a Linear Model? NO -> Skip | YES -> Continue
- Is data skewed? NO -> Skip | YES -> Continue
- Know transformation? YES -> Use NumPy | NO -> Continue
- Has negatives/zeros? NO -> Box-Cox | YES -> Yeo-Johnson

| Scenario | Use This | Why |
|----------|----------|-----|
| Linear/Logistic with skewed | Box-Cox or Yeo-Johnson | Normalizes residuals |
| Tree-based models | Don't transform | Invariant to transforms |
| Neural Networks | Standardize only | Care about scale |
| Learning mode | SciPy directly | Understand the math |
| Production pipeline | Scikit-learn/Feature-engine | Reproducible |
| Messy real data | Feature-engine | Auto-detects types |

---
""")

# PART 1
sections.append("""## Part 1: Why Transform Variables?

### 1.1 The Primary Goal
- Normalize Distributions: Convert skewed data to Gaussian
- Satisfy Model Assumptions: Enable linear models correctly  
- Improve Model Performance: Capture relationships accurately

### 1.2 Key Assumptions of Linear Models

Linear models assume:
1. **Linearity** - Relationship should be a straight line
2. **Normality of Errors** - Residuals follow normal distribution
3. **Homoscedasticity** - Constant error variance across levels

### 1.3 When to Transform

Transform for:
- Linear Regression
- Logistic Regression (sometimes)
- ANOVA and statistical tests
- GLM (Generalized Linear Models)

Don't transform for:
- Tree-based models (invariant to monotone transformations)
- Distance-based models (need scaling, not distribution)
- Neural Networks (care about scale, not Gaussian)

### 1.4 The Golden Rule
Always verify transformations visually. Uniform or bimodal distributions often won't become Gaussian.

---
""")

# PART 2
sections.append("""## Part 2: Mathematical Transformations Reference

### 2.1 Seven Core Transformations

| Transformation | Formula | Best For | Constraints | Example |
|---|---|---|---|---|
| Logarithmic | x' = log(x) | Right-skewed data | x > 0 | Income, prices |
| Reciprocal | x' = 1/x | Ratios and rates | x != 0 | Density, speed |
| Square Root | x' = sqrt(x) | Count data | x >= 0 | Visits, items |
| Arcsin | x' = arcsin(sqrt(x)) | Proportions | 0 <= x <= 1 | Percentages |
| Exponential | x' = x^2 | Left-skewed | x > 0 | Rare |
| Box-Cox | (x^λ - 1)/λ | Auto, positive only | x > 0 | General use |
| Yeo-Johnson | Piecewise | Auto, any values | Any real | Production |

### 2.2 Power Transformation (General)

Formula: x' = x^λ

How to choose λ:
- λ < 1: Right-skewed data (e.g., sqrt is λ=0.5)
- λ = 1: No transformation
- λ > 1: Left-skewed data (e.g., square is λ=2)

### 2.3 Box-Cox Transformation

- Automatically finds optimal λ using MLE
- Requires strictly positive data (x > 0)
- Returns transformed data AND optimal λ
- Use scipy.stats.boxcox() or PowerTransformer

### 2.4 Yeo-Johnson Transformation

- Box-Cox alternative that handles negatives and zeros
- Better for real-world messy data
- Automatically finds optimal λ
- Use scipy.stats.yeojohnson() or PowerTransformer

---
""")

# PART 3
sections.append("""## Part 3: Implementation Guides by Library

### 3.1 NumPy & SciPy: Manual Transformations

```python
import numpy as np
from scipy.stats import boxcox, yeojohnson

# Sample data
data = np.array([1, 2, 5, 10, 50, 100, 500])

# LOGARITHMIC
log_transformed = np.log(data)

# RECIPROCAL
reciprocal_transformed = 1 / data

# SQUARE ROOT
sqrt_transformed = np.sqrt(data)

# EXPONENTIAL
exp_transformed = data ** 2

# POWER with custom lambda
lambda_value = 0.5
power_transformed = data ** lambda_value

# BOX-COX (positive data only)
transformed_data, lambda_val = boxcox(data)
print(f"Optimal lambda: {lambda_val:.4f}")

# YEO-JOHNSON (any data)
data_mixed = np.array([-5, -2, 0, 1, 5, 10, 50])
transformed_data, lambda_val = yeojohnson(data_mixed)
```

### 3.2 Scikit-learn: PowerTransformer in Pipelines

```python
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# Box-Cox Method
transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X_train)
lambdas = transformer.lambdas_
X_new_transformed = transformer.transform(X_new)

# Yeo-Johnson Method  
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X_train)

# In a Pipeline
pipe = Pipeline([
    ('transformer', PowerTransformer(method='yeo-johnson')),
    ('model', LinearRegression())
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_new)
```

### 3.3 Feature-engine: Production-Ready

```python
from feature_engine.transformation import YeoJohnsonTransformer, BoxCoxTransformer
import pandas as pd

# Yeo-Johnson auto-detect
transformer = YeoJohnsonTransformer()
transformer.fit(X_train)
X_transformed = transformer.transform(X_train)
print(transformer.lambda_dict_)

# Box-Cox with specific variables
transformer = BoxCoxTransformer(variables=['feature1', 'feature2'])
transformer.fit(X_train)
X_transformed = transformer.transform(X_train)

# Key advantages:
# - Auto-detects numerical columns
# - Ignores text/categorical automatically
# - Returns DataFrame (preserves column names)
# - Production-grade
```

---
""")

# PART 4
sections.append("""## Part 4: Practical Examples and Cheat Sheets

### 4.1 Scenario 1: Single Column Exploration

```python
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy import stats

# Your skewed data
salary = np.array([25000, 30000, 35000, 40000, 50000, 100000, 500000])

# Try different transformations
log_sal = np.log(salary)
sqrt_sal = np.sqrt(salary)
box_sal, lambda_bc = boxcox(salary)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(salary, bins=10)
axes[0, 0].set_title('Original')
axes[0, 1].hist(log_sal, bins=10)
axes[0, 1].set_title('Log')
axes[1, 0].hist(sqrt_sal, bins=10)
axes[1, 0].set_title('Sqrt')
axes[1, 1].hist(box_sal, bins=10)
axes[1, 1].set_title(f'Box-Cox (lambda={lambda_bc:.2f})')
plt.show()

# Q-Q plots for normality
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats.probplot(salary, dist='norm', plot=axes[0, 0])
stats.probplot(log_sal, dist='norm', plot=axes[0, 1])
stats.probplot(sqrt_sal, dist='norm', plot=axes[1, 0])
stats.probplot(box_sal, dist='norm', plot=axes[1, 1])
plt.show()
```

### 4.2 Scenario 2: DataFrame with Multiple Columns

```python
from feature_engine.transformation import YeoJohnsonTransformer

# Dataset with mixed types
data = pd.DataFrame({
    'age': [18, 25, 35, 45, 55, 80],
    'income': [20000, 50000, 75000, 150000, 500000, 2000000],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank']
})

# Auto-detect and transform
transformer = YeoJohnsonTransformer()
data_transformed = transformer.fit_transform(data)
print(transformer.lambda_dict_)
```

### 4.3 Scenario 3: Train/Test Pipeline

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = Pipeline([
    ('transform', PowerTransformer(method='yeo-johnson')),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print(f"Train R2: {train_score:.4f}")
print(f"Test R2: {test_score:.4f}")
```

### 4.4 Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| ValueError: non-positive values | Box-Cox with negatives/zeros | Use Yeo-Johnson |
| NaN in results | Invalid domain | Check constraints |
| Column names lost | Scikit-learn returns array | Use Feature-engine |
| Transformation made worse | Wrong choice | Try different or skip |
| Different train vs test | Refit on test | Fit ONLY on training |
| Memory error | Too much data | Use feature-engine |

### 4.5 Decision Tree

1. Linear model? NO -> Stop | YES -> Continue
2. Data skewed? NO -> Stop | YES -> Continue
3. Know transformation? YES -> Use NumPy | NO -> Continue
4. Has negatives/zeros? NO -> Box-Cox | YES -> Yeo-Johnson
5. Learning/exploration? YES -> SciPy | NO -> Continue
6. Scikit-learn pipeline? YES -> PowerTransformer | NO -> Feature-engine

---
""")

# PART 5
sections.append("""## Part 5: Library Comparison and Selection

### 5.1 Feature Comparison Matrix

| Feature | NumPy | SciPy | Scikit-learn | Feature-engine |
|---------|-------|-------|---|---|
| Manual transformations | Excellent | Good | No | No |
| Box-Cox | No | Yes | Yes | Yes |
| Yeo-Johnson | No | Yes | Yes | Yes |
| Auto-detect columns | No | No | No | Yes |
| Ignore non-numeric | No | No | No | Yes |
| Returns DataFrame | No | No | No | Yes |
| Preserves names | No | No | No | Yes |
| Pipeline-ready | No | No | Yes | Yes |
| Lambda dictionary | No | Yes | Yes | Yes |
| Production-grade | No | Development | Production | Production |

### 5.2 When to Use Each Library

**NumPy/SciPy:** Learning and experimentation
- Best for understanding the math
- Use for testing single variables
- Visual diagnostics

**Scikit-learn:** Standard ML pipelines
- Best for clean data with known structure
- Use in production if data is clean
- Integrates with rest of sklearn

**Feature-engine:** Production with messy data
- Best for real-world raw data
- Auto-detects numerical columns
- Handles mixed types automatically
- Returns DataFrame with names preserved

---
""")

# PART 6
sections.append("""## Part 6: Complete Summary

### 6.1 The "Why" Recap
- Transform skewed data into Gaussian distributions
- Enable linear models to meet their assumptions
- Improve model performance and interpretability
- Not needed for tree-based or distance-based models

### 6.2 The "What" Recap
Seven core transformations:
1. Logarithmic - Right-skewed data
2. Reciprocal - Ratio data
3. Square Root - Count data
4. Arcsin - Proportions (0-1)
5. Power (general) - Any skewness with tuned exponent
6. Box-Cox - Automated for positive-only data
7. Yeo-Johnson - Automated for any real values

### 6.3 The "How" Recap

For Learning: NumPy/SciPy
```python
import numpy as np
from scipy.stats import boxcox, yeojohnson

log_data = np.log(data)
boxcox_data, lambda_bc = boxcox(data)
yeojohnson_data, lambda_yj = yeojohnson(data)
```

For Pipelines: Scikit-learn PowerTransformer
```python
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('transform', PowerTransformer(method='yeo-johnson')),
    ('model', YourModel())
])
```

For Production: Feature-engine
```python
from feature_engine.transformation import YeoJohnsonTransformer

transformer = YeoJohnsonTransformer()
transformed = transformer.fit_transform(raw_data)
```

### 6.4 Professional Workflow
1. Load raw data
2. Visualize distributions
3. Decide: Linear model needed? NO -> Use as-is | YES -> Continue
4. Is data skewed? NO -> Use as-is | YES -> Continue
5. Try transformation (Box-Cox or Yeo-Johnson)
6. Verify with Q-Q plots and visualization
7. Build pipeline with transformation + model
8. Fit ONLY on training data
9. Evaluate on test data

### 6.5 Final Checklist
- [ ] Understand when transformations needed (linear models only)
- [ ] Know which transformation for each data shape
- [ ] Can manually apply with NumPy
- [ ] Can use Box-Cox/Yeo-Johnson for automation
- [ ] Can implement in scikit-learn pipelines
- [ ] Can use Feature-engine for production
- [ ] Always verify transformations visually
- [ ] Fit transformers on training data only
- [ ] Preserve parameters for test/production

---
"""
)

content = "\n".join(sections)
file_path.write_text(content, encoding='utf-8')
lines = len(content.split("\n"))
print(f"File restructured successfully!")
print(f"New file line count: {lines} lines")
print(f"Saved to: 9_ Variable Transformation.md")
