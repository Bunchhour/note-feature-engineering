# Alternative Imputation Methods: Beyond the Basics
**Advanced Techniques for Handling Missing Data**

---

## Table of Contents
1. [Quick Method Comparison](#quick-comparison)
2. [Complete Case Analysis (CCA)](#cca)
3. [End of Tail Imputation](#end-of-tail)
4. [Random Sample Imputation](#random-sample)
5. [Group-Based Mean/Median](#group-based)
6. [Implementation with Tools](#implementation)
7. [Method Selection Guide](#selection-guide)

---

## Quick Method Comparison {#quick-comparison}

| **Method** | **Numerical** | **Categorical** | **Pros** | **Cons** |
|---|---|---|---|---|
| **Complete Case Analysis** | ✅ | ✅ | Simple, no artificial data | Loses rows, production issues |
| **End of Tail** | ✅ | ❌ | Automated flagging, obvious outlier | Distorts distribution, increases variance |
| **Random Sample** | ✅ | ✅ | Preserves distribution perfectly | Memory intensive, randomness in production |
| **Group-Based Mean/Median** | ✅ | ❌ | More accurate estimates | Requires grouping variable, complex |

---

## Complete Case Analysis (CCA) {#cca}

### What is it?

**Complete Case Analysis** (also called "list-wise deletion") is the simplest approach: **delete any row that contains ANY missing value**.

**What counts as a "complete case"?** A row that has data present in **every single variable**.

### When to Use It

✅ **Use when:**
- Data is **Missing Completely At Random** (MCAR)
- Very **small amount of data is missing** (<1-5%)
- **Not deploying to production** (or willing to handle missing data in production differently)
- You can afford to lose rows without introducing bias

❌ **Avoid when:**
- Large percentage of data is missing
- Data is MAR or MNAR (systematic missingness)
- Will deploy to production (missing data will cause failures)

### Pros and Cons

**Pros:**
✅ Extremely simple - just delete rows  
✅ No artificial data injected  
✅ Preserves original distributions (if truly MCAR)  
✅ No parameters to learn/store  

**Cons:**
❌ Can lose massive amounts of data quickly  
❌ Introduces bias if data is not truly MCAR  
❌ Production problem: can't handle new missing data  
❌ One column with lots of missing values can delete entire dataset  

### The MCAR Assumption: Why It Matters

CCA assumes data is **Missing Completely At Random (MCAR)**. If this is true:
- Deleted rows are just a random sample
- Original distributions remain intact
- No bias introduced

If data is **NOT MCAR** (e.g., rich people don't report income):
- Deleting creates a biased sample
- Model learns wrong patterns
- Results are unreliable

### Implementation

**Pandas:**
```python
# Simple approach - risky!
X_clean = X.dropna()

# Better - target specific columns
X_clean = X.dropna(subset=['age', 'price'])

# Controlled - keep rows with at least some data
X_clean = X.dropna(thresh=5)  # Keep rows with 5+ non-missing values
```

**Feature-engine:**
```python
from feature_engine.imputation import DropMissingData

# Method 1: Target specific variables
cca = DropMissingData(variables=['age', 'price'])
cca.fit(X_train)
X_train_clean = cca.transform(X_train)

# Method 2: Auto-detect columns with missing data
cca = DropMissingData(missing_only=True)

# Method 3: Threshold approach
cca = DropMissingData(threshold=0.75)  # Keep rows with 75%+ non-missing data
```

**Production Tool - Identify dropped rows:**
```python
dropped_rows = cca.return_na_data(X_test)  # See which rows would be removed
```

---

## End of Tail Imputation {#end-of-tail}

### What is it?

**End of Tail Imputation** is an automated version of arbitrary value imputation. Instead of guessing a number, you calculate a value that sits at the **extreme edge (tail) of the distribution**.

**Why "tail"?** The idea is to pick a mathematically extreme value that clearly stands out as "not real" but is based on the actual data.

### How It Works

Calculate an extreme value using **only the training set**, then apply it to all missing data.

#### Method 1: Gaussian Approximation (Normal Distributions)

For normally distributed data, approximately 99% falls within 3 standard deviations of the mean.

**Formula:** $ \text{Tail} = \mu + 3\sigma $ (or $ \mu - 3\sigma $ for lower tail)

**Python:**
```python
import numpy as np

# Calculate from training set only
mean = X_train['age'].mean()
std = X_train['age'].std()
tail_value = mean + (3 * std)

# Apply to both sets
imputation_dict = {'age': tail_value}
X_train.fillna(imputation_dict, inplace=True)
X_test.fillna(imputation_dict, inplace=True)

# Or with Feature-engine
from feature_engine.imputation import EndTailImputer
imputer = EndTailImputer(distribution='normal')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

#### Method 2: IQR-Based (Skewed Distributions)

For skewed data, use percentiles instead of mean/std:

**Formula:** $ \text{Tail} = Q_3 + 3 \times \text{IQR} $ (or $ Q_1 - 3 \times \text{IQR} $ for lower tail)

**Python:**
```python
Q1 = X_train['price'].quantile(0.25)
Q3 = X_train['price'].quantile(0.75)
IQR = Q3 - Q1
tail_value = Q3 + (3 * IQR)

# Apply
imputation_dict = {'price': tail_value}
X_train.fillna(imputation_dict, inplace=True)
X_test.fillna(imputation_dict, inplace=True)

# Or with Feature-engine (automatically detects distribution)
imputer = EndTailImputer(distribution='skewed')
```

#### Method 3: Maximum Value Multiplier (Simplest)

Just take the largest value and multiply it:

**Formula:** $ \text{Tail} = \max(x) \times 3 $

**Python:**
```python
tail_value = X_train['age'].max() * 3
imputation_dict = {'age': tail_value}
X_train.fillna(imputation_dict, inplace=True)
```

### Distribution Impact

```
BEFORE:                    AFTER (with Tail Imputation):
  |    ╱╲                    |        ╱╲
  |   ╱  ╲                   |       ╱  ╲
  |  ╱    ╲                  |  ╱╱  ╱    ╲
  |_╱______╲___        →     |_╱╱╱_╱______╲_____________
    Normal          Extreme spike at tail end
    range
```

### Pros and Cons

**Pros:**
✅ Automated (no manual arbitrary choice)  
✅ Creates obvious outlier flag  
✅ Works well for tree-based models  

**Cons:**
❌ Massive distribution distortion  
❌ Dramatically increases variance  
❌ Can mask or create false outliers  
❌ Breaks relationships between variables  

---

## Random Sample Imputation {#random-sample}

### What is it?

**Random Sample Imputation** fills missing values by randomly drawing from the **pool of existing values** in that variable.

**Simple idea:** If you have 10 missing ages, randomly pick 10 existing ages from non-missing data and use them.

**Works for:** Both numerical AND categorical variables

### The Core Assumption

Assumes data is **Missing At Random (MAR)**. The goal is to preserve the original distribution by sampling from it.

### Why It Works

Since you're pulling existing values, you perfectly preserve:
- Distribution shape ✅
- Variance ✅
- Percentiles ✅
- Outlier status ✅

### The Randomness Problem (Critical!)

**Problem:** Each time you run the script, a different random value is selected for the same missing observation.

**Scenario:**
```
Person A is missing age
Run 1: Randomly selected age = 35
Run 2: Randomly selected age = 42 (different!)
→ Same person, different predictions each time!
```

**Impact:**
- Unfair: Two people with same data get different predictions
- Unreliable: Same person gets different prediction each day
- Unacceptable in production!

### The Solution: Seeding by Observation ID

**Don't use a global random seed** (that would give everyone the same imputed value).

**Instead: Tie seed to a unique variable of that observation:**

```
Person A: has unique_id=25, use seed=25
→ Always selects same random age (e.g., 35)

Person B: has unique_id=73, use seed=73
→ Always selects same random age (e.g., 42)

Person C arrives (new, same as Person A):
unique_id=25 → uses seed=25 → gets age=35 (consistent!)
```

### Implementation

**Pandas (complex):**
```python
for var in vars_with_na:
    # Count missing
    train_na_count = X_train[var].isnull().sum()
    test_na_count = X_test[var].isnull().sum()
    
    # Sample from training pool only
    train_samples = X_train[var].dropna().sample(
        n=train_na_count, 
        random_state=0
    )
    test_samples = X_train[var].dropna().sample(
        n=test_na_count, 
        random_state=0
    )
    
    # Match indexes (critical step!)
    train_samples.index = X_train[X_train[var].isnull()].index
    test_samples.index = X_test[X_test[var].isnull()].index
    
    # Fill
    X_train.loc[X_train[var].isnull(), var] = train_samples
    X_test.loc[X_test[var].isnull(), var] = test_samples
```

**Feature-engine:**
```python
from feature_engine.imputation import RandomSampleImputer

imputer = RandomSampleImputer(random_state=0)
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

### Pros and Cons

**Pros:**
✅ Perfectly preserves distribution  
✅ Preserves variance and percentiles  
✅ Works for both numerical and categorical  
✅ No distribution assumptions needed  

**Cons:**
❌ Introduces randomness (need seeding fix)  
❌ Must store entire training dataset  
❌ Memory intensive in production  
❌ Can break correlations between variables  

---

## Group-Based Mean/Median {#group-based}

### What is it?

Instead of **one global mean** for everyone, calculate **different means for each group/category**.

**Setup:**
- **Target**: Numerical variable with missing data
- **Grouping Variable**: Categorical variable that splits the population

### Example

```
GLOBAL MEAN: Everyone gets average height = 70 inches

GROUP MEANS:
├─ Males: average height = 75 inches
└─ Females: average height = 65 inches

If person is missing height AND is Male → impute 75
If person is missing height AND is Female → impute 65
```

### When to Use

✅ **Use when:**
- Target variable has **different distributions by group**
- Group is **strongly predictive** of the target
- You have **strong domain knowledge**
- Categorical grouping variable is clean

❌ **Avoid when:**
- Groups have very **small sample sizes** (<30 observations)
- Only small differences between group means
- Missing values are **MNAR** (need arbitrary value instead)

### Implementation

**Pandas:**
```python
# Build nested dictionary of group means
imputation_dict = {}

for category in X_train['Gender'].unique():
    # Calculate mean within this group
    imputation_dict[category] = X_train[
        X_train['Gender'] == category
    ][vars_with_na].mean().to_dict()

# Result: {'Male': {'height': 75, 'weight': 190}, 
#          'Female': {'height': 65, 'weight': 140}}

# Apply to train set
for category in imputation_dict.keys():
    X_train.loc[X_train['Gender'] == category, vars_with_na] = \
        X_train[X_train['Gender'] == category][vars_with_na].fillna(
            imputation_dict[category]
        )

# Apply to test set (same learned parameters)
for category in imputation_dict.keys():
    X_test.loc[X_test['Gender'] == category, vars_with_na] = \
        X_test[X_test['Gender'] == category][vars_with_na].fillna(
            imputation_dict[category]
        )
```

**Feature-engine:**
```python
from feature_engine.imputation import MeanMedianImputer

imputer = MeanMedianImputer(
    imputation_method='median', 
    group_variables=['Gender']
)

imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

### Important Checks

**Before using, verify:**

```python
# Check sample size in each group
print(X_train['Gender'].value_counts())

# Verify distributions are different
print(X_train.groupby('Gender')['height'].mean())
print(X_train.groupby('Gender')['height'].std())
```

### Pros and Cons

**Pros:**
✅ More accurate than global mean  
✅ Respects group differences  
✅ Better for stratified populations  

**Cons:**
❌ Requires enough data per group  
❌ More complex logic  
❌ Requires including grouping variable in model  
❌ Still "blends in" imputed data (use Missing Indicator)  

---

## Implementation with Tools {#implementation}

### Tool Comparison

| **Task** | **Pandas** | **Scikit-learn** | **Feature-engine** |
|---|---|---|---|
| Complete Case | `dropna()` | ❌ | `DropMissingData()` |
| End of Tail | Manual calc | ❌ | `EndTailImputer()` |
| Random Sample | Complex loop | ❌ | `RandomSampleImputer()` |
| Group-Based | Manual loop | ❌ | `MeanMedianImputer()` with `group_variables` |

### Production-Ready Pipeline Example

```python
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    AddMissingIndicator,
    EndTailImputer,
    MeanMedianImputer
)

# Create pipeline
pipeline = Pipeline([
    # 1. Add missing indicators first (before imputing)
    ('missing_indicator', AddMissingIndicator(missing_only=True)),
    
    # 2. Use end-of-tail for skewed variables
    ('tail_imputer', EndTailImputer(
        distribution='skewed',
        variables=['price', 'lot_size']
    )),
    
    # 3. Use group-based for others
    ('group_imputer', MeanMedianImputer(
        imputation_method='median',
        group_variables=['neighborhood'],
        variables=['bedrooms', 'bathrooms']
    ))
])

pipeline.fit(X_train)
X_train_clean = pipeline.transform(X_train)
X_test_clean = pipeline.transform(X_test)
```

---

## Method Selection Guide {#selection-guide}

### Decision Flowchart

```
START: You have missing data

1. What % is missing?
   ├─ <2%? → Consider Complete Case Analysis
   ├─ 2-10%? → Statistical methods (Mean/Median, Group-based)
   └─ >10%? → Random Sample or Arbitrary String

2. Missing data type?
   ├─ MCAR? → Any method works
   ├─ MAR? → Mean/Median + (Missing Indicator recommended)
   └─ NMAR? → End of Tail OR Arbitrary Value (MUST have Missing Indicator)

3. Variable distribution?
   ├─ Normal? → Mean Imputation
   ├─ Skewed? → Median or End of Tail
   └─ Categorical? → Mode or Random Sample

4. Deploying to production?
   ├─ YES? → Must have imputation method (not CCA)
   └─ NO? → Any method acceptable

→ SELECT METHOD
```

### Comparison Matrix

| **Situation** | **Best Method** | **Second Best** | **Avoid** |
|---|---|---|---|
| <2% missing, MCAR | Complete Case Analysis | Mean (normal) | - |
| Normally distributed | Mean | Median | End of Tail |
| Skewed distribution | Median | End of Tail | Mean |
| Categorical | Random Sample | Mode (if few missing) | - |
| Stratified population | Group-Based Mean | Random Sample | - |
| NMAR data | End of Tail | Arbitrary Value | Mean/Median alone |
| Production model | Random Sample | Group-Based | Complete Case |

---

## Key Takeaways

🎯 **Method Selection Principles:**

1. **Complete Case Analysis**: Only when <2% missing AND truly MCAR AND not going to production
2. **End of Tail**: When missing is predictive (NMAR) OR distribution is very skewed
3. **Random Sample**: When you want perfect distribution preservation AND can afford memory costs
4. **Group-Based Mean/Median**: When groups have distinctly different distributions AND you have domain knowledge

✅ **Golden Rules:**
- Calculate parameters from **TRAINING SET ONLY**
- Apply same parameters to **BOTH train AND test**
- Check **sample sizes** per group before grouping
- Use **Missing Indicators** for MAR and NMAR data
- Consider **production implications** when choosing

---
