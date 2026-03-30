# Missing Data Imputation: Complete Guide
**Learn to Fill Missing Values Without Losing Information**

---

## Table of Contents
1. [Quick Decision Guide](#quick-decision)
2. [Why Imputation Matters](#why-matters)
3. [The Three Mechanisms of Missing Data](#mechanisms)
4. [Imputation Strategies](#strategies)
5. [Tool Comparison](#tool-comparison)
6. [Step-by-Step Implementation](#implementation)
7. [Common Pitfalls & Solutions](#pitfalls)
8. [Complete Workflow Examples](#workflows)

---

## Quick Decision Guide {#quick-decision}

**Choose your imputation method based on your data and assumptions:**

```
START: You have missing data

1. What TYPE of variable?
   ├─ NUMERICAL → Go to 2
   └─ CATEGORICAL → Go to 3

2. NUMERICAL Variable Missing Data
   ├─ Is the data NORMALLY DISTRIBUTED?
   │   ├─ YES → Use MEAN imputation
   │   └─ NO (Skewed) → Use MEDIAN imputation
   ├─ Missing is PREDICTIVE (NMAR)?
   │   └─ YES → Use ARBITRARY VALUE (-999, 999, etc.)
   └─ Use Missing Indicator? → ADD MISSING FLAG (0/1 column)

3. CATEGORICAL Variable Missing Data
   ├─ Is LOTS of data missing (>10%)?
   │   ├─ YES → Use ARBITRARY STRING ("Missing")
   │   └─ NO → Use MODE (most frequent category)
   └─ Use Missing Indicator? → ADD MISSING FLAG (0/1 column)

IMPORTANT: Calculate mean/median/mode from TRAINING SET ONLY
          Apply to BOTH training and test sets
```

---

## Why Imputation Matters {#why-matters}

### The Core Problem

Most machine learning libraries (scikit-learn, XGBoost, etc.) **cannot handle missing values**:
- Algorithms expect a complete grid of numbers
- `NaN` and `Null` values cause the model to crash
- Can't just delete rows—you lose valuable data

### The Solution: Imputation

**Imputation** = Intelligently filling missing values without distorting the data

**Options:**
1. **Delete rows** (loses information) ❌
2. **Fill intelligently with imputation** (preserves information) ✅

---

## The Three Mechanisms of Missing Data {#mechanisms}

Understanding **WHY** data is missing is crucial for choosing the right imputation method.

### 1. Missing Completely At Random (MCAR)

**Definition:** Missing values are purely random and unrelated to any other variable or to the value itself.

**Real Example:**
- A survey respondent's internet disconnects randomly → missing Income value
- Not related to their actual income or any other factor

**Best Imputation:** 
- Simple deletion OR Mean/Median/Mode
- ✓ Missing Indicator optional
- ✅ **Risk: Low bias**

---

### 2. Missing At Random (MAR)

**Definition:** Missing values are systematically related to **other observed variables** (but not to the unobserved value itself).

**Real Example:**
- Women less likely to disclose weight than men
- Missing weight is related to Gender variable
- You can observe and account for this relationship

**Best Imputation:**
- Mean/Median/Mode
- ✓ **Always add Missing Indicator** (important!)
- Include the related variable in your model
- ⚠️ **Risk: Medium bias** (controlled by including related variable)

---

### 3. Missing Not At Random (NMAR)

**Definition:** Missing values are related to **the unobserved value itself** (the outcome).

**Real Example:**
- People with very high salaries less likely to report it
- Missing Salary = likely high earner
- People with depression skip depression survey

**Best Imputation:**
- **MUST use Arbitrary Value** (-999, "Missing")
- **MUST add Missing Indicator** (critical!)
- Let the model learn the pattern
- 🔴 **Risk: High bias** if ignored

---

## Imputation Strategies {#strategies}

### NUMERICAL DATA IMPUTATION

#### Strategy 1: Mean Imputation

**Use When:** Variable is **normally distributed** (symmetric bell curve)

**How:** Replace `NaN` with the average of all non-missing values

**Pros:**
✅ Fast and simple  
✅ Preserves the mean of the distribution  
✅ Good when distribution is normal  

**Cons:**
❌ Creates spike in the middle (shrinks variance)  
❌ Destroys relationships between variables  
❌ Doesn't work well with skewed data  

**Python Code:**
```python
# Pandas
mean_dict = X_train.median().to_dict()
X_train.fillna(mean_dict, inplace=True)
X_test.fillna(mean_dict, inplace=True)

# Scikit-learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Feature-engine
from feature_engine.imputation import MeanMedianImputer
imputer = MeanMedianImputer(imputation_method='mean')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
```

---

#### Strategy 2: Median Imputation

**Use When:** Variable is **skewed** OR has **outliers**

**How:** Replace `NaN` with the middle value (50th percentile)

**Why Better for Skewed Data:**
- Mean gets pulled toward outliers
- Median stays at the true center

**Example:**
```
House Prices: $300K, $350K, $400K, $450K, $500K, $50M

MEAN = $8.8M (distorted by mansion!)
MEDIAN = $425K (true center) ← Use this!
```

**Pros:**
✅ Robust to outliers  
✅ Better for skewed distributions  
✅ More realistic for non-normal data  

**Cons:**
❌ Still creates spike  
❌ Shrinks variance  

**Python Code:**
```python
# Pandas
median_dict = X_train.median().to_dict()
X_train.fillna(median_dict, inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)

# Feature-engine
imputer = MeanMedianImputer(imputation_method='median')
imputer.fit(X_train)
```

---

#### Strategy 3: Arbitrary Value Imputation

**Use When:** Missing data is **Not Missing At Random (NMAR)**

**How:** Replace `NaN` with a value that **stands out** from normal data

**Common Arbitrary Values:**
- `-1` (if data is always positive)
- `-999` or `999` (clearly outside normal range)
- `0` (if zero is uncommon)

**Key Rule:** Pick a value that is **impossible** to be real!

**Why This Works:**
- Creates a "flag" that marks missing data
- Tree-based models recognize this flag
- Model learns: "This flag has predictive power"

**Pros:**
✅ Flags the missing data visually  
✅ Model can learn from the "missingness"  
✅ Good for tree-based models  

**Cons:**
❌ Creates massive outlier spike  
❌ Expands variance dramatically  
❌ Not suitable for linear models  

**Python Code:**
```python
# Pandas
X_train['age'].fillna(-999, inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='constant', fill_value=-999)

# Feature-engine
from feature_engine.imputation import ArbitraryNumberImputer
imputer = ArbitraryNumberImputer(arbitrary_number=-999)
```

---

### CATEGORICAL DATA IMPUTATION

#### Strategy 4: Mode (Frequent Category) Imputation

**Use When:** Only a **small percentage** is missing (<5-10%)

**How:** Replace `NaN` with the **most common category**

**Pros:**
✅ Simple to understand  
✅ Often correct guess  
✅ Preserves category distribution  

**Cons:**
❌ Can distort distribution  
❌ Loses the signal that data was missing  
❌ Wrong if missing is systematic  

**Python Code:**
```python
# Pandas
mode_category = X_train['color'].mode()[0]
X_train['color'].fillna(mode_category, inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='most_frequent')

# Feature-engine
from feature_engine.imputation import CategoricalImputer
imputer = CategoricalImputer(imputation_method='frequent')
```

---

#### Strategy 5: Arbitrary String Imputation

**Use When:** Lots of data is missing (>10%) OR missing is predictive (NMAR)

**How:** Replace all `NaN` with a new category like `"Missing"` or `"Unknown"`

**Why This Works:**
- Creates a new distinct category
- Model can learn: "Missing category has different properties"
- Preserves the signal that data was absent

**Pros:**
✅ Preserves the "missingness signal"  
✅ Most transparent approach  
✅ Safe for all algorithm types  
✅ Works with lots of missing data  

**Cons:**
❌ Creates a new category (increases cardinality)  
❌ May split data too much  

**Python Code:**
```python
# Pandas
X_train['color'].fillna('Missing', inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='constant', fill_value='Missing')

# Feature-engine
imputer = CategoricalImputer(imputation_method='missing', fill_value='Missing')
```

---

#### Strategy 6: Missing Indicator (The "Best of Both Worlds")

**What is it:** A **binary column** (0/1) that flags whether a value was originally missing

**How to Use:**
1. **Create** the indicator BEFORE imputing
2. **Impute** the original column
3. **Result:** Model uses imputed value AND knows it was originally missing

**Why It Works:**
- Lets the model decide: Is missingness predictive?
- If yes → Model learns from the indicator
- If no → Model ignores the indicator

**Pros:**
✅ "Best of both worlds"  
✅ Lets model decide significance  
✅ Captures predictive information  

**Cons:**
❌ Adds extra columns  
❌ Can cause multicollinearity  
❌ Increases complexity  

**Python Code:**
```python
# Pandas - Manual
X_train['price_na'] = X_train['price'].isna().astype(int)

# Scikit-learn
imputer = SimpleImputer(strategy='mean', add_indicator=True)

# Feature-engine
from feature_engine.imputation import AddMissingIndicator
indicator = AddMissingIndicator(missing_only=True)
X_train = indicator.fit_transform(X_train)
```

---

## Tool Comparison {#tool-comparison}

| Feature | **Pandas** | **Scikit-learn** | **Feature-engine** |
|---|---|---|---|
| **Best For** | Exploration & one-offs | Production pipelines | Clean, readable code |
| **DataFrame Output** | ✅ Yes | ⚠️ No (returns array) | ✅ Yes |
| **Column Names Preserved** | ✅ Yes | ❌ No | ✅ Yes |
| **Parameter Storage** | ❌ Manual | ✅ Automatic | ✅ Automatic |
| **ColumnTransformer Needed** | ❌ No | ✅ Yes (for mixed types) | ❌ No |
| **Code Simplicity** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Deployment Ready** | ❌ Manual work | ✅ Yes | ✅ Yes |

---

## Step-by-Step Implementation {#implementation}

### ⚠️ The Golden Rule: Train-Test Split First!

Your imputation parameters (mean, median, mode) are "learned" values!

```
❌ WRONG (Data Leakage):
1. Calculate mean from ENTIRE dataset
2. Fill missing values
3. Split into train/test
→ Test set "sees" training data!

✅ CORRECT:
1. Split data FIRST
2. Calculate mean from TRAINING set ONLY
3. Use that mean for BOTH train and test
→ No leakage!
```

---

### Implementation with Pandas

**Best for:** Quick exploration and prototyping

```python
from sklearn.model_selection import train_test_split

# 1. SPLIT FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. LEARN from training set
imputation_dict = X_train[['age', 'price']].median().to_dict()

# 3. FILL both train and test
X_train.fillna(imputation_dict, inplace=True)
X_test.fillna(imputation_dict, inplace=True)
```

---

### Implementation with Scikit-learn

**Best for:** Production pipelines

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_cols = X_train.select_dtypes(include='number').columns.tolist()

# Setup preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='median'), num_cols),
], remainder='passthrough')

preprocessor.set_output(transform='pandas')

X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)
```

---

### Implementation with Feature-engine

**Best for:** Clean, readable production code

```python
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer
)

imputation_pipeline = Pipeline([
    ('add_indicator', AddMissingIndicator(missing_only=True)),
    ('num_impute', MeanMedianImputer(imputation_method='median')),
    ('cat_impute', CategoricalImputer(imputation_method='frequent'))
])

imputation_pipeline.fit(X_train)
X_train_clean = imputation_pipeline.transform(X_train)
X_test_clean = imputation_pipeline.transform(X_test)
```

---

## Common Pitfalls & Solutions {#pitfalls}

### Pitfall 1: Data Leakage

**Problem:** Calculate mean from entire dataset before splitting

**Symptom:** Training R² = 95%, Test R² = 45% (huge drop!)

**Solution:** Always split FIRST, learn from training set ONLY

```python
# ❌ WRONG
mean = full_data['age'].mean()
X_train, X_test = split(data)
X_train.fillna(mean)

# ✅ RIGHT
X_train, X_test = split(data)
mean = X_train['age'].mean()
X_train.fillna(mean)
X_test.fillna(mean)
```

---

### Pitfall 2: Accidental Type Casting

**Problem:** Filling numerical columns with strings

**Symptom:** Model crashes with "expected number, got string"

**Solution:** Use `ColumnTransformer` to separate types

```python
# ❌ WRONG - fills numbers with strings!
imputer = SimpleImputer(strategy='constant', fill_value='Missing')
imputer.fit(mixed_data)

# ✅ RIGHT - separate by type
ct = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), num_cols),
    ('cat', SimpleImputer(strategy='constant', fill_value='Missing'), cat_cols)
])
```

---

### Pitfall 3: ColumnTransformer Deletes Columns

**Problem:** Setting `remainder='drop'` permanently removes columns

**Solution:** Always use `remainder='passthrough'`

```python
# ❌ WRONG - deletes unlisted columns!
ct = ColumnTransformer([
    ('impute', SimpleImputer(), some_cols)
], remainder='drop')

# ✅ RIGHT - keeps everything
ct = ColumnTransformer([
    ('impute', SimpleImputer(), some_cols)
], remainder='passthrough')
```

---

### Pitfall 4: Missing Indicators on No Data

**Problem:** Adding indicators AFTER imputation (no holes left to find!)

**Solution:** Add indicators BEFORE imputing

```python
# ❌ WRONG - no missing data to find after fillna
X['age'].fillna(mean, inplace=True)
X['age_na'] = X['age'].isna().astype(int)  # All 0s!

# ✅ RIGHT - indicators first
pipeline = Pipeline([
    ('indicators', AddMissingIndicator()),  # First!
    ('impute', MeanMedianImputer())  # Then!
])
```

---

## Complete Workflow Examples {#workflows}

### Example 1: Simple Numerical Data

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, np.nan, 35, 40, np.nan],
    'salary': [50000, 60000, np.nan, 80000, 90000]
})

# Impute with median
median_dict = df[['age', 'salary']].median().to_dict()
df.fillna(median_dict, inplace=True)
print(df)
```

---

### Example 2: Mixed Data Types

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

X = pd.DataFrame({
    'age': [25, np.nan, 35, 40],
    'color': ['Red', np.nan, 'Blue', 'Red'],
    'price': [100, 150, np.nan, 200]
})

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), ['age', 'price']),
    ('cat', SimpleImputer(strategy='constant', fill_value='Unknown'), ['color']),
], remainder='passthrough')

preprocessor.set_output(transform='pandas')
X_clean = preprocessor.fit_transform(X)
```

---

### Example 3: Production-Ready Pipeline

```python
from sklearn.pipeline import Pipeline
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer
)

full_pipeline = Pipeline([
    ('missing_indicator', AddMissingIndicator(missing_only=True)),
    ('num_imputer', MeanMedianImputer(imputation_method='median')),
    ('cat_imputer', CategoricalImputer(imputation_method='missing')),
])

full_pipeline.fit(X_train)
X_train_clean = full_pipeline.transform(X_train)
X_test_clean = full_pipeline.transform(X_test)
```

---

## Quick Decision Matrix

| **Situation** | **Variable Type** | **% Missing** | **Best Method** |
|---|---|---|---|
| Quick test | Numerical | <5% | Mean |
| Real data | Numerical | <10% | Median |
| NMAR case | Numerical | Any | Arbitrary (-999) |
| Few missing | Categorical | <5% | Mode |
| Lots missing | Categorical | >10% | String ("Missing") |
| Production | Any | Any | Full Pipeline |

---

## Key Takeaways

🎯 **Core Principles:**

1. **Always split data BEFORE learning imputation parameters**
2. **Choose method based on distribution shape & missingness type**
3. **Use Missing Indicators for MAR/NMAR data**
4. **Use appropriate tool for your workflow** (Pandas → Scikit-learn → Feature-engine)
5. **Be careful with mixed data types** (use ColumnTransformer)

✅ **Imputation Decision Format:**
- Normal distribution → Mean imputation
- Skewed distribution → Median imputation
- NMAR data → Arbitrary value imputation
- Categorical → Mode (few missing) or String (many missing)
- Always consider → Missing Indicator flag

---
