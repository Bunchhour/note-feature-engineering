# Multivariate Imputation Methods: Using Data Relationships
**Advanced Techniques That Leverage Variable Relationships to Fill Missing Data**

---

## Table of Contents
1. [Univariate vs. Multivariate](#univariate-vs-multivariate)
2. [When to Use Multivariate Methods](#when-to-use)
3. [KNN Imputation](#knn-imputation)
4. [MICE (Multivariate Imputation by Chained Equations)](#mice)
5. [MissForest (MICE with Random Forests)](#missforest)
6. [Implementation Guide](#implementation)
7. [Performance Comparison](#comparison)
8. [Choosing Your Method](#choosing-method)

---

## Univariate vs. Multivariate {#univariate-vs-multivariate}

### The Core Difference

| Aspect | **Univariate** | **Multivariate** |
|---|---|---|
| **Data Source** | Uses only the column itself (Mean, Median, Mode) | Uses ALL other features in the dataset |
| **Logic** | "Fill with average of this column" | "Predict from other columns" |
| **Complexity** | Very simple & fast | Complex & slower (trains models) |
| **Relationships** | Can destroy correlations | Preserves relationships & joint distribution |
| **Use Case** | Quick baseline, production efficiency | Research, complex data, high accuracy needs |
| **Example** | Missing age → use average age | Missing age → predict from income, education, etc. |

### Visual Comparison

```
UNIVARIATE (Simple):
Variable A: [10, 15, ?, 25, 30]
            Use average of A → [10, 15, 20*, 25, 30]
            *Based only on A's values

MULTIVARIATE (Smart):
Variable A: [10, 15, ?, 25, 30]
Variable B: [0.5, 0.6, 0.8, 0.9, 1.0]
            Use B to predict A → [10, 15, 22*, 25, 30]
            *Based on relationship: A ≈ 30×B
```

---

## When to Use Multivariate Methods {#when-to-use}

### Use Multivariate When:

✅ **Strong correlations exist** between variables
- Income correlated with education level
- Price correlated with size and location
- Age correlated with job experience

✅ **Data is Missing at Random (MAR)**
- Missingness can be explained by other variables
- Example: people with low income don't report salary

✅ **High-stakes applications**
- Medical diagnosis (accuracy critical)
- Loan approval (fairness matters)
- Research studies

✅ **Complex, non-linear relationships**
- Simple mean/median fails to capture patterns
- Data has interactions between variables

### Avoid Multivariate When:

❌ **Time constraints** - need results quickly
❌ **Simple, linear relationships** - univariate works fine
❌ **Very large datasets** - too slow
❌ **Many variables are missing** - error compounds
❌ **Data is MCAR** - no relationships to exploit
❌ **Production performance critical** - univariate is faster

---

## KNN Imputation {#knn-imputation}

### What is it?

**KNN Imputation** fills missing values by finding the **k most similar rows** (nearest neighbors) and averaging their values.

**Logic:**
```
Row A has missing age
Find rows B, C, D that are most similar to row A
Average their ages → Row A's imputed age
```

### How "Similarity" is Measured

Uses **Euclidean Distance** based on all other features:

$$\text{Distance} = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2 + ... + (x_n - x_n')^2}$$

⚠️ **Critical:** You **must scale features** before KNN!

Without scaling:
```
Feature 1: Age (0-100)
Feature 2: Income (0-1,000,000)
→ Income dominates distance calculation
→ Age variations are ignored!
```

### Weighting Methods

#### Uniform Weights
- All k neighbors get equal vote
- Simple average: $ \text{value} = \frac{v_1 + v_2 + v_3}{3} $
- Treats near and far neighbors the same

#### Distance Weights
- Closer neighbors have more influence
- Weight: $ w = \frac{1}{d} $ (reciprocal of distance)
- Far neighbors are "muted"

**Example:**
```
k=3 neighbors for missing age

Uniform:
Age values: [30, 35, 45]
Imputed: (30 + 35 + 45) / 3 = 36.7

Distance-weighted:
Distances: [0.5, 1.0, 2.0]
Weights: [1/0.5=2.0, 1/1.0=1.0, 1/2.0=0.5]
Imputed: (30×2.0 + 35×1.0 + 45×0.5) / (2.0+1.0+0.5) = 33.3
→ Closer neighbor (30) has more influence
```

### Choosing k

| k Value | Effect |
|---|---|
| **k=3-5** | Very local, may miss broader patterns |
| **k=5-10** | **Good default, balanced** |
| **k=10-20** | More stable, smooths noise |
| **k>50** | Too global, loses local patterns |

**How to find optimal k:**
1. Hide some known values (pretend they're missing)
2. Impute them with different k values
3. Calculate RMSE for each k
4. Pick k with lowest error

### Distribution Impact

```
BEFORE:                    AFTER (KNN Imputation):
  |    ╱╲                    |    ╱╲
  |   ╱  ╲                   |   ╱  ╲ (shape preserved!)
  |  ╱    ╲                  |  ╱    ╲
  |_╱______╲___        →     |_╱______╲___
  Original  Normal shape    Same distribution
```

✅ **Preserves distribution shape**  
✅ **Preserves correlations** (uses other variables)  
✅ **Preserves outliers**  

### Pros and Cons

**Pros:**
✅ Very intuitive - uses actual neighboring values  
✅ Captures local patterns well  
✅ No assumptions about distribution  
✅ Preserves relationships between variables  

**Cons:**
❌ **Extremely slow** on large datasets (calculate distances to all rows)  
❌ Sensitive to outliers (weird neighbor can distort value)  
❌ Must scale features first (Euclidean distance-based)  
❌ Struggles with high-dimensional data  

### Implementation

**Scikit-learn:**
```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Must scale first!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5, weights='distance'))
])

pipeline.fit(X_train)
X_train_clean = pipeline.transform(X_train)
X_test_clean = pipeline.transform(X_test)
```

**Key Parameters:**
```python
KNNImputer(
    n_neighbors=5,           # How many neighbors to use
    weights='distance',      # 'uniform' or 'distance'
    metric='nan_euclidean'   # Distance metric
)
```

---

## MICE (Multivariate Imputation by Chained Equations) {#mice}

### What is it?

**MICE** treats each variable with missing data as a **target to be predicted** by all other variables. It loops through this process multiple times (chains).

**Core Idea:** Predict missing Age using Income, Education, Job, etc. Then predict missing Income using Age, Education, Job, etc.

### How It Works: Step-by-Step

**Step 1: Placeholder Fill**
```
All missing values → Fill with means (temporary!)
Now every variable has complete data
```

**Step 2: Revert and Predict (Round 1)**
```
Column A:
  - Set missing values back to NaN
  - Train model: A ~ B + C + D + E
  - Predict the NaN values in A
  - Now A has better values (predicted)

Column B:
  - Set missing values back to NaN
  - Train model: B ~ A + C + D + E (using new A values!)
  - Predict NaN values in B
  
Column C, D, E... (same process)
```

**Step 3: Repeat**
```
Cycle 2: Do the same thing for all columns again
         (but use the newly predicted values)
Cycle 3, 4, 5... (usually 10 cycles total)
```

**Why cycles matter:**
- Round 1: Predictions are "polluted" by mean imputation
- Round 2-10: Values refine as they're based on better estimates
- Round 10: Joint distribution stabilizes

### Assumptions

- **Missing At Random (MAR):** Missingness can be explained by other variables
- **No structural zeros:** All values are theoretically possible
- **Enough data:** Informative variables exist to predict the missing ones

### Choosing a Base Learner

| Data Type | Best Model | Pros |
|---|---|---|
| **Linear relationships** | Bayesian Ridge (default) | Fast, stable |
| **Non-linear relationships** | Random Forest | Handles interactions, robust |
| **Mixed relationships** | Decision Trees | Simple, versatile |
| **Binary outcomes** | Logistic Regression | Appropriate for categories |

### Implementation

**Scikit-learn:**
```python
from sklearn.experimental import enable_iterative_imputer  # Enable first!
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Basic MICE (uses Bayesian Ridge)
imputer = IterativeImputer(
    max_iter=10,
    random_state=42
)

# MissForest (MICE with Random Forest)
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10),
    max_iter=10,
    random_state=42
)

imputer.fit(X_train)
X_train_clean = imputer.transform(X_train)
X_test_clean = imputer.transform(X_test)
```

**Key Parameters:**
```python
IterativeImputer(
    estimator=None,                    # Model to use (default: BayesianRidge)
    max_iter=10,                       # Number of cycles
    initial_strategy='mean',           # First-round placeholder
    imputation_order='ascending',      # Order to impute columns
    skip_complete=True,                # Skip columns with no missing data
    random_state=42                    # Reproducibility
)
```

---

## MissForest (MICE with Random Forests) {#missforest}

### What Makes It Special

**MissForest** is simply MICE using **Random Forest** as the base learner. It combines the best of both:

- ✅ MICE's chained iteration (preserves relationships)
- ✅ Random Forest's power (handles non-linear relationships)

### Why Random Forests for Imputation?

| Advantage | Impact |
|---|---|
| **Non-linear relationships** | Captures curved, complex patterns |
| **Feature interactions** | Detects "A only matters when B is high" |
| **Mixed data types** | Handles continuous AND categorical naturally |
| **Robust to outliers** | One weird value doesn't distort imputation |
| **Automatic feature importance** | Learns which variables matter most |

### Example: Comparing Estimators

```
Predicting missing Age from Income, Education, Job:

LINEAR MODEL (Bayesian Ridge):
Age ≈ -15 + 0.00005×Income + 5×Education_level + 2×Job_code
→ Assumes straight-line relationships

RANDOM FOREST:
IF Income > 80000 AND Education = High:
   Age ← older values (wealthy, educated people are older on average)
ELIF Income < 30000:
   Age ← younger values (low earners are younger)
IF Education = Low:
   Ignore Income (education is more predictive)
→ Captures non-linear patterns and interactions
```

### The Speed Trade-off

⚠️ **Warning: MissForest is SLOW**

Why?
- Must train Random Forest for **every variable**
- Does this **multiple times** (10 iterations usually)
- Large dataset? Can take hours!

**Speed comparison (approximate):**
```
Simple Mean Imputation:     0.1 seconds
KNN Imputation (k=5):       5 seconds
MICE (Bayesian Ridge):      10 seconds
MissForest:                 300 seconds (5 minutes!)
```

---

## Implementation Guide {#implementation}

### Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import Ridge

# Choose your method and build pipeline
imputation_pipeline = Pipeline([
    # Option 1: KNN (requires scaling)
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5)),
    
    # Option 2: MICE Basic
    # ('imputer', IterativeImputer(max_iter=10)),
    
    # Option 3: MissForest
    # ('imputer', IterativeImputer(
    #     estimator=RandomForestRegressor(n_estimators=10),
    #     max_iter=10
    # )),
])

# Fit and transform
imputation_pipeline.fit(X_train)
X_train_clean = imputation_pipeline.transform(X_train)
X_test_clean = imputation_pipeline.transform(X_test)
```

### Adding to Full ML Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Full pipeline: Imputation → Scaling → Model
full_pipeline = Pipeline([
    ('imputation', IterativeImputer()),
    ('scaling', StandardScaler()),
    ('model', Ridge())
])

# Grid search to optimize imputation method
param_grid = {
    'imputation__estimator': [None, RandomForestRegressor()],
    'imputation__max_iter': [10, 20],
    'model__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## Performance Comparison {#comparison}

### Example: Predicting House Prices with Different Imputation Methods

```
Model Performance (R² Score):

Imputation Method           Train R²    Test R²    Time
─────────────────────────────────────────────────────────
No Imputation (CCA)         0.78        0.72       0.01s
Mean Imputation             0.79        0.73       0.01s
Median Imputation           0.79        0.73       0.02s
KNN Imputation (k=5)        0.81        0.75       3.2s
MICE (Bayesian Ridge)       0.80        0.74       8.1s
MissForest                  0.81        0.75       245s

Conclusion:
KNN and MissForest have same accuracy (R²=0.75)
but KNN is 76× faster! Choose simplicity.
```

### Complexity vs. Accuracy Trade-off

```
Accuracy
   |           MissForest
   |            (r²=0.75)
   |         /
   |        /  KNN
   |       /   (r²=0.75)
   |      /
   |     /  MICE
   |    /  (r²=0.74)
   |   / Mean
   |  / (r²=0.73)
   |_/__________________ Computational Time
   0.01s     8s      240s
```

**The Lesson:** Above a certain accuracy threshold, more complex methods don't buy you much improvement, and you waste time/resources.

---

## Choosing Your Method {#choosing-method}

### Decision Flowchart

```
START: Choose an imputation method

1. Dataset Size?
   ├─ HUGE (>1M rows) → KNN too slow
   │   ├─ Simple data → Mean/Median
   │   └─ Complex data → MICE (Bayesian Ridge)
   │
   └─ Medium/Small → Any method acceptable

2. Data Relationships?
   ├─ Linear → MICE (Bayesian Ridge) or KNN
   ├─ Non-linear → Random Forest or KNN
   └─ Unknown → KNN (safest bet)

3. Accuracy Critical?
   ├─ YES (research, medical) → MissForest
   └─ NO (business, production) → Simple method

4. Time Budget?
   ├─ Tight → Mean/Median or KNN
   ├─ Flexible → MICE
   └─ Unlimited → MissForest

→ CHOOSE METHOD
```

### Quick Selection Guide

| **Situation** | **Best Method** | **Why** |
|---|---|---|
| Huge dataset (>1M rows) | KNN or Mean | Fast enough |
| Complex non-linearity | MissForest | Captures patterns |
| Linear relationships | MICE (default) | Still captures relationships, faster |
| Time is critical | Mean/Median | Instant |
| Need best accuracy | MissForest | Most flexible model |
| Production deployment | KNN | Best balance |
| Research/publication | MissForest | Defensible choice |
| Quick baseline | Mean/Median | For comparison |

### Parameter Guidelines

**KNN:**
```python
n_neighbors = 5-10        # Default 5, try 10 if slow
weights = 'distance'      # Better than 'uniform' usually
```

**MICE:**
```python
max_iter = 10             # 10 cycles is standard
imputation_order = 'ascending'  # Columns with less missing first
skip_complete = True      # Save time on complete columns
```

**MissForest:**
```python
estimator = RandomForestRegressor(n_estimators=10)  # Use 10-50
max_iter = 10             # Standard
```

---

## Key Takeaways

🎯 **Critical Insights:**

1. **Multivariate beats Univariate** when relationships exist, but the difference diminishes with more complex data
2. **KNN is the practical choice** - good accuracy, reasonable speed, easy to implement
3. **MICE is versatile** - can use different estimators, handles most scenarios
4. **MissForest is powerful but slow** - use when accuracy is worth the wait
5. **Simplicity often wins** - if Mean imputation gives 73% accuracy and MissForest gives 75%, mean wins for production

✅ **Golden Rules:**

- **Always scale** before KNN (distance-based)
- **Check correlations** before choosing - if weak correlations, univariate is enough
- **Time your methods** - profile before committing to slow approaches
- **Compare methods** using GridSearch - let data decide
- **Use Missing Indicators** - even with multivariate methods

---
