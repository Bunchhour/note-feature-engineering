# Categorical Encoding: Handling Rare Labels & High Cardinality

## 📚 Learning Guide for Data Preprocessing

> **This guide teaches you how to handle categorical variables with many unique values (high cardinality) and rare categories that appear in only a tiny fraction of your data.**

---

## 🎯 Quick Decision Guide: Which Approach to Use?

| **Your Situation** | **Best Approach** | **Why** |
| :--- | :--- | :--- |
| Many unique categories (20+) | **Top-N One-Hot Encoding** | Reduces dimensionality, prevents feature explosion |
| Rare categories present (< 5% of data) | **Rare Label Encoding** | Improves model stability and deployment safety |
| Need to prevent model crashes on unseen data | **Either method** | Both handle new categories gracefully |
| Working with pandas | **RareLabelEncoder** | Simpler explicit grouping approach |
| Building production pipeline | **Feature-engine methods** | Better for reproducibility and deployment |
| Want automatic unseen data handling | **Scikit-learn's OneHotEncoder** | Built-in `handle_unknown` parameter |
| Simple, interpretable solution | **RareLabelEncoder** | Groups rare into single "Rare" category |

---

## 🚨 What's the Problem?

### High Cardinality Issue
- **Definition:** Categorical variable with many unique values (neighborhood with 25 categories)
- **Problem:** Standard One-Hot Encoding creates 25 columns → feature explosion, curse of dimensionality
- **Impact:** Models become slow, complex, overfitting prone, harder to deploy

### Rare Labels Issue
- **Definition:** Categories appearing in < 5% of observations

- **Four Critical Problems:**
  1. **Overfitting:** Model learns specific noise from tiny sample sizes
  2. **Unseen categories:** New category in production → encoding crashes ❌
  3. **Unreliable statistics:** Target mean on 2 samples ≠ trustworthy estimate
  4. **Deployment failure:** Pipeline breaks when encountering new category

### Why It Matters: The Deployment Reality
In production, new categories **will** appear. Without handling them, your pipeline crashes.

**Example:** Your model trained on 25 neighborhoods. Production introduces "New Estate" neighborhood → encoding system doesn't know how to handle it → pipeline breaks.

---

## 💡 Two Core Solutions

### **Solution 1: Top-N One-Hot Encoding** (Implicit Grouping)
- Create dummy columns ONLY for most frequent categories
- Rare categories automatically get zeros across all columns
- All rare + unseen categories treated as baseline (the "all-zero" pattern)
- **Use when:** You want simplicity and automatic handling

### **Solution 2: Rare Label Encoding** (Explicit Grouping)
- Replace rare category strings with single label: "Rare" or "Other"
- Reduces cardinality explicitly BEFORE any encoding
- More interpretable: can see exactly which records got grouped
- **Use when:** You want explicit control and clarity

---

## 📚 Implementation Libraries

| Library | Best For | Complexity | Transparency |
| :--- | :--- | :--- | :--- |
| **pandas + numpy** | Learning & understanding | Manual | Fully visible |
| **Feature-engine** | Production pipelines | Simple API | Clear mappings |
| **Scikit-learn** | Standard ML workflows | Good | Moderate |

---

# 1️⃣ METHOD: TOP-N ONE-HOT ENCODING (Implicit Grouping)

## Overview
Creates binary dummy columns **ONLY** for the most frequent categories. Rare and unseen categories automatically get zeros across all columns (implicit baseline grouping).

## How It Works: Visual Example

**Original data - Neighborhood column:**
- London: 1,000 observations (40%)
- Manchester: 500 observations (20%)
- Leeds: 200 observations (8%)
- Yorkshire: 100 observations (4%)
- Milton-Keynes: 80 observations (3%)
- Cambridge: 20 observations (1%)
- Other 100+ observations (4%)

**After Top-3 One-Hot Encoding:**

| Original | is_London | is_Manchester | is_Leeds |
| :--- | :--- | :--- | :--- |
| London | 1 | 0 | 0 |
| Manchester | 0 | 1 | 0 |
| Leeds | 0 | 0 | 1 |
| Yorkshire | **0** | **0** | **0** |
| Milton-Keynes | **0** | **0** | **0** |
| Cambridge | **0** | **0** | **0** |
| Liverpool (unseen) | **0** | **0** | **0** |

**Key insight:** All rare + unseen categories share identical representation (0,0,0) → implicit grouping

## Step-by-Step Process

1. **Count category frequencies** → Sort by frequency (highest first)
2. **Select top N** → Identify most frequent categories
3. **Create dummy columns** → Binary columns for top N only
4. **Rare categories get zeros** → All rare/unseen map to "all-zero" pattern
5. **Drop original column** → Replace categorical with new dummies

## Pros & Cons

✅ **Advantages:**
- Dramatically controls feature explosion (3 columns vs 7+)
- Handles unseen categories gracefully (as all-zeros)
- Simple and straightforward implementation
- Perfect for linear models
- Automatic, no manual decisions needed

❌ **Limitations:**
- Still expands feature space (just less extreme)
- Loses specific information about individual rare categories
- All rare categories treated as identical (no distinction)
- Doesn't add predictive target information (structural only)
- Multiple categorical variables still add many columns

---

## Implementation: Top-N One-Hot Encoding

### With pandas (Learning)

```python
import pandas as pd
import numpy as np

# Step 1: Split first (crucial!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Find top categories (ONLY from training set)
top_categories = X_train['neighborhood'].value_counts().head(10).index.tolist()
print(f"Top 10 categories: {top_categories}")

# Step 3: Create dummy columns for top categories only
for cat in top_categories:
    X_train[f'neighborhood_{cat}'] = np.where(X_train['neighborhood'] == cat, 1, 0)
    X_test[f'neighborhood_{cat}'] = np.where(X_test['neighborhood'] == cat, 1, 0)

# Step 4: Drop original column
X_train = X_train.drop('neighborhood', axis=1)
X_test = X_test.drop('neighborhood', axis=1)

print(f"Original shape: {X.shape}, After encoding: {X_train.shape}")
```

**Reusable Function:**

```python
def encode_top_categories(X_train, X_test, variable, top_n=10):
    """
    Encode only top N categories, rare ones become all-zeros
    
    Parameters:
    - X_train, X_test: Train/test dataframes
    - variable: Column name to encode
    - top_n: Number of top categories to encode
    """
    # Learn from train only
    top_cats = X_train[variable].value_counts().head(top_n).index.tolist()
    
    # Create columns
    for cat in top_cats:
        X_train[f'{variable}_{cat}'] = (X_train[variable] == cat).astype(int)
        X_test[f'{variable}_{cat}'] = (X_test[variable] == cat).astype(int)
    
    # Drop original
    X_train = X_train.drop(variable, axis=1)
    X_test = X_test.drop(variable, axis=1)
    
    return X_train, X_test

# Usage
X_train, X_test = encode_top_categories(X_train, X_test, 'neighborhood', top_n=10)
X_train, X_test = encode_top_categories(X_train, X_test, 'exterior1st', top_n=10)
```

### With Feature-engine (Production) ⭐ Recommended

```python
from feature_engine.encoding import OneHotEncoder

# Step 1: Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create encoder (learns from training data)
encoder = OneHotEncoder(
    top_categories=10,                                    # Top 10 categories per variable
    variables=['neighborhood', 'exterior1st', 'exterior2nd'],  # Or None for auto-detect
    drop_last=False                                       # Keep all 10 columns (IMPORTANT!)
)

# Step 3: Fit (learns which are the top 10 in each column)
encoder.fit(X_train)

# Step 4: Inspect learned mappings (optional)
print(encoder.encoder_dict_)  # Shows which categories were selected

# Step 5: Transform both sets
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

print(f"New shape: {X_train_encoded.shape}")
```

### With Scikit-learn

```python
from sklearn.preprocessing import OneHotEncoder

# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create encoder
# max_categories=11 gives us: 10 top categories + 1 catch-all for rare/unseen
encoder = OneHotEncoder(
    max_categories=11,                    # Top 10 + 1 for infrequent
    handle_unknown='infrequent_if_exist', # Route unseen to "infrequent" column
    sparse_output=False                   # Dense output (easier to work with)
)

# Set to pandas output (get DataFrame instead of NumPy array)
encoder.set_output(transform="pandas")

# Fit and transform
encoder.fit(X_train[['neighborhood']])
X_train_encoded = encoder.transform(X_train[['neighborhood']])
X_test_encoded = encoder.transform(X_test[['neighborhood']])
```

⚠️ **Key Difference:** Scikit-learn's `max_categories=11` means: top 10 categories + 1 column for all infrequent/unseen

---

# 2️⃣ METHOD: RARE LABEL ENCODING (Explicit Grouping)

## Overview
Replace rare category strings with single label: "Rare" or "Other". Reduces cardinality explicitly and improves interpretability.

## How It Works: Step by Step

1. **Set frequency threshold** → Define what counts as "rare" (e.g., < 5%)
2. **Count frequencies** → Calculate what percentage each category represents
3. **Identify rare categories** → Find all categories below threshold
4. **Replace with "Rare"** → Overwrite rare category strings with new label
5. **Encode as normal** → Apply standard encoding to grouped data

## Example: Street Type Variable

**Before grouping:**
- Paved: 2,000 observations (80%)
- Gravel: 400 observations (16%)
- Cobblestone: 50 observations (2%) ← Rare
- Brick: 40 observations (1.6%) ← Rare
- Stone: 10 observations (0.4%) ← Rare

**After grouping (tol=0.05):**
- Paved: 2,000 observations (80%)
- Gravel: 400 observations (16%)
- Rare: 100 observations (4%) ← All rare grouped together

## The Three Cardinality Scenarios

### Scenario A: Binary / Very Low Cardinality (2 categories)
- **Example:** Street type = "Paved" (99%) or "Gravel" (1%)
- **Impact:** Renaming doesn't reduce cardinality much
- **Still valuable:** Prepares pipeline for unseen "NewType" → "Rare"

### Scenario B: Low Cardinality (4-5 categories)
- **Example:** 2 frequent categories + 2 rare ones
- **Impact:** Reduces from 4 to 3 categories
- **Caution:** Check if rare categories had strong predictive power

### Scenario C: High Cardinality (15+ categories, many rare)
- **Example:** Neighborhood with 25 categories, 15 are rare
- **Impact:** Dramatically reduces noise, drastically cuts cardinality
- **Best case:** This is where the technique shines

## The "Deployment Superpower"

**The Real Problem:** When your model goes to production, it WILL encounter new categories.

**The Crash Scenario:**
```
Model trained on: London, Manchester, Leeds
Test data encounters: Liverpool (never seen before)
Result: Pipeline crashes ❌
```

**The Solution - Rare Label Encoding:**
```
Training: Categories seen = [London, Manchester, Leeds]
Production: Liverpool appears
Pipeline logic: "Liverpool not in my frequent list → route to 'Rare'"
Result: Pipeline continues smoothly ✅
```

---

## Implementation: Rare Label Encoding

### With pandas (Learning)

```python
import pandas as pd
import numpy as np

# Step 1: Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Function to find frequent categories
def find_frequent_labels(df, variable, tolerance):
    """Find categories appearing in more than tolerance% of rows"""
    frequencies = df[variable].value_counts() / len(df)
    return frequencies[frequencies > tolerance].index.tolist()

# Step 3: Function to group rare categories
def group_rare_labels(X_train, X_test, variable, tolerance):
    """Replace rare categories with 'Rare' label"""
    # Learn from train only
    frequent_labels = find_frequent_labels(X_train, variable, tolerance)
    
    # Replace rare in both sets
    X_train[variable] = np.where(
        X_train[variable].isin(frequent_labels),
        X_train[variable],
        'Rare'
    )
    X_test[variable] = np.where(
        X_test[variable].isin(frequent_labels),
        X_test[variable],
        'Rare'
    )
    
    return X_train, X_test, frequent_labels

# Step 4: Apply to multiple columns
columns_to_group = ['neighborhood', 'street_type', 'exterior']

for col in columns_to_group:
    X_train, X_test, freq_cats = group_rare_labels(X_train, X_test, col, tolerance=0.05)
    print(f"{col}: Frequent categories = {freq_cats}")
```

### With Feature-engine (Production) ⭐ Recommended

```python
from feature_engine.encoding import RareLabelEncoder

# Step 1: Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create encoder with rules
rare_encoder = RareLabelEncoder(
    tol=0.05,                    # Group categories in < 5% of rows
    n_categories=4,              # Ignore variables with fewer than 4 categories
    variables=None               # Auto-detect all categorical columns
)

# Step 3: Fit (learns which categories to keep)
rare_encoder.fit(X_train)

# Step 4: Inspect what it learned (optional)
print(rare_encoder.encoder_dict_)

# Step 5: Transform both sets
X_train_grouped = rare_encoder.transform(X_train)
X_test_grouped = rare_encoder.transform(X_test)
```

### Feature-engine Parameters Explained

| Parameter | What It Does | Example |
| :--- | :--- | :--- |
| `tol=0.05` | Group categories appearing < 5% of time | Categories with < 5% rows become "Rare" |
| `n_categories=4` | Don't touch columns with ≤ 4 unique values | Binary columns stay untouched |
| `variables=None` | Auto-detect all categorical columns | Or pass `['col1', 'col2']` for specific ones |

**Override `n_categories`:**
```python
# To force grouping on ALL variables (even binary ones)
rare_encoder = RareLabelEncoder(tol=0.05, n_categories=1)
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Choosing the Right Tolerance

**Problem:** What's the right threshold? 5%? 1%? 10%?

**Solution:** It depends on your data and model

```python
# Analyze before deciding
for col in categorical_columns:
    freq_dist = X_train[col].value_counts(normalize=True)
    print(f"\n{col}:")
    print(freq_dist)
    print(f"Categories < 5%: {(freq_dist < 0.05).sum()}")
    print(f"Categories < 1%: {(freq_dist < 0.01).sum()}")
```

**Guidelines:**
- **High cardinality (20+ categories):** Use 5-10% threshold
- **Low cardinality (5-10 categories):** Use 1-2% threshold  
- **Very rare specific case:** Use 0.5% or lower

### Issue 2: Information Loss

**Problem:** "I'm grouping real categories together - am I losing predictive power?"

**Answer:** Possibly, but benefits outweigh costs for rare categories

**Tradeoff:**
- ❌ Lose: Specific patterns of rare categories
- ✅ Gain: Model stability, deployment safety, reduced overfitting, cleaner data

**When to reconsider:** If rare categories have strong target relationship → maybe keep them separate with frequency encoding instead

### Issue 3: "Rare" Category Has No Predictive Power

**Problem:** After grouping, "Rare" category doesn't seem to help the model

**Solution:** This is actually GOOD - you've isolated noise

You can then:
- Drop it entirely (merge with largest frequent category)
- Keep it as baseline for deployment safety
- Use in target encoding (so "Rare" gets smooth mean value)

---

## 📊 Comparison: When to Use Each Method

| Aspect | Top-N OHE | Rare Label Encoding |
| :--- | :--- | :--- |
| **Cardinality Reduction** | Implicit, automatic | Explicit, visible |
| **Feature Expansion** | Still adds columns | Reduces before encoding |
| **Interpretability** | Less clear which rare together | Clear - all in "Rare" |
| **Unseen Handling** | Automatic (all-zeros) | Automatic (→ "Rare") |
| **Deployment** | Good | Excellent |
| **Data Loss** | Some (lose rare category distinctness) | Some (but grouped clearly) |
| **Best for** | Linear models, simple pipelines | Complex hierarchies, safety-critical |
| **Flexibility** | Fixed to Top N | Threshold-based, adjustable |

---

## 🎯 Decision Tree: Which Method Should You Use?

```
START: I have high cardinality categorical data
  │
  ├─ Do you need to reduce feature space?
  │  ├─ YES → Continue
  │  └─ NO → Skip rare label handling
  │
  ├─ How many rare categories do you have?
  │  ├─ MANY (10+) → Use Rare Label Encoding
  │  ├─ Few (2-3) → Either method works
  │  └─ None → Not needed
  │
  ├─ Do you need explicit visibility on grouping?
  │  ├─ YES → Use Rare Label Encoding
  │  └─ NO → Use Top-N OHE
  │
  └─ DEFAULT RECOMMENDATION:
     ├─ For production/safety → Use Rare Label Encoding
     └─ For simplicity → Use Top-N OHE
```

---

## 📋 CHEAT SHEET: Quick Reference

### When to Use What

| **Use Case** | **Method** | **Why** |
| :--- | :--- | :--- |
| Many categories (20+) | Top-N OHE | Automatic, simple |
| Many rare categories | Rare Label Encoding | Explicit grouping |
| Production pipeline | Rare Label Encoding | Better safety |
| Learning purpose | Rare Label Encoding (pandas) | More transparent |
| High cardinality + linear | Rare Label then standard OHE | Works well together |
| Unseen category safety | Either method | Both handle it |

### Common Parameters

```python
# Feature-engine: Top-N OHE
OneHotEncoder(top_categories=10, variables=None, drop_last=False)

# Feature-engine: Rare Label Encoding
RareLabelEncoder(tol=0.05, n_categories=4, variables=None)

# Scikit-learn: Top-N OHE
OneHotEncoder(max_categories=11, handle_unknown='infrequent_if_exist')
```

### Checking Your Results

```python
# Before and after comparison
print("Before grouping:")
print(X_train['neighborhood'].nunique(), "unique categories")
print(X_train['neighborhood'].value_counts())

print("\nAfter grouping:")
print(X_train_grouped['neighborhood'].nunique(), "unique categories")
print(X_train_grouped['neighborhood'].value_counts())
```

---

## ⚠️ Key Warnings for Students

1. **ALWAYS split before grouping** - Learn frequency distribution from training set ONLY
2. **Document your threshold** - Record why you chose 5% vs 1% vs 10%
3. **Check the "Rare" category** - Ensure it's not overshadowing important patterns
4. **Validate the impact** - Compare model performance with/without grouping
5. **Consider your model type** - Rare labels affect linear and tree models differently
6. **Test on validation set** - Make sure grouping generalizes
7. **Save your encoder** - For production, save the list of frequent categories
8. **Communicate to stakeholders** - Explain why some categories got grouped

---

## 🔑 Summary: The Two Methods

| Aspect | Top-N OHE | Rare Label Encoding |
| :--- | :--- | :--- |
| **What it does** | Create dummies for top N categories | Replace rare strings with "Rare" |
| **Process** | Identify top N, create columns, rest → 0s | Identify rare, overwrite with label |
| **Feature space** | Expands (but limited) | Contracts (before any encoding) |
| **Interpretability** | Moderate | High (explicit "Rare" label) |
| **Rare handling** | Implicit (all-zero baseline) | Explicit (labeled) |
| **Unseen handling** | Automatic (→ all-zeros) | Automatic (→ "Rare") |
| **Library support** | Feature-engine, Scikit-learn | Feature-engine, pandas |
| **Deployment** | Good | Excellent |
| **Overfitting risk** | Reduced | Reduced |
| **Information preserved** | Most categories | Grouped rare info |

---

## ✅ Practical Workflow Example

```python
from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Group rare labels explicitly
rare_grouper = RareLabelEncoder(tol=0.05, n_categories=4, variables=None)
rare_grouper.fit(X_train)
X_train = rare_grouper.transform(X_train)
X_test = rare_grouper.transform(X_test)

# 3. One-Hot encode the grouped data
ohe = OneHotEncoder(top_categories=10, drop_last=False, variables=None)
ohe.fit(X_train)
X_train_encoded = ohe.transform(X_train)
X_test_encoded = ohe.transform(X_test)

# 4. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# 5. Evaluate
y_pred = model.predict_proba(X_test_encoded)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

# 6. Save encoders for production
import pickle
pickle.dump([rare_grouper, ohe], open('encoders.pkl', 'wb'))
```

---

## 🎓 Next Steps

1. **Inspect your data** - Use `.value_counts()` to understand cardinality
2. **Experiment** - Try different tolerance thresholds
3. **Compare models** - Build with/without rare label grouping
4. **Document decisions** - Record why you grouped categories a certain way
5. **Validate** - Test on holdout set that the grouping generalizes

* **London:** 1,000 observations 
* **Manchester:** 500 observations
* **Leeds:** 200 observations
* **Yorkshire:** 10 observations
* **Milton-Keynes:** 8 observations
* **Cambridge:** 5 observations

If you decide to limit your encoding to the **top 3 most frequent categories**, the algorithm will only create binary columns for London, Manchester, and Leeds. 

| Original City | `is_London` | `is_Manchester` | `is_Leeds` |
| :--- | :--- | :--- | :--- |
| London | 1 | 0 | 0 |
| Manchester | 0 | 1 | 0 |
| Leeds | 0 | 0 | 1 |
| Yorkshire | **0** | **0** | **0** |
| Milton-Keynes| **0** | **0** | **0** |

**The "All-Zero" Trick:** Notice what happens to the rare labels (Yorkshire, Milton-Keynes). Because they aren't in the top 3, they get a `0` in every new column. This effectively groups all rare and unseen categories together into a single, collective "other" representation without needing to explicitly create an "Other" column.



### **3. Advantages of this Method**
* **Straightforward implementation:** It is easy to understand and apply.
* **Saves time:** It does not require hours of deep variable exploration to figure out manual groupings.
* **Controls feature space:** It prevents the massive expansion of columns (the "curse of dimensionality") that standard OHE causes on highly cardinal variables.
* **Handles unseen data perfectly:** If a brand new, never-before-seen city appears in your test set (e.g., "Liverpool"), the model simply assigns it `0` across all top-category columns. It interprets the new category exactly the same way it interpreted rare labels during training.
* **Model compatibility:** It is perfectly suitable for linear models.

### **4. Limitations to Consider**
* **Still expands the feature space:** While mitigated, you are still adding columns. If you have many categorical variables and choose a high "Top N" threshold, the dataset can still become quite wide.
* **Loses rare label information:** By grouping all rare labels into an "all-zero" bucket, you destroy any specific predictive power a particular rare label might have had. 
* **No added value:** Unlike techniques like Target Encoding, OHE doesn't add any new mathematical information or insights about the target variable; it purely restructures the existing data.

### **5. Implementation Tools**
To apply this in your own projects, you can use:
* **Pandas:** By finding the `.value_counts().head(n)` to identify the top categories, and then using `np.where` or custom list comprehensions alongside `pd.get_dummies()`.
* **Feature-engine:** Contains a dedicated `OneHotEncoder` where you can pass the `top_categories` parameter to easily automate this across your pipeline.

# Video 3: This is a great practical continuation. Now that the theory is established, this video walks through how to actually build this encoding step-by-step from scratch using **Pandas** and **NumPy**, rather than relying on a pre-built library. 

Here is your study guide for implementing **One-Hot Encoding for Top Categories**.

### **1. The Goal and Context**
Using the *House Prices* dataset, the instructor focuses on three categorical variables: `neighborhood` (25 categories), `exterior1st` (15 categories), and `exterior2nd` (16 categories). 

If you used standard One-Hot Encoding (like `pd.get_dummies()`) on all of them, you would generate **53** new binary columns. By limiting the encoding to only the **top 10** categories per variable, the feature space is reduced to **30** columns, saving memory and reducing noise.

### **2. Crucial Prerequisite: The Train/Test Split**
Before doing *any* encoding, the dataset must be split into a training set and a testing set. 
* **Why?** The list of the "most frequent categories" is a parameter that your model needs to learn **strictly from the training set**. If you find the top categories using the entire dataset, you are leaking information from the test set into your training process (Data Leakage).

### **3. The Pandas & NumPy Implementation (Step-by-Step)**
Since Pandas does not have a built-in `top_n` parameter for its `get_dummies()` function, you have to build the logic manually. Here is the workflow described in the video:

* **Step 1: Find the frequencies.** Use `df['variable'].value_counts()` to count how many times each category appears.
* **Step 2: Sort and slice.**
  Sort the counts in descending order (highest to lowest) and slice the top 10 using `.head(10)`.
* **Step 3: Capture the category names.**
  Extract the index of that sliced series (which contains the actual category names like 'Names', 'CollgCr', 'OldTown') and save them into a Python list.
* **Step 4: Generate the binary columns.**
  Loop over your new list of top categories. For each category in the list, create a new column in your dataframe (e.g., `neighborhood_CollgCr`). Use a conditional function (like `np.where`) to populate the column: **1** if the original row matches the category, and **0** if it does not.

### **4. Automating the Process**
To avoid repeating those steps for every single categorical variable, the instructor recommends wrapping the logic into reusable Python functions:
1.  **A function to identify top categories:** It takes a dataframe, a variable name, and a target number (*n*), and returns the list of the top *n* categories.
2.  **A function to encode:** It takes the dataframe, the variable name, and the list of top categories, and runs the loop to append the new binary columns. 

# Video 4: Here is the study guide for the third video, focusing on how to vastly simplify the Top-N One-Hot Encoding process using the **Feature-engine** library. 

As promised in the previous section, I have also included the actual Python code snippets below so you can compare the manual Pandas approach with the automated Feature-engine approach.

### **1. The Goal: Streamlining with Feature-engine**
While doing this manually with Pandas and NumPy is great for understanding the underlying logic, it requires writing custom functions, managing lists of categories, and looping through columns. 

**Feature-engine** is a third-party library designed specifically for feature engineering. Its `OneHotEncoder` class has built-in parameters to handle top frequent categories automatically, saving you from writing custom boilerplate code.

### **2. Key Parameters in Feature-engine's `OneHotEncoder`**
When setting up the encoder, there are three main arguments to pay attention to:
* `top_categories=10`: This is the magic parameter. By setting this to an integer (like 10), the encoder knows to only create dummy variables for the top 10 most frequent categories in each column.
* `variables=None`: If left as `None`, the encoder will automatically detect all categorical (`object` or `category` type) columns in your dataframe. Alternatively, you can pass a specific list of column names (e.g., `['neighborhood', 'exterior1st']`).
* **`drop_last=False` (CRITICAL):** In standard One-Hot Encoding, you sometimes drop the last dummy variable to avoid perfect collinearity (the "dummy variable trap"). However, when encoding *only* top categories, you must set this to `False`. If you set it to true, it will only generate 9 columns instead of 10.

### **3. The Implementation Workflow**
Just like Scikit-learn models, Feature-engine uses the `fit()` and `transform()` methodology:
1.  **`fit(X_train)`:** The encoder scans the training data, counts the frequencies, and memorizes the top 10 categories for each variable. You can inspect what it learned by calling the `encoder.encoder_dict_` attribute.
2.  **`transform(X_train)` and `transform(X_test)`:** The encoder generates the new binary columns and drops the original categorical columns automatically.

---

### **Code Comparison: Pandas vs. Feature-engine**

Here is how the code actually looks when you apply both methods to your Jupyter Notebook.

#### **Method A: The Manual Pandas/NumPy Way (From the previous video)**
```python
import pandas as pd
import numpy as np

# 1. Function to find top categories
def find_top_categories(df, variable, top_n=10):
    return [
        x for x in df[variable].value_counts().sort_values(ascending=False).head(top_n).index
    ]

# 2. Function to encode
def encode_top_categories(df, variable, top_categories):
    for cat in top_categories:
        # Creates a new column with 1s and 0s
        df[f"{variable}_{cat}"] = np.where(df[variable] == cat, 1, 0)
    return df

# 3. Execution (Assuming X_train and X_test are already split)
categorical_cols = ['neighborhood', 'exterior1st', 'exterior2nd']

for col in categorical_cols:
    # Learn from Train ONLY
    top_cats = find_top_categories(X_train, col, top_n=10)
    
    # Apply to Train and Test
    X_train = encode_top_categories(X_train, col, top_cats)
| **Interpretability** | Moderate | High (explicit "Rare" label) |
| **Rare handling** | Implicit (all-zero baseline) | Explicit (labeled) |
| **Unseen handling** | Automatic (? all-zeros) | Automatic (? "Rare") |
| **Library support** | Feature-engine, Scikit-learn | Feature-engine, pandas |
| **Deployment** | Good | Excellent |
| **Overfitting risk** | Reduced | Reduced |
| **Information preserved** | Most categories | Grouped rare info |

---

## ? Practical Workflow Example

```python
from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Group rare labels explicitly
rare_grouper = RareLabelEncoder(tol=0.05, n_categories=4, variables=None)
rare_grouper.fit(X_train)
X_train = rare_grouper.transform(X_train)
X_test = rare_grouper.transform(X_test)

# 3. One-Hot encode the grouped data
ohe = OneHotEncoder(top_categories=10, drop_last=False, variables=None)
ohe.fit(X_train)
X_train_encoded = ohe.transform(X_train)
X_test_encoded = ohe.transform(X_test)

# 4. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# 5. Evaluate
y_pred = model.predict_proba(X_test_encoded)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

# 6. Save encoders for production
import pickle
pickle.dump([rare_grouper, ohe], open('encoders.pkl', 'wb'))
```

---

## ?? Next Steps

1. **Inspect your data** - Use `.value_counts()` to understand cardinality
2. **Experiment** - Try different tolerance thresholds
3. **Compare models** - Build with/without rare label grouping
4. **Document decisions** - Record why you grouped categories a certain way
5. **Validate** - Test on holdout set that the grouping generalizes
6. **Deploy safely** - Save your encoder configuration
