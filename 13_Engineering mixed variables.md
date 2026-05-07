# Engineering Mixed Variables: Complete Learning Guide

> **TL;DR Quick Decision:** Need to handle variables with both text and numbers?
> - **Mixed Across Observations:** Column mixes rows of pure strings and pure numbers. Strategy: Coerce numbers, extract remaining to a category column.
> - **Mixed Within Observations:** Single row has joined text/numbers (e.g., 'C85'). Strategy: Regex/string indexing to separate letters from digits.

## Quick Decision Guide

| Problem Type | Example Data | Best Approach | Key Pandas Tool |
|--------------|--------------|---------------|-----------------|
| Mixed *Across* Rows | `1`, `2`, `D`, `A` | Coerce to numeric, capture leftovers | `pd.to_numeric(errors='coerce')` |
| Mixed *Within*, Clean | `C85`, `B12` | String index for char, regex for number | `.str[0]`, `.str.extract('(\d+)')` |
| Mixed *Within*, Dirty | `A/5 21171`, `113803` | Split by space, parse conditionally | `.str.split()`, `np.where()` |

---

## Part 1: Why Engineer Mixed Variables?

Variables of mixed types contain both numbers and strings. Machine learning models generally expect clean numerical inputs or distinct categorical variables. These mixed variables cannot be used in their raw form and must be separated. 

The goal is to split the original column into two separate variables:
1. A numerical variable
2. A categorical variable

---

## Part 2: Types of Mixed Variables

### 2.1 Mixed Across Observations
These are variables where a single observation contains *either* a number *or* a string, but the column as a whole contains both.

**Example:** `number of missed payments` 
* `0`, `1`, `2` (Numerical)
* `D` (Categorical: Defaulted)

**Result Structure:**

| Original | Derived Categorical | Derived Numerical |
| :--- | :--- | :--- |
| `0` | *Missing* | `0` |
| `D` | `D` | *Missing* |

### 2.2 Mixed Within the Same Observation
These are variables where both letters and numbers are combined together inside the *exact same* data point. Treating them as purely categorical is rarely useful due to high cardinality.

**Example:** `cabin` (`C85`), `ticket` numbers, vehicle registration.

**Result Structure:**

| Original | Derived Categorical | Derived Numerical |
| :--- | :--- | :--- |
| `C85` | `C` | `85` |

---

## Part 3: Implementation Guide (Pandas)

### 3.1 Handling Variables Mixed *Across* Observations

For columns where a single row is *either* a number *or* a string.

```python
import pandas as pd
import numpy as np

# 1. Extract the numerical part
# errors='coerce' forces strings to become NaN
df['numerical_part'] = pd.to_numeric(df['mixed_column'], errors='coerce')

# 2. Extract the categorical part
# If the numerical part is NaN, pull the string from the original. Else, make it NaN.
df['categorical_part'] = np.where(
    df['numerical_part'].isnull(), 
    df['mixed_column'], 
    np.nan
)
```

### 3.2 Handling Variables Mixed *Within* Observations (Clean)

For structured text like the Titanic `cabin` variable (e.g., "C85").

```python
# 1. Extract numeric part using regex targetting digits (\d+)
df['cabin_num'] = df['cabin'].str.extract('(\d+)')

# 2. Extract categorical part by grabbing the character at index 0
df['cabin_cat'] = df['cabin'].str[0]
```

### 3.3 Handling Variables Mixed *Within* Observations (Dirty)

For messy unstructured text like the Titanic `ticket` variable (e.g., "A/5 21171" or "112053").

```python
# 1. Extract numerical part (usually the second element after a split)
df['ticket_num'] = pd.to_numeric(
    df['ticket'].str.split().str[1], 
    errors='coerce'
)

# 2. Extract the categorical part (usually the first element)
first_part = df['ticket'].str.split().str[0]

# If the first part is purely digits, it's not a category (make NaN). Otherwise keep it.
df['ticket_cat'] = np.where(
    first_part.str.isdigit(), 
    np.nan, 
    first_part
)
```