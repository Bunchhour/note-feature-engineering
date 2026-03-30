# Variable Characteristics for Feature Engineering
**Understanding Data Quality Issues & How to Handle Them**

---

## Table of Contents
1. [Quick Diagnosis Guide](#quick-diagnosis)
2. [Missing Data (Missingness)](#missing-data)
3. [Cardinality (Too Many Categories)](#cardinality)
4. [Rare Labels (Infrequent Categories)](#rare-labels)
5. [Probability Distributions](#distributions)
6. [Outliers (Extreme Values)](#outliers)
7. [Linear Model Assumptions](#linear-assumptions)
8. [Feature Magnitude (Scale Issues)](#magnitude)
9. [Quick Reference Matrix](#reference-matrix)

---

## Quick Diagnosis Guide {#quick-diagnosis}

**Does your variable have a problem? Use this checklist:**

```
ISSUE SPOTTING FLOWCHART:

1. Missing Values Present?
   ├─ YES → See MISSING DATA section
   └─ NO → Go to question 2

2. Categorical variable with MANY unique values?
   ├─ YES → See CARDINALITY section
   └─ NO → Go to question 3

3. Some categories appear VERY RARELY?
   ├─ YES → See RARE LABELS section
   └─ NO → Go to question 4

4. Using Linear Regression?
   ├─ YES → Check LINEAR MODEL ASSUMPTIONS
   └─ NO → Go to question 5

5. Numbers have EXTREME values (outliers)?
   ├─ YES → See OUTLIERS section
   └─ NO → Go to question 6

6. Using algorithms like KNN, SVM, or Neural Networks?
   ├─ YES → Check FEATURE MAGNITUDE section
   └─ NO → You may be ready to train!
```

---

## Missing Data (Missingness) {#missing-data}

### What is Missing Data?

**Missing data** (or missing values) refers to the **absence of information** for a specific observation within a variable. It's one of the most common data quality issues you'll encounter, and it can significantly impact your model's performance.

### Why Does Data Go Missing?

Understanding *how* data went missing is crucial for choosing the right solution.

| **Cause** | **Description** | **Example** |
|---|---|---|
| **Manual Entry & Omissions** | People skip non-compulsory fields or don't know the answer | Survey respondent skips income question |
| **Undefined Values** | The value logically doesn't exist | Debt-to-income ratio for someone with $0 income |
| **Matching Errors** | Mismatches when joining datasets | Postal code doesn't match between two tables |
| **Data Collection Errors** | Equipment failure, recording mistakes | Sensor malfunctions, data entry typos |
| **Intended Absence** | Field not applicable to this record | Spouse income for a single person |

### The Three Mechanisms of Missing Data

Identifying *why* data is missing helps you choose the right imputation strategy. This is more important than it seems!

#### 1. Missing Completely At Random (MCAR)

**Definition:** The probability of data being missing is the **same for all observations** and has **no relationship** with any other variable in the dataset.

**Key Trait:** It's essentially random chance.

**Example:**
- A survey respondent's computer crashes while filling out the form, so their "Income" value is missing. The crash is completely random and unrelated to their actual income.

**Impact on Models:**
✓ Safe to ignore or handle simply  
✓ Removing these rows introduces minimal bias  
✓ The missing data is not predictive of anything  

**Best Strategy:**
- Simple deletion (remove rows with missing values)
- Random imputation with no special considerations

---

#### 2. Missing At Random (MAR)

**Definition:** There is a **systematic relationship** between the missing values and **other observed variables**. The missingness depends on information you can observe.

**Key Trait:** The missingness is predictable from other variables.

**Example:**
- Women are statistically less likely to disclose their weight in a survey than men. The missing weight values are tied to the "Gender" variable.
- Older people are less likely to fill out a web form. Missing values correlate with age.

**Impact on Models:**
⚠️ Can introduce bias if ignored  
✓ Bias can be reduced by including the related variable in your model  
✓ More sophisticated imputation works well  

**Best Strategy:**
- Include the related variable (Gender, Age) in your model
- Use advanced imputation that considers other variables
- Multiple imputation techniques

---

#### 3. Missing Not At Random (MNAR)

**Definition:** The missingness is **directly related to the unobserved value itself**. People don't answer because of their true value.

**Key Trait:** The missing data is highly predictive and non-random.

**Example:**
- Patients with severe depression skip a depression survey **because they are too depressed to complete it**. Missing = likely severe depression
- Employees with very high salaries are less likely to disclose salary information. Missing = likely high earner
- Job applicants with low test scores don't submit their results. Missing = likely poor performance

**Impact on Models:**
🔴 Very problematic if ignored  
⚠️ Bias cannot be completely removed  
💡 The missingness itself is predictive information  

**Best Strategy:**
- Create a **"Missing Indicator" variable** (flag: 1=missing, 0=not missing)
  - Example: `Income_missing = 1` is predictive of low income
- Do NOT simply impute with mean/median—you lose important information
- Consider this missing pattern as a real, predictive feature
- Use domain knowledge to decide how to handle

---

### How to Identify Which Type You Have

**Use these questions:**

1. **Is the missingness random or systematic?**
   - Check if missing values cluster in certain groups (indicates MAR/MNAR)
   - Does missingness correlate with other variables? (indicates MAR)

2. **Can you predict missingness from observed data?**
   - YES → Likely MAR or MNAR
   - NO → Could be MCAR

3. **Is the missingness likely related to the unobserved value?**
   - YES → Likely MNAR
   - NO → Likely MAR or MCAR

---

### Handling Missing Data: Action Plan

| **Missing Type** | **Impact** | **Best Action** |
|---|---|---|
| **MCAR** | Low bias risk | Delete rows OR simple imputation (mean/median) |
| **MAR** | Medium bias risk | Keep related variable in model + imputation |
| **MNAR** | High bias risk | Create missing indicator + imputation |

---



## Cardinality (Too Many Categories) {#cardinality}

### What is Cardinality?

**Cardinality** refers to the **number of unique categories** (distinct values) within a categorical variable.

**Examples:**
- Gender typically has cardinality of **2** (Male, Female) → **Low Cardinality**
- City in a country has cardinality of **hundreds** → **High Cardinality**
- Postcode has cardinality in the **thousands** → **Very High Cardinality**

---

### Low vs. High Cardinality

| **Type** | **Definition** | **Examples** | **Issues** |
|---|---|---|---|
| **Low Cardinality** | Very few unique categories | Gender (2), Yes/No (2), Color (5-10) | Usually fine to encode & use |
| **High Cardinality** | Many unique categories | City (100+), Zip code (1000+), ID codes | Major preprocessing challenges |

---

### Why High Cardinality is a Problem

High cardinality introduces **three specific challenges** that must be addressed:

#### Challenge 1: String Incompatibility

**The Problem:** Categorical data is stored as text/strings (e.g., "New York", "Los Angeles"). Most algorithms like scikit-learn **do NOT accept text input**—they only work with numbers.

**What You Must Do:** Convert categories into numbers using encoding techniques.

**Common Methods:**
- **One-Hot Encoding**: Create a binary column for each category
- **Label Encoding**: Assign a unique number to each category
- **Target Encoding**: Replace each category with the average target value

---

#### Challenge 2: Train/Test Split Mismatch - Overfitting

**The Problem:** When you split data into training and test sets, high-cardinality variables often end up with unequal label distribution.

**Scenario:** 
- Training set contains City: New York, Los Angeles, Chicago
- Test set also includes: Portland, Seattle
- Some labels are only in ONE set!

**Why This Matters:**
- **Tree-based models** (Decision Trees, Random Forests) can overfit to rare labels
- Each unique label becomes a special case the model memorizes
- With hundreds of categories, the model gets overwhelmed with noise
- The model overfits to training labels that never appear in testing

---

#### Challenge 3: Train/Test Split Mismatch - Operational Errors

**The Problem:** Your model never saw certain labels during training, so it can't make predictions on them.

**Scenario:**
- Model trained on Cities: New York, Los Angeles (the frequent ones)
- Customer in production from City: Portland (rare, wasn't in training data)
- Model throws an **ERROR** because it has no rule for Portland

**Real-World Impact:**
- ❌ Model crashes in production
- ❌ Customer gets no prediction or service
- ❌ Your system experiences downtime

---

### The Solution: Reduce Cardinality

**To improve model performance AND prevent operational failures, reduce cardinality by grouping:**

#### Strategy 1: Group by Frequency
```
BEFORE (50 unique cities):
New York (8000), Los Angeles (6000), small_city_1 (15), small_city_2 (8), ...

AFTER (grouping rare cities):
New York (8000), Los Angeles (6000), Other (9000+)
```

#### Strategy 2: Domain Knowledge Grouping
```
BEFORE (100 car makes):
Toyota, Honda, Ford, BMW, Audi, Chrysler, Hyundai, ...

AFTER (grouped by region/market):
Asian (Toyota, Honda, Nissan, Hyundai, ...)
European (BMW, Audi, Mercedes, Volkswagen, ...)
American (Ford, Chevrolet, Chrysler, ...)
Other
```

#### Strategy 3: Target-Based Grouping
```
Group categories by their relationship with the target variable
(see Target Encoding or Rare Labels section)
```

---

### When Cardinality Becomes Critical

**⚠️ Watch out for cardinality when:**
- Categorical variable has **>10-20 unique values**
- Using **Tree-based algorithms** (extra prone to overfitting on high cardinality)
- Building a **production model** (must handle unseen categories)
- Very **imbalanced cardinality** (a few frequent + many rare)

---



## Rare Labels (Infrequent Categories) {#rare-labels}

### What are Rare Labels?

**Rare labels** are categories within a categorical variable that appear in only a **tiny proportion** of your dataset.

**Examples:**
- In a City variable: New York (40,000 observations) vs. Leavenworth (5 observations)
- In a Vehicle Make: Toyota (25,000) vs. Citroën (12)
- In a Building Material: Wood (15,000) vs. Stone (8)

**How to Identify:**
- Plot value counts: `variable.value_counts()`
- Look for categories with <1-5% of total observations
- These are candidates for "rare label" engineering

---

### Why Rare Labels Cause Problems

Rare labels introduce three major machine learning challenges:

#### Problem 1: String Incompatibility

**Issue:** Like all categorical data, rare labels are text and must be encoded to numbers.

**Challenge:** Encoding a rare category that appears only 5 times in 10,000 rows:
- Model has very little information to learn from
- The encoded number is essentially noise

---

#### Problem 2: Train/Test Split Mismatch (Same as Cardinality)

When splitting data, rare labels create two problems:

**a) Overfitting Risk:**
- Rare label lands only in training set
- Model overfits to that specific rare observation
- Model fails when similar customers without that label appear in test set

**b) Operational Error Risk:**
- Rare label lands only in test/production set
- Model never saw this label in training
- Model throws an error or crashes

---

#### Problem 3: Signal vs. Noise Problem (The Biggest Issue!)

**The Core Problem:** With very few samples, it's impossible to distinguish between **true patterns (signal)** and **random variation (noise)**.

**Real Example: House Price Prediction**

```
Building Material Frequency & Avg Price:
─────────────────────────────────────────
Wood          15,000 houses → Avg price: $280,000 (reliable)
Brick         8,000 houses  → Avg price: $310,000 (reliable)
Stone         45 houses     → Avg price: $850,000 (NOISE!)
Glass         12 houses     → Avg price: $155,000 (NOISE!)
```

**What's Happening:**
- Stone houses average $850,000, but with only 45 samples:
  - Is this a real effect? (Signal)
  - Or just random coincidence? (Noise)
  - Maybe those 45 houses happened to be in a wealthy area
  - Or maybe a few extremely expensive stone mansions skew the average
  

**For the model:**
- It tries to learn: "If building material = Stone → predict $850,000"
- But this "pattern" is unreliable and won't generalize
- When new stone houses appear, they're average price, not $850,000

---

### The Solution: Group Rare Labels

**The most effective feature engineering technique is to combine all rare labels into a single "Rare" or "Other" category.**

#### Example Transformation:

**BEFORE:**
```
Building Material (50 unique values):
Wood (15,000), Brick (8,000), Metal (3,000), Stone (45), 
Concrete (32), Glass (12), Marble (7), ... [44 more tiny categories]
```

**AFTER:**
```
Building Material (4 categories):
Wood (15,000), Brick (8,000), Metal (3,000), Other (1,200+)
```

#### Implementation Steps:

1. **Count value frequencies:**
   ```python
   counts = df['material'].value_counts()
   ```

2. **Define a threshold** (e.g., 1% of total):
   ```python
   threshold = len(df) * 0.01  # 1% threshold
   rare_labels = counts[counts < threshold].index
   ```

3. **Group rare labels:**
   ```python
   df['material'] = df['material'].apply(
       lambda x: x if x not in rare_labels else 'Other'
   )
   ```

---

### Benefits of Grouping Rare Labels

✅ **Reduced overfitting:** "Other" has enough samples to learn from  
✅ **Better stability:** "Other" category is consistent in train/test  
✅ **Signal vs. noise:** Avoids learning from random patterns  
✅ **Production safety:** New rare labels map to "Other" (no errors)  
✅ **Simpler model:** Fewer categories = easier to interpret  

---

### Decision: When to Group Rare Labels

| **Rare Label %** | **# Samples** | **Action** |
|---|---|---|
| **>5%** | >500 | KEEP - Large enough sample |
| **1-5%** | 100-500 | CONSIDER - Depends on context |
| **<1%** | <100 | GROUP - Almost always group |

---

### Key Difference: Cardinality vs. Rare Labels

- **Cardinality Issue:** Many categories, some might be rare
- **Rare Label Issue:** Specific focus on the infrequent ones

**Solution for both:** Group/reduce the categories!

---



## Probability Distributions {#distributions}

### What is a Probability Distribution?

A **probability distribution** is a mathematical description of how likely a variable is to take on different values.

**Key Rules:**
- All probabilities sum to exactly 1 (100%)
- Each individual probability is between 0 and 1
- The shape tells you how values are spread

**Visual Example:**
- Height of human adults: Most people cluster between 160-180 cm, very few are 200+ cm
- House prices: Most houses cluster in a certain price range, some expensive outliers exist

---

### Distribution Shapes & Why They Matter

The **shape of your variable's distribution directly determines which preprocessing techniques you should use**. This is crucial!

---

### Type 1: Normal (Gaussian) Distribution

#### Characteristics:

```
        ↑ Frequency
        |     ╱╲
        |    ╱  ╲
        |   ╱____╲
        |  ╱      ╲
        |_╱________╲___ Variable Value
          ←mean→
```

- **Shape:** Symmetric bell curve
- **Center Peak:** Most observations cluster around the middle
- **Tails:** Equal and thin on both sides
- **Key Property:** Mean = Median = Mode (all at the center)

#### Natural Examples:
- Height of adults
- Blood pressure
- IQ scores
- Many natural phenomena

#### In Datasets:
- Annual income in a stable job market
- Test scores across a population

#### Why It's Good for ML:
✅ Most statistical methods assume normal distribution  
✅ Many algorithms perform best with normal data  
✅ Easy to apply standard techniques  

---

### Type 2: Skewed Distributions

#### Left-Skewed (Negative Skew)

```
    ↑ Frequency
    |         ╱╲
    |        ╱  ╲
    |       ╱    ╲___
    |      ╱         ╲
    |_____╱___________╲__
         tail        mean
    (long left tail)
```

- **Shape:** Long tail pointing LEFT
- **Peak:** Shifted to the right
- **Order:** Mean < Median < Mode
- **Interpretation:** Most values are high, a few low outliers

#### Real Examples:
- Retirement ages (most people retire 60-70, few retire very young/old)
- Exam scores when exam is easy (most get high scores, few fail)
- Age of death (most live to 70+, some die young)

#### Right-Skewed (Positive Skew)

```
    ↑ Frequency
    |  ╱╲
    |  ╱ ╲____
    | ╱      ╲
    |╱        ╲___
    ╱____________╲___
    ↑           (long right tail)
   mean
```

- **Shape:** Long tail pointing RIGHT
- **Peak:** Shifted to the left
- **Order:** Mode < Median < Mean
- **Interpretation:** Most values are low, a few high outliers pull the mean right

#### Real Examples:
- **House prices** (most houses $200K-400K, few multi-million dollar mansions)
- **Salary** (most workers $30K-80K, few CEOs earn millions)
- **Website visit duration** (most visits <5 min, few spend hours)
- **Income distribution** (most poor/middle class, few billionaires)

---

### Why Distribution Shape Matters for Preprocessing

**THE CRUCIAL INSIGHT:** Different distribution shapes require different preprocessing!

#### For Missing Data Imputation:

| **Distribution** | **Use This to Impute** | **Why** |
|---|---|---|
| **Normal** | Mean | Mean represents typical value well |
| **Skewed** | Median | Mean is distorted by the tail; Median is more representative |

**Example:**
```
House prices distribution (right-skewed):
$300K, $350K, $400K, $450K, $500K, $1M, $5M

Mean = $629,285 (pulled up by luxury homes)
Median = $425,000 (true center of data)

If missing → impute with MEDIAN, not mean!
```

#### For Feature Scaling:

| **Distribution** | **Scaling Technique** | **Why** |
|---|---|---|
| **Normal** | Standardization (Z-score) | Works with mean & std dev |
| **Skewed** | Robust Scaling | Uses percentiles, immune to outliers |

**Standardization Formula:** $ z = \frac{x - mean}{std\_dev} $

**Robust Scaling Formula:** $ x\_{scaled} = \frac{x - Q1}{Q3 - Q1} $ (uses median percentiles)

#### For Discretization (Binning):

| **Distribution** | **Binning Method** | **Why** |
|---|---|---|
| **Normal** | Equal Width Bins | Each bin has equal range |
| **Skewed** | Equal Frequency Bins | Each bin has equal number of observations |

**Example - Equal Width vs. Equal Frequency:**

```
House Prices (Right-Skewed):
$300K, $350K, $400K, $450K, $500K, $1M, $5M

EQUAL WIDTH (bins of $1.5M):
Bin1: $0-$1.5M (6 houses) ← Unbalanced
Bin2: $1.5M-$3M (0 houses)
Bin3: $3M-$5M (1 house)

EQUAL FREQUENCY (3 houses per bin):
Bin1: $300K-$450K (3 houses) ← Balanced
Bin2: $450K-$500K (3 houses)
Bin3: $500K-$5M (3 houses)  ← Better!
```

---

### How to Identify Distribution Shape

**Visual Methods:**
1. **Histogram:** Is it symmetrical? Does it have a long tail?
2. **Box plot:** Is the median line centered in the box? Are whiskers equal?
3. **Q-Q plot:** Do points follow the diagonal line?

**Statistical Methods:**
- **Skewness value:** 
  - = 0 → Normal
  - > 0 → Right-skewed
  - < 0 → Left-skewed
- **Kurtosis:** Measures how extreme the tails are

---

### Key Takeaway for Students

🎯 **BEFORE doing ANY preprocessing, check your variable's distribution!**

✅ Normal distribution → Use simple techniques (mean, standard scaling)  
✅ Skewed distribution → Use robust techniques (median, quantile scaling)  

This one decision affects multiple preprocessing steps downstream!

---



## Outliers (Extreme Values) {#outliers}

### What is an Outlier?

An **outlier** is a data point that deviates **so significantly** from the rest of the dataset that it appears to come from a completely different mechanism or source.

**Real Examples:**
- Most house prices $200K-$500K, but one house $50M (celebrity mansion)
- Most salaries $30K-$120K, but one person $500M (billionaire CEO)
- Most website visits 2 minutes, but one visit 6 hours (bot/crawler)

---

### Should You Keep or Remove Outliers?

**This depends entirely on your context and the use case!**

#### Keep & Investigate ✅

**When outliers represent real, important events:**

- **Fraud Detection:** Outlier transactions ARE the fraud you're looking for
- **Network Intrusion:** Unusual traffic patterns indicate attacks
- **Health Anomalies:** Extreme test results indicate serious conditions
- **Revenue Forecasting:** High-value customers are crucial business signals

**Decision:** Investigate these outliers—they're valuable!

---

#### Remove or Engineer ⚠️

**When outliers are errors or irrelevant extremes:**

- **Data Entry Errors:** Someone typed "999" instead of "99"
- **Equipment Failures:** Sensor malfunction produces impossible readings
- **One-off Anomalies:** A random event unlikely to repeat (rare weather event, accident)
- **Outliers Distort Learning:** Model learns to chase outliers instead of the main pattern

**Decision:** Remove, cap, or transform these outliers.

---

### Impact on Different Machine Learning Algorithms

**Algorithms are NOT equally affected by outliers!**

#### Sensitive to Outliers (Will Be Distorted) 🔴

**Linear Regression:**
- Tries to minimize distance to ALL points
- Even one extreme outlier pulls the regression line toward it
- Coefficients become unreliable

**Example:**
```
Normal data:   y = 0.5x (good fit)
With outlier: y = 2.0x (pulled up by one extreme point!)
```

**AdaBoost:**
- Assigns huge weight to misclassified points
- Outliers are almost always misclassified
- Model overfits to outliers trying to correct them

**Neural Networks:**
- Gradient descent optimization breaks with extreme values
- Convergence becomes unstable
- Often requires outlier removal or extreme value clipping

---

#### Robust to Outliers (Handles Well) ✅

**Tree-Based Algorithms:**
- Decision Trees, Random Forests, Gradient Boosting, XGBoost
- Make decisions by splitting at **thresholds**, not calculating distances
- Whether a value is "100" or "1000000", the split is the same
- Example: "Is Price > $500K?" works the same with any extreme value

```
Split: Is House Price > $500K?
├─ YES (whether $501K or $50M, same outcome)
└─ NO
```

---

### How to Identify Outliers Mathematically

The detection method depends on your variable's distribution!

#### Method 1: Standard Deviation Rule (Normal Distributions)

**Best for:** Variables with normal/symmetric distributions

**Rule:** In a normal distribution, ~99% of data falls within **3 standard deviations** of the mean.

**Formula:**
- **Upper Bound:** $ mean + (3 \times stdev) $
- **Lower Bound:** $ mean - (3 \times stdev) $
- **Outlier:** Any value outside these bounds

**Example:**
```
House prices: Mean=$400K, StdDev=$100K
Upper bound = $400K + (3 × $100K) = $700K
Lower bound = $400K - (3 × $100K) = $100K

Houses priced <$100K or >$700K = outliers
```

---

#### Method 2: Interquartile Range (IQR) Rule (Skewed Distributions)

**Best for:** Skewed variables (distributions not symmetric)

**Why Different:** Skewed data distorts mean and standard deviation. Use percentiles instead.

**Step 1: Find the quartiles**
- **Q1** = 25th percentile (25% of data below this)
- **Q3** = 75th percentile (75% of data below this)
- **IQR** = Q3 - Q1 (the middle 50% of data)

**Step 2: Calculate bounds**
- **Upper Bound:** $ Q3 + (1.5 \times IQR) $
- **Lower Bound:** $ Q1 - (1.5 \times IQR) $

**Step 3: Identify outliers**
- Values outside these bounds = outliers
- *Optional:* Use 3 × IQR instead of 1.5 × IQR for "extreme" outliers

**Real Example: House Prices**
```
Q1 = $300,000 (25th percentile)
Q3 = $500,000 (75th percentile)
IQR = $200,000

Upper bound = $500K + (1.5 × $200K) = $800K
Lower bound = $300K - (1.5 × $200K) = $0K

Houses <$0K or >$800K = outliers
```

---

### Visualizing Outliers: The Box Plot

A box plot is the visual implementation of the IQR rule:

```
        Single Dots = Outliers (beyond whiskers)
        ·  ·
        
    |────────────────────────────|
    ←─ Lower Whisker (Q1-1.5×IQR)
    
    |─────────────────────────────|  ← Box (Q1 to Q3)
    |                |                IQR = middle 50%
    ├─ 25th %       ├─ Median (50th %)
    
                    |────────────────────────────|
                    Lower IQR ─→|
                                |← Upper Whisker (Q3+1.5×IQR)
```

**Reading a Box Plot:**
- **Box:** Middle 50% of data (Q1-Q3)
- **Middle line:** Median (50th percentile)
- **Whiskers:** Extend to IQR bounds
- **Dots:** Individual outliers beyond whiskers

---

### Handling Outliers: Decision Guide

| **Situation** | **Action** | **Method** |
|---|---|---|
| **Data Entry Error** | Remove | Delete the row |
| **Equipment/Sensor Failure** | Remove | Delete the row |
| **Real but Extreme** | Cap/Clamp | Set to max reasonable value |
| **Important Signal** | Keep | Keep & use in model |
| **Distorting Linear Model** | Transform | Log, Box-Cox, or scaling |

---

### Common Outlier Handling Techniques

**1. Removal:** Delete rows with outliers
```python
df = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]
```

**2. Capping/Clipping:** Replace extreme values with bounds
```python
df['price'] = df['price'].clip(lower_bound, upper_bound)
```

**3. Transformation:** Apply log or power transformation
```python
df['price'] = numpy.log(df['price'])  # Makes right-skew less extreme
```

**4. Flag & Keep:** Create indicator variable
```python
df['price_is_outlier'] = (df['price'] > upper_bound).astype(int)
```

---

### Decision Checklist

Before handling outliers, ask yourself:

✅ Is this a **real business event** or an error?  
✅ Am I using a **sensitive algorithm** (linear models, neural nets)?  
✅ Will removing hurt my **model's generalization**?  
✅ Does my **use case care about extreme values**?  

---




## Linear Model Assumptions {#linear-assumptions}

### Overview: Why Linear Models Have Assumptions

**Linear Regression** estimates a target variable ($y$) based on a linear combination of predictors:

$$ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon $$

Where:
- $\beta_0$ = intercept (starting point)
- $\beta_n$ = coefficients (how much each variable influences $y$)
- $\epsilon$ = error/residual (observed - predicted)

**BUT:** For the coefficients to be **reliable and unbiased**, the data must satisfy four mathematical assumptions.

---

### The Four Critical Assumptions

#### Assumption 1: Linearity

**What it means:** The relationship between each predictor ($x$) and the target ($y$) is a **straight line** (not curved, not S-shaped).

**Why it matters:** If the relationship is curved, a linear model can't capture it properly.

**How to Check:**
- Create **scatter plots** of each variable vs. the target
- Do the points roughly form a straight line?
- Or do they curve upward/downward?

**Visual Examples:**
```
LINEAR (✓)           CURVED (✗)           S-SHAPED (✗)
  y                    y                      y
  |     ╱              |    ╱╱              |╱╱
  |   ╱                |  ╱     (curved up)  ╱╱
  | ╱                  | ╱                 ╱╱
  |________________x   |________________x   |________________x
```

**If Violated:**
- Try polynomial features (add $x^2$, $x^3$)
- Non-linear models (trees, splines)

---

#### Assumption 2: Normality of Errors

**What it means:** The **residuals** (errors) follow a **normal distribution** perfectly centered at zero.

**Residual** = Actual Value - Predicted Value

**Why it matters:** If errors are skewed, the model's confidence intervals are wrong.

**How to Check:**

**Method 1: Histogram of Residuals**
- Plot a histogram of all residuals
- Should look like a symmetrical bell curve centered at 0

**Method 2: Q-Q Plot (Quantile-Quantile)**
- Compares residuals to normal distribution
- Points should lie perfectly on the diagonal line
- Deviations from line = violation

```
GOOD (Normal)       BAD (Not Normal)
      |                  |
   ╱  |                  | ╱╱
  ╱   |  (on line)       |╱ (curves away)
 ╱    |                  |
      |                  |───────────
   ───────────────────  
```

**If Violated:**
- Transform the target variable (log, sqrt, Box-Cox)
- Remove extreme outliers
- Check for missing categories

---

#### Assumption 3: Homoscedasticity (Constant Variance)

**What it means:** The **spread of errors** is **consistent** across all prediction values. Errors don't get bigger or smaller as predictions increase.

**In other words:** The variance of residuals is **constant**.

**Why it matters:** If variance changes, the model is less confident in some regions.

**How to Check:**
- Plot **Residuals vs. Fitted Values**
- Should look like a **random scattered cloud** (uniform spread)
- ❌ If it forms a **cone/funnel shape**, you have heteroscedasticity (variance changing)

```
HOMOSCEDASTIC (✓)        HETEROSCEDASTIC (✗)
  residuals                residuals
    |  ·  ·  ·              |
    | · · · · ·             | · · ·
    |  ·  ·  ·              |  ·  · · · ·
    |___·__·__·___          |____·__·_·_____
    0  fitted values      0  fitted values
         (even spread)            (funnel shape!)
```

**If Violated:**
- Variance-stabilizing transformations (log, sqrt)
- Weighted regression (give less weight to high-error regions)
- Robust regression

---

#### Assumption 4: No Perfect Multicollinearity

**What it means:** The **predictor variables are NOT highly correlated** with each other. They measure different things.

**Why it matters:** If two variables move perfectly together, the model can't tell which one is actually causing the effect.

**Problem Example:**
```
Variable A: Customer Age
Variable B: Years Since Graduation
→ These are too correlated! Hard to untangle effects.
```

**How to Check:**
- Create a **Correlation Matrix Heatmap**
- Look for pairs with correlation close to 1 or -1
- Use **Variance Inflation Factor (VIF)** for more detail
  - VIF > 5-10 = high multicollinearity problem

**If Violated:**
- Remove one of the correlated variables
- Combine them into one (e.g., average)
- Use regularization (Ridge, Lasso regression)
- Principal Component Analysis (PCA)

---

### Checking Assumptions: A Practical Workflow

**For each feature engineering decision, ask:**

1. **Linearity Check:** Plot variable vs. target—is it linear?
   - If curved → transform variable (log, sqrt) or engineer non-linear features
   
2. **Distribution Check:** Is the variable normally distributed?
   - If skewed → use appropriate scaling and imputation methods
   - If has outliers → identify and handle them
   
3. **Variance Check:** Is spread consistent across ranges?
   - If funnel shape → apply variance-stabilizing transformation
   
4. **Correlation Check:** Is this variable highly correlated with another feature?
   - If yes → decide which to keep or combine them

---

### Why This Matters for Feature Engineering

Linear model assumptions directly guide **what preprocessing you should do**:

| **Assumption** | **Problem** | **Preprocessing Solution** |
|---|---|---|
| **Linearity** | Curved relationship | Log transform, add polynomial features, binning |
| **Normality of Errors** | Skewed residuals | Transform target variable, remove outliers |
| **Homoscedasticity** | Uneven variance | Log/sqrt transform, robust scaling |
| **Multicollinearity** | Correlated variables | Remove variable, combine features, PCA |

---

### Important Note

⚠️ **These assumptions apply mainly to:**
- Linear Regression
- Logistic Regression
- Classical statistics

✅ **Tree-based models** (Random Forests, XGBoost) don't care about these assumptions!

---



To help you build intuition for how sensitive linear regression is to these assumptions, I've built an interactive tool below. You can introduce heteroscedasticity or drag an outlier to see exactly how it warps the regression line and the residual errors!

## Feature Magnitude (Scale Issues) {#magnitude}

### What is Feature Magnitude?

**Feature magnitude** is the **scale** or **numerical range** of a variable.

**Examples:**
- Age ranges from 0 to 100
- Salary ranges from $20,000 to $2,000,000
- Website visit duration ranges from 1 to 300,000 seconds

These variables have **vastly different scales**—and that's a problem!

---

### The Magnitude Problem

If you feed variables with **different scales** directly into certain algorithms, the variables with **larger numerical ranges will dominate**, overshadowing smaller-scale variables.

**Real Example:**
```
Feature 1: Age (ranges 18-80)
Feature 2: Income (ranges $20,000-$300,000)

In distance calculation:
Distance ≈ sqrt( (age_diff)² + (income_diff)²)
         = sqrt( (5)² + (50,000)²)
         = sqrt( 25 + 2,500,000,000)
         ≈ 50,000 (income dominates completely!)
```

**Result:** The algorithm ignores Age because Income's scale is so much larger!

---

### Why Magnitude Matters: Three Ways It Breaks Models

#### Problem 1: Linear Regression Coefficients

The linear model equation is:
$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n $$

The coefficient ($\beta$) represents: **"For a 1-unit change in X, Y changes by $\beta$"**

**The Issue:** Coefficients are directly tied to the scale of X!

**Example:**
```
Model 1: Distance in kilometers
  Price = 100,000 - (5 × distance_km)
  → Coefficient = -5

Model 2: Same distance in meters
  Price = 100,000 - (0.005 × distance_m)
  → Coefficient = -0.005
```

**Same relationship, completely different coefficients!**

If variables have vastly different scales, some coefficients become huge and others tiny—making them hard to compare or interpret.

---

#### Problem 2: Distance-Based Algorithm Dominance

Many algorithms calculate **Euclidean distance** between points (KNN, K-Means, SVM, etc.):

$$ Distance = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2 + \dots } $$

**The Issue:** Large-scale variables dominate the calculation.

**Example:**
```
Person A: Age=30, Salary=$50,000
Person B: Age=31, Salary=$150,000
Person C: Age=30, Salary=$150,000

Distance(A,B) = sqrt((1)² + (100,000)²) ≈ 100,000
Distance(A,C) = sqrt((0)² + (100,000)²) ≈ 100,000

Age difference is mathematically irrelevant!
(1 vs 0 is tiny compared to 100,000)
```

**Result:** Age is ignored, Salary dominates the distance calculation.

---

#### Problem 3: Gradient Descent Optimization

**Neural Networks** and many iterative algorithms use **gradient descent** to find optimal coefficients.

**The Issue:** Unscaled features cause inefficient optimization.

**What Happens:**
- Large-scale features have steep gradients
- Small-scale features have shallow gradients
- Optimization bounces around erratically
- Takes much longer to converge
- May converge to suboptimal solution

**Visual:**
```
UNSCALED (bouncy):      SCALED (smooth):
   |↗  ↙               |    ╱
   |↙  ↗               |  ╱╱
   |↗  ↙    (zig-zag)  |╱╱╱ (steady)
   |↙  ↗               |╱
```

---

### Which Algorithms Require Feature Scaling?

#### SENSITIVE to Magnitude (Requires Scaling) 🔴

**Must scale before using:**
- **Linear/Logistic Regression** - coefficients are scale-dependent
- **Neural Networks** - gradient descent fails with unscaled data
- **Support Vector Machines (SVM)** - distance-based
- **K-Nearest Neighbors (KNN)** - distance-based
- **K-Means Clustering** - distance-based
- **Principal Component Analysis (PCA)** - variance-based
- **Linear Discriminant Analysis (LDA)** - means and covariances

**Rule of Thumb:** If the algorithm uses **distance, means, or gradients** → scale it!

---

#### INSENSITIVE to Magnitude (Does NOT require scaling) ✅

**No scaling needed:**
- **Decision Trees** - splits at thresholds: "Is Age > 30?" works same regardless of scale
- **Random Forests** - ensemble of trees
- **Gradient Boosting** - ensemble of trees
- **XGBoost, LightGBM, CatBoost** - tree-based
- **Naive Bayes** - works with probabilities

**Why Trees Are Immune:**
```
Split: Is Salary > $50,000?
├─ YES (whether $50,001 or $500,000, same outcome)
└─ NO

Split: Is Age > 30?
├─ YES (whether 31 or 80, same outcome)
└─ NO

Magnitude doesn't matter—only the threshold matters!
```

---

### Common Scaling Techniques

#### 1. Standardization (Z-Score Scaling)

**Formula:** $$ x_{scaled} = \frac{x - mean}{standard\_deviation} $$

**Result:**
- Mean = 0
- Standard deviation = 1
- Data ranges roughly -3 to +3

**Best for:**
- Linear/Logistic Regression
- Neural Networks
- Algorithms assuming normal distribution

**Example:**
```
Original:  [10, 20, 30, 40, 50]  (mean=30, stdev≈15.8)
Scaled:    [-1.27, -0.64, 0, 0.64, 1.27]
```

---

#### 2. Min-Max Scaling (Normalization)

**Formula:** $$ x_{scaled} = \frac{x - min}{max - min} $$

**Result:**
- All values between 0 and 1

**Best for:**
- Bounded output algorithms (NNs with sigmoid/tanh)
- When you want intuitive 0-1 range

---

#### 3. Robust Scaling

**Formula:** $$ x_{scaled} = \frac{x - median}{IQR} $$

**Result:**
- Uses percentiles instead of mean/stdev
- Less affected by outliers

**Best for:**
- Data with outliers
- Skewed distributions

---

### Scaling Decision Guide

| **Algorithm** | **Type** | **Scale?** | **Method** |
|---|---|---|---|
| Linear Regression | Distance/Gradient | YES | Standardization |
| Logistic Regression | Distance/Gradient | YES | Standardization |
| Neural Network | Gradient | YES | Standardization |
| KNN | Distance | YES | Min-Max or Standardization |
| SVM | Distance | YES | Standardization |
| K-Means | Distance | YES | Standardization |
| PCA | Variance | YES | Standardization |
| **Decision Tree** | **Threshold** | **NO** | N/A |
| **Random Forest** | **Threshold** | **NO** | N/A |
| **XGBoost** | **Threshold** | **NO** | N/A |
| Naive Bayes | Probability | NO | N/A |

---

### Practical Checklist

Before training your model, ask:

✅ Am I using a **distance-based or gradient-based algorithm**?  
✅ Do my features have **drastically different scales** (100 vs 1,000,000)?  
✅ Is my algorithm on the **"Requires Scaling"** list?  

If yes to all → **SCALE YOUR FEATURES!**

---

## Quick Reference Matrix {#reference-matrix}

### All Variable Characteristics at a Glance

| **Characteristic** | **What It Is** | **Impact** | **Action** |
|---|---|---|---|
| **Missing Data (MCAR)** | Random missing values | Low bias | Delete rows OR simple imputation |
| **Missing Data (MAR)** | Systematic missing (related to other variable) | Medium bias | Keep related variable + imputation |
| **Missing Data (MNAR)** | Missing due to the value itself | High bias | Create missing indicator + imputation |
| **High Cardinality** | Many unique categories (100+) | Overfitting, operational errors | Group rare categories |
| **Rare Labels** | Categories appearing <1% | Noise, overfitting | Group into "Other" category |
| **Normal Distribution** | Bell-shaped, symmetric | Good for standard techniques | Use mean/standardization |
| **Skewed Distribution** | Long tail on one side | Distorts mean | Use median/robust scaling/equal-freq binning |
| **Outliers** | Extreme values | Distorts linear models | Identify with IQR rule → keep/remove/cap |
| **Non-Linear Relationship** | Curved, not straight | Linear model fails | Transform variable or use non-linear model |
| **Non-Normal Errors** | Residuals not normal | Confidence intervals wrong | Transform target or remove outliers |
| **Heteroscedasticity** | Uneven error variance | Predictions less reliable | Variance-stabilizing transformation |
| **Multicollinearity** | Correlated variables | Can't distinguish effects | Remove variable or combine |
| **Different Scales** | Variables have different ranges | Large-scale vars dominate | Scale (if using distance/gradient algorithm) |

---

## Decision Flowchart: Complete Data Cleaning Process

```
START: New Variable

1. MISSINGNESS
   ├─ Has missing? → Identify type (MCAR/MAR/MNAR) → Handle appropriately
   └─ No missing → Go to 2

2. OUTLIERS
   ├─ Distribution check: Normal or Skewed?
   ├─ Identify outliers (StdDev or IQR method)
   └─ Keep/Remove/Cap based on context

3. DISTRIBUTION SHAPE
   ├─ Normal? → Use standard techniques
   ├─ Skewed? → Use robust techniques
   └─ Other? → Transform or bin

4. CATEGORICAL SPECIFIC
   ├─ Cardinality check: Many unique values?
   │   └─ YES → Group rare categories
   ├─ Rare labels check: Very infrequent categories?
   │   └─ YES → Group into "Other"
   └─ NO → Go to 5

5. CORRELATION CHECK
   ├─ Highly correlated with another variable?
   │   └─ YES → Remove one or combine
   └─ NO → Go to 6

6. ALGORITHM CHECK
   ├─ Using Linear/KNN/SVM/Neural Net?
   │   └─ YES → Scale the variable
   ├─ Using Trees?
   │   └─ NO → Scaling not needed
   └─ Go to 7

7. LINEAR MODEL ASSUMPTIONS (if using linear model)
   ├─ Linearity? → If no → Transform or engineer features
   ├─ Normal errors? → If no → Transform target
   ├─ Homoscedasticity? → If no → Variance-stabilizing transform
   └─ No multicollinearity? → If violated → Remove variable

DONE: Variable is ready for modeling!
```

---

## Key Takeaways for Students

🎯 **Most Common Issues & How to Handle Them:**

1. **Missing Values** → Handle differently based on why they're missing (MCAR/MAR/MNAR)
2. **Categorical with Many Values** → Group rare ones together
3. **Outliers** → Identify with IQR rule, decide based on domain knowledge
4. **Skewed Distribution** → Use median/robust methods instead of mean/standard scaling
5. **Using KNN/SVM/Neural Net?** → SCALE THE FEATURES!
6. **Using Trees?** → No scaling needed
7. **Linear Regression?** → Check all four assumptions and fix violations

✅ **Remember:** The right preprocessing choice depends on:
- Your variable's **distribution shape**
- Your variable's **characteristics** (missing, outliers, cardinality)
- Your **algorithm choice**
- Your **business context**

---

