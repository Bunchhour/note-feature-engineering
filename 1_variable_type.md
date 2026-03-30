# Data Preprocessing: Variable Types Guide
**A Student-Friendly Learning Resource for Feature Engineering**

---

## Table of Contents
1. [Quick Start: Variable Type Decision Tree](#quick-start)
2. [What is a Variable?](#definition)
3. [Variable Type Classification](#classification)
4. [Detailed Variable Types](#detailed-types)
5. [Quick Reference Summary](#reference)
6. [Common Preprocessing Steps](#preprocessing)

---

## Quick Start: Variable Type Decision Tree {#quick-start}

**Ask yourself these questions to identify variable types:**

```
Does your variable contain NUMBERS primarily?
├─ YES → Could be NUMERICAL
│   ├─ Can it be counted or whole numbers only?
│   │   ├─ YES → DISCRETE (e.g., number of items)
│   │   └─ NO → CONTINUOUS (e.g., height, weight, price)
│   └─ Only two values (0/1, Yes/No)?
│       └─ YES → BINARY (special discrete type)
│
├─ NO, contains CATEGORIES/TEXT?
│   └─ Do the categories have a natural order or ranking?
│       ├─ YES → ORDINAL (e.g., Low, Medium, High)
│       └─ NO → NOMINAL (e.g., colors, country names)
│
├─ DATES or TIMES?
│   └─ Contains dates, times, or timestamps → DATE-TIME
│
└─ MIX of NUMBERS and TEXT together?
    └─ MIXED VARIABLES
```

---

## What is a Variable? {#definition}

A **variable** is any characteristic, number, or quantity that can be measured or counted. Variables are called "variables" because their values can (and usually do) vary across different observations or individuals in a population.

### Real-World Examples:
- **Age**: 21, 35, 62 (varies across people)
- **Gender**: Male, Female (varies across people)
- **Income**: $45,000, $62,500, $120,000 (varies across people)
- **Country of Birth**: USA, UK, Canada, etc.
- **Eye Color**: Blue, Brown, Green, etc.
- **Vehicle Make**: Toyota, Ford, Honda, etc.

---

## Variable Type Classification {#classification}

Variables in a dataset fall into **two main categories** with important sub-types:

| **Main Category** | **Sub-Type** | **Description** |
|---|---|---|
| **NUMERICAL** | Discrete | Whole numbers only, countable values |
| | Continuous | Can take any value in a range, decimals allowed |
| **CATEGORICAL** | Nominal | No order or ranking (just labels) |
| | Ordinal | Has a meaningful order or ranking |
| **SPECIAL TYPES** | Binary | Only two possible values |
| | Date-time | Contains dates, times, or timestamps |
| | Mixed | Contains both text and numbers |

---

## Detailed Variable Types {#detailed-types}

### 1. NUMERICAL VARIABLES

#### 1.1 Discrete Variables

**Definition**: Variables that can only take **whole number values** that are countable. No fractions or decimals allowed.

**How to Recognize It:**
- The values form gaps when plotted (e.g., 1, 2, 3, 5... no 1.5 or 2.7)
- You can count the values
- Represents a count of something

**Real-World Examples:**
- Number of items bought in a supermarket (you can buy 25 items, but not 3.7 items)
- Number of children in a family
- Number of credit accounts opened in 12 months
- Number of house rooms

**Dataset Example:**
The number of credit accounts opened in the last 12 months. When plotted, data points jump from one whole number to the next.

**Preprocessing Tips:**
- ✓ Can often be used directly in models
- ⚠️ May need scaling for some algorithms
- 📊 Visualize with bar charts or histograms

---

#### 1.2 Continuous Variables

**Definition**: Variables that can take **any value within a range**, including infinite possibilities of decimals and fractions.

**How to Recognize It:**
- Values form a smooth range when plotted
- Measured (not counted)
- Can have decimal places
- Represents a measurement of something

**Real-World Examples:**
- Total amount paid at checkout: $32.50
- Time spent on a website: 5.1 seconds
- Height of a person: 175.3 cm
- Interest rate on a loan: 5.25%
- Temperature: 22.5°C

**Dataset Example:**
Interest rate charged on a loan. When plotted, data covers the entire range of possible rates seamlessly.

**Preprocessing Tips:**
- ⚠️ Often need scaling (standardization or normalization)
- ✓ Can be binned into discrete categories if needed
- 📊 Visualize with histograms or density plots

---

#### 1.3 Binary Variables

**Definition**: A **special type of discrete variable** that can only ever take **two possible values**.

**How to Recognize It:**
- Only 2 unique values in the entire column
- Usually coded as 0/1, Yes/No, True/False

**Real-World Examples:**
- Default on loan: 1 (defaulted) or 0 (did not default)
- Email opened: Yes or No
- Customer churned: True or False
- Approved for credit: 1 (approved) or 0 (rejected)

**Preprocessing Tips:**
- ✓ Already in good format for most models
- ⚠️ Watch for class imbalance (e.g., 95% vs 5%)
- 💡 Often the target variable in classification problems

---

### 2. CATEGORICAL VARIABLES

#### 2.1 Nominal Variables

**Definition**: Categories with **no meaningful order or hierarchy**. All labels are mathematically equivalent, just different groups.

**How to Recognize It:**
- No natural ranking between categories
- Asking "which is better/higher?" doesn't make sense
- Just different groups or types

**Real-World Examples:**
- Country of birth: USA, UK, Canada, Japan
- Eye color: Blue, Brown, Green, Hazel
- Vehicle make: Toyota, Ford, Honda, BMW
- Mobile network provider: Vodafone, Orange, T-Mobile
- Loan purpose: Debt consolidation, Car purchase, Credit card payoff

**Dataset Example:**
Home Ownership status: Own, Rent, Mortgage. These are just different categories with no ranking.

**Common Preprocessing Steps:**
1. **One-Hot Encoding**: Convert each category into a binary column
   - Example: `Color` (Red, Blue) → `Color_Red` (1,0) + `Color_Blue` (0,1)
2. **Label Encoding**: Assign a number to each category
   - Example: Red=1, Blue=2, Green=3
3. **Target Encoding**: Use the average target value for each category
4. **Frequency Encoding**: Use how often each category appears

**⚠️ Important Note:**
Categorical data is often coded as numbers (e.g., Gender: Male=0, Female=1), but this **does NOT make it numerical**! Always treat it as categorical.

---

#### 2.2 Ordinal Variables

**Definition**: Categories that have a **meaningful, natural order or ranking**. You can logically determine which comes first and last.

**How to Recognize It:**
- Clear ranking or order between categories
- Asking "which is higher/better?" makes sense
- Can be sorted from low to high

**Real-World Examples:**
- Student grades: A, B, C, D, F (F is worst, A is best)
- Education level: High School < Associate < Bachelor < Master < PhD
- Customer satisfaction: Very Dissatisfied < Dissatisfied < Neutral < Satisfied < Very Satisfied
- Days of the week: Monday < Tuesday < ... < Sunday (in sequence)
- Shirt sizes: XS < Small < Medium < Large < XL

**Dataset Example:**
Loan grade or risk level: Grade A (safest) → Grade G (riskiest)

**Common Preprocessing Steps:**
1. **Ordinal Encoding**: Map to numbers preserving order
   - Example: Low=1, Medium=2, High=3
2. **Target Encoding**: Use average target for each ordered category
3. ⚠️ **AVOID One-Hot Encoding** for ordinal data (loses the ordering information)

**💡 Why It Matters:**
Using the wrong encoding for ordinal data can hurt model performance. The order is important information!

---

### 3. SPECIAL VARIABLE TYPES

#### 3.1 Date-Time Variables

**Definition**: Data points that contain **dates, times, or a combination of both**.

**How to Recognize It:**
- Contains date format (YYYY-MM-DD)
- Contains time format (HH:MM:SS)
- Or both together

**Real-World Examples:**
- Date of birth: 1990-05-15 (date only)
- Time of accident: 14:30:00 (time only)
- Payment date & time: 2023-10-27 09:15:22 (both)

**Dataset Example:**
- `issued_date`: The exact date loan money was sent to borrower
- `last_repayment_date`: Most recent loan repayment date

**Why Special Preprocessing is Needed:**
Raw date-time cannot be fed directly into most machine learning algorithms. But they contain **valuable information** that can be extracted!

**Common Preprocessing Steps:**
1. **Extract Components**:
   - Year, Month, Day, Quarter, Week
   - Hour, Minute, Second
   - Day of week, Is weekend?, Is holiday?
2. **Create Time-Based Features**:
   - Days since an event
   - Time between two dates
   - Age at a reference date
3. **Cyclical Encoding** (for month, day of week):
   - Convert repetitive values into sine/cosine features
4. **Aggregate by Time**: Count events per day/month/year

**Example:**
```
From: issued_date = "2023-10-27"

Extract:
- year = 2023
- month = 10 (October)
- day = 27
- quarter = 4
- day_of_week = 4 (Friday)
- is_weekend = 0
```

---

#### 3.2 Mixed Variables

**Definition**: Variables that contain a **combination of both numbers and text (strings)** within their values. Two structural types exist:

**Type 1: Either Numbers OR Text (but not both in one value)**

Example: Number of credit accounts
- Could be a number: `25`, `100`, `8`
- Or a text code: `"Unknown"`, `"Unverified"`, `"Unmatched"`

Example: Number of missed payments
- Could be a number: `1`, `2`, `3`
- Or a letter code: `"D"` (Defaulted), `"A"` (Arrangement with lender)

**Type 2: Combined Strings AND Numbers (both in one value)**

Example: Cabin numbers from Titanic dataset
- `"C123"` (letter C for deck, followed by room number 123)
- `"E456"`, `"F789"`

Example: Vehicle registration plates
- `"ABC123XYZ"`

Example: Postcodes
- `"SW1A 1AA"` (UK postcode format)

**Why They Cause Problems:**
Most machine learning algorithms require numeric input. Mixed variables can't be used directly.

**Common Preprocessing Steps:**
1. **Split the components**:
   ```
   "C123" → cabin_deck = "C", cabin_number = 123
   "ABC123XYZ" → plate_letters = "ABC", plate_numbers = "123"
   ```
2. **Extract the numeric part** and treat as numerical
3. **Extract the text part** and encode as categorical
4. **Create new features** from patterns found
   - Example: Cabin deck location might be predictive
5. **Handle missing codes**: Understand what special codes mean (Unknown, Unverified, etc.)

**Opportunity:**
Mixed variables can be **richly engineered** to create powerful new features!

---

## Quick Reference Summary {#reference}

### Decision-Making Flowchart

| **Question** | **Answer** | **Variable Type** |
|---|---|---|
| Can it only be whole numbers/counts? | YES | **DISCRETE** |
| Can it have decimal values? | YES | **CONTINUOUS** |
| Only has 2 values (0/1, Yes/No)? | YES | **BINARY** |
| Has categories with NO order? | YES | **NOMINAL** |
| Has categories WITH natural order? | YES | **ORDINAL** |
| Contains dates or times? | YES | **DATE-TIME** |
| Mix of text and numbers? | YES | **MIXED** |

---

## Common Preprocessing Steps {#preprocessing}

### By Variable Type:

| **Variable Type** | **Key Preprocessing Task** | **Common Methods** |
|---|---|---|
| **DISCRETE** | Check for outliers; handle skewness | Scaling, binning, log transformation |
| **CONTINUOUS** | Scaling & normalization | Standardization, Min-Max scaling, Log/Box-Cox |
| **NOMINAL** | Encoding | One-Hot Encoding, Label Encoding, Target Encoding |
| **ORDINAL** | Preserve order during encoding | Ordinal Encoding, do NOT use One-Hot |
| **BINARY** | Handle class imbalance if needed | Balancing techniques, thresholding |
| **DATE-TIME** | Extract useful information | Extract components, create lag features, time diffs |
| **MIXED** | Split into components | Separate text from numbers, process separately |

---

## Key Takeaways for Students

✅ **ALWAYS** identify the variable type first  
✅ **DIFFERENT types require DIFFERENT preprocessing**  
✅ **Nominal and Ordinal are DIFFERENT** - handle them differently!  
✅ **Do NOT encode numeric-coded categorical variables** directly into models  
✅ **Date-time variables are opportunities**, not problems - extract features!  
✅ **When in doubt**, explore the data visually and check the variable's meaning  

---

## Next Steps

1. **Know your data**: Correctly identify each variable's type
2. **Choose preprocessing wisely**: Based on type and use case
3. **Validate your choices**: Does the preprocessing make sense for the business context?
4. **Iterate**: Experiment with different preprocessing approaches







