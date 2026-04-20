from pathlib import Path

file_path = Path(r'C:\Users\USER\note-feature-engineering\9_ Variable Transformation.md')

# ========== HEADER AND QUICK DECISION GUIDE ==========
header = """# Variable Transformation: Complete Student Learning Guide

> **Quick Decision Guide:** Need to transform your data? Use this flow chart to choose the right method.

## Quick Decision Flowchart

\\\
Do you have a Linear Model (LR, Logistic, ANOVA)?
├─ NO → Don't transform (skip this entire guide)
└─ YES → Is your data skewed or non-normal?
    ├─ NO → Don't transform
    └─ YES → Do you know which transformation?
        ├─ YES → Use NumPy (np.log, np.sqrt, etc.)
        └─ NO → Do your values include negative or zero?
            ├─ NO (all positive) → Use Box-Cox
            │   └─ For learning: scipy.stats.boxcox()
            │   └─ For pipelines: PowerTransformer(method='box-cox')
            │   └─ For production: BoxCoxTransformer()
            └─ YES (has negatives/zeros) → Use Yeo-Johnson
                └─ For learning: scipy.stats.yeojohnson()
                └─ For pipelines: PowerTransformer(method='yeo-johnson')
                └─ For production: YeoJohnsonTransformer()
\\\

| Scenario | Use This | Why |
|----------|----------|-----|
| Linear/Logistic Regression with skewed data | Box-Cox (positive only) or Yeo-Johnson (any values) | Normalizes residuals, satisfies model assumptions |
| Tree-based models (Random Forest, XGBoost) | Don't transform | Transformation-invariant, distribution irrelevant |
| Neural Networks | Standardize, not transform | Care about scale, not distribution shape |
| Learning/Exploration | SciPy functions directly | Easy to understand, see lambda values |
| Production pipelines | Scikit-learn or Feature-engine | Reproducible, trainable transformers |
| Real-world messy data | Feature-engine | Auto-detects types, handles mixed data |

---

"""

restructured = [header]

# ========== WHY, WHEN, AND THEORY ==========
why_when = """## Part 1: Why Transform Variables?

### 1.1 The Primary Goal
Variable transformation makes data "behave" better for specific mathematical models by:
- **Normalizing Distributions:** Converting skewed data to Gaussian (Normal) distributions
- **Satisfying Model Assumptions:** Enabling linear models to give unbiased results
- **Improving Model Performance:** Helping models accurately capture relationships between variables

### 1.2 Key Assumptions of Linear Models

| Assumption | What It Means | Why Transformation Helps |
|-----------|---|---|
| **Linearity** | The relationship between predictor (x) and target (y) should be a straight line. | Transforming non-linear relationships can linearize them |
| **Normality of Errors** | Residuals should follow a normal distribution centered at zero. | Transforming the predictor distribution often normalizes residuals |
| **Homoscedasticity** | Error variance should be constant across all predictor levels. | Many transformations stabilize variance across ranges |

### 1.3 When to Transform

#### ✓ TRANSFORM FOR:
- Linear Regression
- Logistic Regression (sometimes)
- ANOVA and statistical tests
- GLM (Generalized Linear Models)

#### ✗ DON'T TRANSFORM FOR:
- **Tree-based models:** Decision Trees, Random Forests, Gradient Boosting (XGBoost, LightGBM)
  - These models are invariant to monotone transformations
  - Distribution shape is irrelevant
- **Distance-based models:** KNN needs scaling, not distribution transformation
- **Neural Networks:** Care about scale/normalization, not Gaussian distribution

### 1.4 The Golden Rule
**Always verify transformations visually.** Don't blindly transform:
- Uniform or bimodal distributions often won't become Gaussian
- If transformation doesn't improve, drop it
- Use histograms and Q-Q plots to validate

---

"""
restructured.append(why_when)

# Write to file
new_content = "\n".join(restructured)
file_path.write_text(new_content, encoding='utf-8')

# Count lines
new_lines = len(new_content.split('\n'))
print(f"✓ Testing restructure script")
print(f"✓ New file line count so far: {new_lines} lines")
