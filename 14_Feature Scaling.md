# Feature Scaling: Complete Learning Guide

> **TL;DR Quick Decision:** Need to scale variables?
> - **Default/Assumption of Normality:** Standardization (Z-score)
> - **Strict Boundaries (0 to 1):** Min-Max Scaling
> - **Preserve Sparsity:** MaxAbsScaling
> - **Extreme Outliers:** Robust Scaling (Median/IQR)

## Quick Decision Guide

| Scaling Goal | Best Method | Math | Scikit-Learn Class |
|--------------|-------------|------|--------------------|
| Default/Center at 0 | Standardization |  = \frac{X - \mu}{\sigma}$ | \StandardScaler\ |
| Compress between 0 and 1 | Min-Max Scaling | $\frac{X - X_{min}}{X_{max} - X_{min}}$ | \MinMaxScaler\ |
| Handle Extreme Outliers | Robust Scaling | $\frac{X - Median}{IQR}$ | \RobustScaler\ |
| Sparse Data / No center | MaxAbsolute | $\frac{X}{|X_{max}|}$ | \MaxAbsScaler\ |
| Center and Bound | Mean Normalization | $\frac{X - \mu}{X_{max} - X_{min}}$ | Custom (Standard + Robust) |

---

## Part 1: Why Feature Magnitude Matters

Feature scaling ensures the range of values across independent variables is normalized to a similar scale. 
* **Coefficient Domination:** Larger magnitudes can disproportionately skew linear regression coefficients.
* **Faster Convergence:** Gradient descent algorithms converge much faster.
* **Distance Calculations:** Crucial for distance-based ML models.

### Algorithm Sensitivity
* **Requires Scaling:** Linear & Logistic Regression, Neural Networks, SVM, KNN, K-Means, PCA, LDA.
* **No Scaling Needed:** Tree-based models (CART, Random Forests, Gradient Boosted Trees).

**Golden Rule:** Scaling changes the **range** but preserves the **shape** (distribution, skewness) of the data. To change the shape, you must use a Variable Transformation (e.g., Log, Square Root).

---

## Part 2: Feature Scaling Techniques

### 2.1 Standardization (Z-score Normalization)
Centers data around a mean ($\mu$) of zero with a standard deviation ($\sigma$) of one. Output is a Z-score.

### 2.2 Min-Max Scaling
Compresses data tightly between a specific range, usually 0 and 1. Does not center the mean at 0.

### 2.3 Mean Normalization
Centers data around zero but bounds it using the absolute range ({max} - X_{min}$). Scikit-learn has no native transformer for this.

### 2.4 Maximum Absolute Scaling
Finds the largest absolute number and divides everything by that value. Bounded between -1 and 1. Excellent for **sparse data** because it doesn't subtract a mean (which would destroy zeros).

### 2.5 Robust Scaling
Uses statistics resistant to outliers: the **Median** and the **Interquartile Range (IQR)**. Ignores extreme values on the edges.

---

## Part 3: Scaling Categorical Variables (Special Note)

Most categorical variables mapped to numbers don't require scaling if they are already between 0 and 1:
* **One-Hot Encoding & Target Encoding (Classification):** Scales between 0 and 1. No further scaling needed.
* **Weight of Evidence:** Puts variables on a logit scale.
* **Ordinal Encoding:** Replaces with integers (e.g. 1, 2, 3), **should be scaled** before training models like SVM/Linear.

*Note:* Standardizing a binary dummy variable (0 and 1) can arbitrarily distort the distance between the two categories depending on prevalence.

---

## Part 4: Implementation by Library (Scikit-Learn)

**Golden Rule:** ALWAYS split your data into Train/Test first. Fit the scaler ONLY on \X_train\ to prevent data leakage!

### 4.1 Standardization & Min-Max Scaling

\\python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Select Scaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

# 3. Fit ONLY on train
scaler.fit(X_train)

# 4. Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
\
> **Checking Learned Attributes:**
> - StandardScaler: \scaler.mean_\, \scaler.scale_\ (std dev)
> - MinMaxScaler: \scaler.data_min_\, \scaler.data_max_\, \scaler.data_range_
### 4.2 Mean Normalization (Workaround)

\\python
# Scikit-learn has no MeanNormalizer right out of the box. We chain two!
from sklearn.preprocessing import StandardScaler, RobustScaler

# Step 1: Subtract mean only
mean_scaler = StandardScaler(with_mean=True, with_std=False)

# Step 2: Divide by absolute Min/Max range
range_scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(0, 100))

mean_scaler.fit(X_train)
range_scaler.fit(X_train)

X_train_final = range_scaler.transform(mean_scaler.transform(X_train))
X_test_final = range_scaler.transform(mean_scaler.transform(X_test))
\
### 4.3 Maximum Absolute Scaling

\\python
from sklearn.preprocessing import MaxAbsScaler

# Standard MaxAbs (Great for Sparse Data)
scaler = MaxAbsScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Attribute: scaler.max_abs_
\
### 4.4 Robust Scaling

\\python
from sklearn.preprocessing import RobustScaler

# Robust to extreme outliers
scaler = RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Attributes: scaler.center_ (median), scaler.scale_ (IQR)
\