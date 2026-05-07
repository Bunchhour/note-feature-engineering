# Engineering Datetime Variables: Complete Learning Guide

> **TL;DR Quick Decision:** Need to extract features from Datetime?
> - **Ad-hoc/Exploratory:** Pandas (\.dt\ accessor)
> - **Production Pipeline:** Feature-engine (\DateTimeFeatures\)
> - **Global/Mixed Timezones:** Always standardize to UTC first!

## Quick Decision Guide

| Your Data / Goal | Best Method | Why | Library |
|-----------|------|---|---|
| Single column, custom extraction | Pandas \.dt\ | High flexibility, built-in | pandas |
| Multiple columns, automated | \DateTimeFeatures\ | Streamlined, avoids code repetition | Feature-engine |
| Mixed timezones (e.g. +02:00, -05:00) | UTC Standardization | Prevents skew from offsets | pandas / Feature-engine |
| High-frequency logs (trades, servers) | Micro/Nanoseconds | Captures sub-second precision | pandas |

---

## Part 1: Why Engineer Datetime Variables?

When preparing data for machine learning, raw timestamps (which can contain dates, times, or both) are rarely used directly. Instead, we extract structured features from them to help predictive models find seasonal, weekly, or daily patterns.

Prerequisite: **Data Casting**. Before extracting features, your variable **must** be cast as a datetime object.
- Excel files often cast automatically.
- CSVs load as strings (\object\). Convert first: \pd.to_datetime(df['dateColumn'])\.

---

## Part 2: Extracting Components

### 2.1 Date-Derived Features
From the date portion of a timestamp, you can pull a wide variety of granular data points:
* **Year/Month/Day:** Basic temporal features.
* **Quarter/Semester:** Useful for financial or academic data.
* **Day of Week:** Numeric (0-6) or string name.
* **Edge Cases & Special Periods:** First/last day of the year/month/quarter, leap years, week of the year.

### 2.2 Time-Derived Features
From the time portion, extract sequential components:
* Hours, minutes, and seconds.
* Microseconds and nanoseconds (if dataset requires high precision).

### 2.3 Managing Time Zones
Time zones can skew your data if not handled correctly.
* **Standardization:** If your data contains mixed time zones, standardize them before extracting features.
* **Best Practice:** Convert all timestamps to **UTC** (Universal Time Coordinated) to normalize the data. You can then shift it to a specific local time zone if needed.

---

## Part 3: Implementation by Library

### 3.1 Pandas: The \.dt\ Accessor (Cheat Sheet)

\\python
import pandas as pd
import numpy as np

# 0. Casting to datetime (handling UTC automatically)
df['date_col'] = pd.to_datetime(df['date_col'], utc=True)

# 1. Year Features
df['year'] = df['date_col'].dt.year
df['is_leap'] = df['date_col'].dt.is_leap_year
df['is_year_start'] = df['date_col'].dt.is_year_start

# 2. Quarter & Semester Features
df['quarter'] = df['date_col'].dt.quarter
# Semester (Custom derivation)
df['semester'] = np.where(df['date_col'].dt.quarter <= 2, 1, 2)

# 3. Month & Week Features
df['month'] = df['date_col'].dt.month
df['week_of_year'] = df['date_col'].dt.isocalendar().week

# 4. Day Features
df['day'] = df['date_col'].dt.day
df['day_of_week'] = df['date_col'].dt.dayofweek # 0=Monday, 6=Sunday
df['is_weekend'] = np.where(df['date_col'].dt.dayofweek > 4, 1, 0)

# 5. Time Features
df['hour'] = df['date_col'].dt.hour
df['minute'] = df['date_col'].dt.minute
df['second'] = df['date_col'].dt.second
\
### 3.2 Feature-engine: Automated Pipeline

Feature-engine's \DateTimeFeatures\ automates the extraction across multiple columns, neatly naming the new columns (e.g. \InvoiceDate_month\).

\\python
from feature_engine.datetime import DatetimeFeatures

# Initialize the transformer
dt_transformer = DatetimeFeatures(
    variables=None, # None automatically finds all datetime columns
    features_to_extract='all', # Or specific ones: ['month', 'year', 'day_of_week']
    drop_original=True, # Drop the original datetime column
    utc=True # Handle mixed timezones by standardizing to UTC first
)

# Fit and Transform within a scikit-learn standard pipeline
dt_transformer.fit(X_train)

X_train_transformed = dt_transformer.transform(X_train)
X_test_transformed = dt_transformer.transform(X_test)
\