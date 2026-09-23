# DSN Mart Sales Forecasting — DSN AI Bootcamp 2026 Qualifier

An end-to-end Machine Learning project to forecast product-level retail sales across diverse store formats and locations in Nigeria for the **Data Science Nigeria (DSN) AI Bootcamp 2026 Qualifier Hackathon**.

---

## Problem Overview

**DSN Mart** operates an omnichannel chain of retail outlets across Nigeria, ranging from local corner shops to flagship hypermarkets across major urban centers, state capitals, and smaller towns. 

The goal of this project is to build an accurate and robust regression model that predicts **total sales (`total_sales`)** for a given product at a given store format and location, enabling the retail network to intelligently plan inventory, pricing strategies, and store investments.

* **Target Variable:** `total_sales`
* **Evaluation Metric:** **Root Mean Squared Error (RMSE)**

---

## Dataset Description

| Feature | Type | Description |
|---|---|---|
| `id` | Identifier | Unique row identifier |
| `product_code` | Categorical | Unique alphanumeric product SKU |
| `product_weight_kg` | Numeric | Physical weight of the product in kilograms |
| `fat_content` | Categorical | Nutritional classification (`Low Fat`, `Regular`) |
| `shelf_visibility` | Numeric | Percentage of total display area allocated to the product |
| `product_category` | Categorical | Product category (e.g. `fruits and vegetables`, `household`) |
| `product_price` | Numeric | Listed unit selling price |
| `store_code` | Categorical | Unique store identifier (10 distinct outlets) |
| `store_age_years` | Numeric | Operating age of the store outlet |
| `store_size` | Categorical (Ordinal) | Physical store size footprint (`Small`, `Medium`, `Large`) |
| `store_location_tier`| Categorical (Ordinal) | City / geographic tier (`Tier_1`, `Tier_2`, `Tier_3`) |
| `store_format` | Categorical | Store layout (`Corner Shop`, `Supermarket`, `Flagship Hypermarket`) |

---

## Methodology & Architecture

### 1. Data Cleaning & Hierarchical Imputation
* **`product_weight_kg`:** Imputed using a tiered hierarchy:
  1. Product-specific median (`product_code`)
  2. Category-level median (`product_category`)
  3. Global median fallback
* **`store_size`:** Missing values resolved by mapping the mode of `(store_format, store_location_tier)` combinations, falling back to `store_format` mode, then global mode.
* **`shelf_visibility` Zero-Correction:** Values equal to `0.0` represent tracking anomalies (an item cannot be sold if invisible). These were replaced with category-specific median visibility.

### 2. Feature Engineering
* **`visibility_ratio`:** Ratio of an item's store visibility to its mean visibility across the entire store chain, identifying prioritized display placement.
* **`price_per_weight`:** Value density indicator (`product_price / product_weight_kg`).
* **Multi-Feature Interaction (`store_format × product_price`)**: Captures the fundamental retail economic identity $\text{total\_sales} \approx \text{units} \times \text{price}$, where store formats scale item demand linearly with price.

### 3. Model Architecture & Ensembling
We implement a dual-model ensemble that combines two complementary mathematical paradigms:
1. **Regularized Linear Regression (ElasticNet)**: With $L_1$ and $L_2$ regularization, this model perfectly fits the global linear scaling of store format interactions.
2. **Gradient Boosted Decision Trees (CatBoost)**: Leveraging native target statistics on high-cardinality nominals (`product_code`, `store_code`, `product_category`) without one-hot explosion or target leakage.

```
                  ┌──────────────────────┐
                  │ Preprocessed Features│
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ ElasticNet Regression│      │  CatBoost Regressor  │
   │  (Linear Physics)    │      │ (Non-linear & Items) │
   └──────────┬───────────┘      └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                Weighted Blend (0.35 / 0.65)
                             │
                             ▼
                    Final Sales Forecast
```

### 4. Empirical Benchmark Performance

| Model | Evaluation Metric (Holdout RMSE) | Public Leaderboard RMSE |
|---|---|---|
| **Mean Baseline** | 1,716.87 | — |
| **LightGBM Regressor** | 1,102.39 | — |
| **Ridge Regression (Interactions)** | 1,073.42 | — |
| **ElasticNet Regression (Interactions)** | 1,072.75 | — |
| **CatBoost Regressor (Native Categoricals)** | 1,071.93 | — |
| **Dual Ensemble (CatBoost + ElasticNet)** | **1,070.39** | **1,072.15993 (Rank 47)** |

---

## Repository Structure

```text
├── DSN_Bootcamp_project_train.ipynb         # Main end-to-end training & analysis notebook
├── DSN_Bootcamp_project_test.ipynb          # Test inference pipeline
├── train.csv                                # Training dataset (6,818 rows)
├── test.csv                                 # Test dataset (1,705 rows)
├── submission.csv                           # Current top-performing submission file
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## Setup & Execution

### 1. Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/ajibua/DSN-Bootcamp-2026-Qualifier.git
cd DSN-Bootcamp-2026-Qualifier
pip install -r requirements.txt
```

### 2. Running the Pipeline
Open and execute [DSN_Bootcamp_project_train.ipynb](file:///c:/Users/HomePC/Documents/Data%20Science/DSN/DSN_Bootcamp_project_train.ipynb) in Jupyter / VS Code from top to bottom. It will perform all data cleaning, EDA, feature encoding, cross-validation, and generate `submission.csv`.

---

## Tech Stack
* **Language:** Python 3.14+
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `catboost`, `lightgbm`
