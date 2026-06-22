# Credit Card Fraud Detection System

## Overview
End-to-end Machine Learning pipeline and interactive Streamlit dashboard for detecting fraudulent credit card transactions. The system covers exploratory analysis, data preprocessing, class balancing with SMOTE, model training across three classifiers, and a fully featured web dashboard with real-time and batch prediction capabilities.

---

## Dashboard Pages

### Overview
Summary of the dataset — total transactions, legitimate vs flagged counts, average spend, class balance donut chart, data snapshot table, and transaction distribution by merchant category.

### Fraud patterns
Three-tab analysis page exploring where and how fraud occurs:
- **Amount & hour** — fraud frequency by hour of day with observation text, transaction amount box plot by class
- **Merchant insights** — fraud rate per merchant category
- **Risk indicators** — account age distribution and risk flag proportions (CVV mismatch, international transaction, card present, location match)

### Model results
Side-by-side performance cards for Logistic Regression, Decision Tree, and Random Forest. Includes confusion matrices with plain-English cell labels (Correct: genuine, Correct: fraud, Missed: fraud not caught, Wrong: flagged genuine) and a summary card for the selected model.

### Check a transaction
Manual single-transaction prediction form with 15 input fields across two columns. Includes:
- **Run prediction** — solid blue primary button, runs the model and shows a result box with confidence bar
- **Simulate Live Transaction** — opens a dialog modal that pulls a random batch of 8–15 transactions with a realistic multi-step loading sequence (skeleton rows + progress bar), then displays a results table with colored fraud/genuine badges and a 3-card summary

### Batch prediction
Upload a CSV file to run predictions on every row at once. Features:
- **Format guide** — always-visible amber card showing all 15 required columns with types and accepted values, plus a collapsible sample row
- **Validation** — column presence check, missing value detection, value range checks for binary columns and transaction hour, accepted set checks for day_of_week and merchant_category. Shows specific column-level error messages on failure
- **Results** — 4 summary metric cards, paginated predictions table (50 rows/page) with fraud rows highlighted in red, colored result badges, formatted amounts with ₹ symbol, Yes/No for binary columns
- **Download** — exports full CSV with `Predicted_Fraud` and `Fraud_Probability_%` columns appended
- **Charts** — fraud vs genuine donut and fraud count by merchant category bar chart

---

## Project Structure

```
credit_card_fraud_dataset.csv   Raw transaction dataset
requirements.txt                Python dependencies
README.md                       This file
AI_CONTEXT.md                   Development rules and phase tracking
simulate_live_transaction_feature.md   Feature spec for the simulate dialog
upload_and_predict_feature.md          Feature spec for batch prediction page
ui_humanization_guide.md               UI copy and polish guidelines
```

### Phase Notebooks
| File | Phase | Description |
|------|-------|-------------|
| `exploratory_data_analysis.ipynb` | Phase 2 | EDA with 7 plots and observations |
| `data_preprocessing.ipynb` | Phase 3 | Encoding, train/test split, StandardScaler |
| `handling_class_imbalance.ipynb` | Phase 4 | SMOTE applied only to the training partition |
| `model_training_and_evaluation.ipynb` | Phase 5 | Logistic Regression, Decision Tree, Random Forest training and evaluation |
| `model_selection_and_saving.ipynb` | Phase 6 | Model selection by Recall, artifact serialization |

### Modular Scripts (`src/`)
| File | Purpose |
|------|---------|
| `src/preprocess.py` | Encoding, scaling, SMOTE pipeline |
| `src/train.py` | Classifier training logic |
| `src/evaluate.py` | Metrics, confusion matrix, best model selection |

### Model Artifacts (`models/`)
| File | Contents |
|------|---------|
| `models/best_model.pkl` | Trained Random Forest classifier |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/label_encoder.pkl` | LabelEncoder and day-of-week mapping |

### Dashboard (`app/`)
| File | Purpose |
|------|---------|
| `app/dashboard.py` | Full Streamlit dashboard — all 5 pages, simulation dialog, batch prediction, dark/light mode toggle |

### Plots (`plots/`)
Generated during EDA — class distribution, transaction amounts, hour patterns, merchant category breakdown, correlation heatmap, feature comparisons, account age distribution, and model confusion matrices.

---

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training pipeline to generate model artifacts:
   ```bash
   python src/evaluate.py
   ```

3. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

---

## Model

- **Algorithm**: Random Forest Classifier
- **Training data**: SMOTE-balanced training split (50/50 class ratio)
- **Evaluation data**: Original imbalanced test set — never resampled
- **Selection criterion**: Highest Recall to minimize missed fraud cases
- **Features**: 15 input features including transaction amount, hour, merchant category, distance from home, CVV mismatch, card presence, and account age

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| ML | Scikit-learn |
| Class balancing | Imbalanced-learn (SMOTE) |
| Serialization | Joblib |
| Dashboard | Streamlit |
