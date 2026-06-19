# AI Context: Credit Card Fraud Detection Project

## Project Overview
This project is an end-to-end Machine Learning pipeline and Streamlit dashboard for detecting fraudulent credit card transactions. The system is designed to handle imbalanced dataset scenarios using SMOTE and will provide an interactive web interface for both exploratory data analysis and live predictions.

## Strict Rules for AI Agents
1. **Humanized Code**: Write clean, structured, and readable code just as an experienced human Senior Data Scientist / ML Developer would.
2. **Humanized Comments**: Explain the "why" behind the code, not just the "what". Comments should be clear and concise.
3. **NO EMOJIS**: Do not use emojis anywhere in the codebase, comments, or conversations. Strictly forbidden.
4. **Jupyter Notebook Preference**: Use the `.ipynb` file (`fraud_detection_notebook.ipynb`) for data exploration, preprocessing, model training, and evaluation.
5. **Role**: Always act as a Senior Data Science and ML Developer.

## Technology Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Class Imbalance Handling**: Imbalanced-learn (SMOTE)
- **Deployment/Dashboard**: Streamlit
- **Serialization**: Joblib

## Core Files & Architecture
- **Dataset**: `credit_card_fraud_dataset.csv` (Located in the project root). This is highly imbalanced with fraud cases typically occurring at unusual hours, showing CVV mismatches, etc. It contains 15 features plus a target variable `is_fraud`.
- **Reference Document**: `fraud_detection_project_doc.docx` (Located at `D:\BRNB\`). This document contains the original project requirements.
- **`src/`**: Contains Python scripts for preprocessing, training, and evaluation (`preprocess.py`, `train.py`, `evaluate.py`).
- **`models/`**: To store serialized artifacts (`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).
- **`app/`**: Contains the Streamlit app script (`dashboard.py`).
- **`requirements.txt`**: Project dependencies.

## Development Phases
This project is being built phase-by-phase:

- **Phase 1: Project Setup** - Setup folder structure and basic files. *(Status: COMPLETED)*
- **Phase 2: Exploratory Data Analysis (EDA)** - Read the CSV and generate at least 6 meaningful plots with observations in markdown cells below each.
- **Phase 3: Data Preprocessing** - Handle categorical features (Label Encoding), split features and target variable, apply scaling (StandardScaler/MinMaxScaler), and save the scaler/encoders. (Never fit scaler on full dataset).
- **Phase 4: Handling Class Imbalance with SMOTE** - Apply SMOTE *only* on the training data after train/test split. Verify 50/50 class distribution.
- **Phase 5: Model Training & Evaluation** - Train Logistic Regression, Decision Tree, and Random Forest models on balanced data. Evaluate using Precision, Recall, F1-Score, and Accuracy on the original (unbalanced) test set. Create a comparison table.
- **Phase 6: Model Selection & Saving** - Prioritize Recall to catch fraud. Save the best model, scaler, and label encoders using joblib to the `models/` folder.
- **Phase 7: Streamlit Dashboard Development** - Build a multi-page interactive web dashboard (`Dataset Overview`, `Fraud Analysis`, `Model Performance`, `Live Fraud Prediction`). The prediction page must accept all 15 features dynamically.

## Current State
- Phase 1 (Environment Setup & Data Loading) is complete.
- Phase 2 (Exploratory Data Analysis) is complete in `fraud_detection_notebook.ipynb`. It contains 7 plots and observations saved in `plots/`.
- Phase 3 (Data Preprocessing) is complete. Code is written in `phase3_preprocessing.ipynb` and modularized in `src/preprocess.py`.
- Phase 4 (Handling Class Imbalance) is complete. SMOTE is applied strictly on the training partition to balance classes (50/50), written in `phase4_smote.ipynb` and modularized in `src/preprocess.py`.
- Phase 5 (Model Training & Evaluation) is complete. Logistic Regression, Decision Tree, and Random Forest models are trained on balanced data and evaluated on the original imbalanced test set, written in `phase5_training_evaluation.ipynb` and modularized in `src/train.py` and `src/evaluate.py`.
- Phase 6 (Model Selection & Saving) is complete. The optimal classifier has been selected prioritizing Recall and F1-Score, and saved to `models/` along with the scaler and encoders, written in `phase6_model_saving.ipynb` and modularized in `src/evaluate.py`.
- Phase 7 (Streamlit Dashboard Development) is complete. The application is located in `app/dashboard.py` and includes dataset statistics, interactive plots, model evaluation comparisons, and a dynamic 15-feature inference page. All serialized artifacts load successfully.
