# Credit Card Fraud Detection System

## Overview
This project is an end-to-end Machine Learning pipeline and Streamlit dashboard for detecting fraudulent credit card transactions. It contains exploratory analysis, preprocessing pipelines, class balancing using SMOTE, and multiple classification algorithms.

## Project Structure
- `credit_card_fraud_dataset.csv`: Raw transactional dataset.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation and setup guidelines.
- `AI_CONTEXT.md`: System development rules and current progress tracking.

### Phase Notebooks (.ipynb)
- `fraud_detection_notebook.ipynb`: Phase 1 and Phase 2 (Exploratory Data Analysis with 7 detailed plots).
- `phase3_preprocessing.ipynb`: Phase 3 (Data cleaning, encoding categorical features, train/test split, and StandardScaler).
- `phase4_smote.ipynb`: Phase 4 (Class balancing using SMOTE applied strictly to the training partition).
- `phase5_training_evaluation.ipynb`: Phase 5 (Model training and evaluation comparing Logistic Regression, Decision Tree, and Random Forest).
- `phase6_model_saving.ipynb`: Phase 6 (Performance comparison, optimal model selection, and artifact serialization).

### Modular Python Scripts (src/)
- `src/preprocess.py`: Modular data cleaning, label encoding, scaling, and oversampling pipeline.
- `src/train.py`: Model training logic for classifier candidates.
- `src/evaluate.py`: Performance metrics reporting, confusion matrix plotting, and selection of the best model.

### Saved Model Artifacts (models/)
- `models/best_model.pkl`: Serialized best-performing classification model.
- `models/scaler.pkl`: Fitted StandardScaler.
- `models/label_encoder.pkl`: Label encoder and day-of-week mappings.

### Dashboard Interface (app/)
- `app/dashboard.py`: Interactive Streamlit dashboard showing statistics, exploratory charts, model performance metrics, and a real-time transaction prediction interface.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Train models and generate serialized artifacts:
   ```bash
   python src/evaluate.py
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```
