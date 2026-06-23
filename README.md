# 💳 Credit Card Fraud Detection System

## 🎯 What is This?

Welcome! This is a complete **Machine Learning system** that automatically detects fraudulent credit card transactions. It combines powerful data science with an easy-to-use interactive dashboard where you can:

- 📊 Explore fraud patterns in your data
- 🔍 Check if a single transaction is fraud or genuine
- 📤 Upload batch files and get predictions on thousands of transactions at once
- 📈 View detailed model performance metrics and charts

The system is trained on real credit card transaction data and uses a **Random Forest model** to make predictions with high accuracy.

---

## 🎨 Dashboard Overview (5 Interactive Pages)

### 📋 **Overview Page**
Get a quick snapshot of your data:
- Total transactions processed
- Count of legitimate vs fraudulent transactions
- Average transaction amount
- Visual breakdown of fraud rates
- Sample data table to see what's in the dataset

### 🔎 **Fraud Patterns Page**
Deep dive into where and how fraud happens:
- **When do frauds occur?** — See fraud trends by hour of day with transaction amounts
- **Which merchants are hit most?** — Fraud rate breakdown by merchant category
- **What are the warning signs?** — Account age, CVV mismatches, international transactions, card presence, and location mismatches

### 🤖 **Model Results Page**
Compare our three trained models side-by-side:
- **Logistic Regression** — Simple and interpretable
- **Decision Tree** — Easy to understand decision paths
- **Random Forest** — Our best performer (selected for the dashboard)

Each model shows its confusion matrix with clear labels and overall performance metrics.

### ✅ **Check a Transaction Page**
Test the fraud detector with a single transaction:
1. Fill in 15 transaction details (amount, hour, merchant, etc.)
2. Click "Run Prediction" to see if it's fraud or genuine
3. Try "Simulate Live Transaction" to see a realistic batch of transactions being processed live

### 📤 **Batch Prediction Page**
Upload and process multiple transactions at once:
- **Upload formats**: CSV, JSON, or Excel (.xlsx) files
- **Instant validation**: We check that your data is correct before processing
- **Results table**: See each transaction with its prediction (Fraud ✗ or Genuine ✓)
- **Summary cards**: Total transactions, fraud count, and fraud rate
- **Download results**: Export predictions as CSV, JSON, or Excel
- **Visual charts**: Fraud distribution and merchant category breakdown

---

## 🚀 Getting Started (Complete Setup Guide)

### Prerequisites
- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package installer)
- **Git** (optional, for cloning the repo)

### Step 1️⃣: Clone or Download the Project
```bash
# If using git
git clone <your-repo-url>
cd fraud_detection_project

# Or simply download and extract the folder
cd fraud_detection_project
```

### Step 2️⃣: Create a Virtual Environment (Recommended)
It's best practice to use a virtual environment to avoid conflicts:

**On Windows (PowerShell or CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line when activated.

### Step 3️⃣: Install Dependencies
Once your virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

This installs:
- 🐼 **Pandas & NumPy** — Data manipulation
- 📊 **Plotly, Matplotlib, Seaborn** — Visualization
- 🤖 **Scikit-learn** — Machine learning models
- ⚖️ **Imbalanced-learn** — SMOTE for handling imbalanced data
- 💾 **Joblib** — Model serialization
- 🌐 **Streamlit** — Interactive dashboard

### Step 4️⃣: Train the Model (First Time Only)
The first time you use this project, generate the trained model:

```bash
python src/evaluate.py
```

This will:
- Load and preprocess the data
- Train three models (Logistic Regression, Decision Tree, Random Forest)
- Save the best model and supporting files in the `models/` folder
- Generate charts and save them in the `plots/` folder

**Note:** This step is only needed once. The trained model is then reused every time you launch the dashboard.

### Step 5️⃣: Launch the Dashboard 🎉
Start the interactive web dashboard:

```bash
streamlit run app/dashboard.py
```

Your browser will automatically open to `http://localhost:8501` where you can explore all 5 pages!

---

## 📁 Project Structure

```
fraud_detection_project/
│
├── 📄 README.md                          ← You are here!
├── 📄 requirements.txt                    ← All dependencies
├── 📊 credit_card_fraud_dataset.csv       ← Raw training data
│
├── 📓 Jupyter Notebooks (Data Pipeline):
│   ├── exploratory_data_analysis.ipynb           (Phase 2: Explore the data)
│   ├── data_preprocessing.ipynb                  (Phase 3: Clean & encode)
│   ├── handling_class_imbalance.ipynb            (Phase 4: Balance classes)
│   ├── model_training_and_evaluation.ipynb       (Phase 5: Train models)
│   └── model_selection_and_saving.ipynb          (Phase 6: Pick the best)
│
├── 🐍 src/ (Modular Python Scripts):
│   ├── preprocess.py      ← Encoding, scaling, SMOTE pipeline
│   ├── train.py           ← Train classifiers
│   └── evaluate.py        ← Evaluate and save the best model
│
├── 🎯 models/ (Trained Artifacts):
│   ├── best_model.pkl          ← Random Forest classifier (ready to use)
│   ├── scaler.pkl              ← Data normalizer
│   └── label_encoder.pkl       ← Encoding mappings
│
├── 💾 app/ (Streamlit Dashboard):
│   └── dashboard.py       ← The interactive web interface (5 pages)
│
└── 📈 plots/ (Generated Charts):
    └── *.png files        ← Visualizations from analysis
```

---

## 🧠 How the Model Works

**Algorithm:** Random Forest Classifier
- Ensemble of 100+ decision trees voting on predictions
- Very accurate and handles complex patterns well

**Training Data:** 
- Used 70% of transactions, balanced with SMOTE to have 50% fraud
- SMOTE creates synthetic fraud examples so the model learns both classes equally

**Test Data:**
- Used remaining 30% of transactions in their original imbalanced state
- This gives realistic fraud detection performance metrics

**Features (15 inputs):**
- Transaction amount and time
- Merchant category
- Distance from customer's home
- Card present status
- CVV and location mismatches
- Customer and account age
- Recent transaction history

**Evaluation Metric:** Recall (sensitivity)
- We prioritize catching fraud (even if some legitimate transactions get flagged)
- Better to verify a few good transactions than miss fraud

---

## 📦 Dependencies & Tech Stack

| Layer | What | Why |
|-------|------|-----|
| **Data** | Pandas, NumPy | Fast data manipulation and analysis |
| **Visualization** | Plotly, Matplotlib, Seaborn | Create interactive and static charts |
| **Machine Learning** | Scikit-learn | Robust ML models and preprocessing |
| **Class Balancing** | Imbalanced-learn (SMOTE) | Handle fraud being rare in real data |
| **Serialization** | Joblib | Save and load trained models |
| **Dashboard** | Streamlit | Beautiful web interface with minimal code |

---

## 🔧 Troubleshooting

**Q: I see an error when running `streamlit run app/dashboard.py`**
- Make sure your virtual environment is activated (you see `(venv)` in the terminal)
- Run `pip install -r requirements.txt` again to ensure all packages are installed
- Check that you're in the correct project folder

**Q: The model predictions seem off**
- This is normal! ML models make mistakes sometimes
- Check the "Model results" page to see performance metrics
- The model was trained on real fraud patterns but may not catch 100% of cases

**Q: I want to retrain the model with new data**
- Replace `credit_card_fraud_dataset.csv` with your new data
- Keep the same 15 columns (exact names and format)
- Run `python src/evaluate.py` again

**Q: Can I upload a file with different columns?**
- The batch prediction page requires exactly these 15 columns (see the format guide)
- You can download a template CSV to see the exact format needed

---

## ⭐ Key Features

✅ **5 Interactive Dashboard Pages** — Full end-to-end analysis and prediction interface  
✅ **Batch Predictions** — Upload CSV, JSON, or Excel and get instant fraud scores  
✅ **Single Transaction Checking** — Test individual transactions manually  
✅ **Live Simulation** — See realistic fraud detection in action  
✅ **Multiple Models** — Compare Logistic Regression, Decision Tree, and Random Forest  
✅ **Dark/Light Mode** — Toggle between themes for comfortable viewing  
✅ **Smart Validation** — File format checks before processing  
✅ **Export Results** — Download predictions in your preferred format  
✅ **Data Visualization** — Interactive charts and confusion matrices  
✅ **Well-Documented** — Code comments and clear documentation throughout

---

## 🎓 Learning the Code

This project is great for learning:
- **Data Science Workflow** — From raw data to production model
- **Machine Learning** — Training, evaluation, and model selection
- **Data Preprocessing** — Encoding, scaling, handling imbalanced data
- **Streamlit** — Building interactive web dashboards in Python
- **Python Best Practices** — Clean code, modular functions, virtual environments

---

## 📞 Need Help?

- Check the **Troubleshooting** section above
- Review the Jupyter notebooks to understand each phase
- Look at `src/` scripts for the actual processing logic
- The dashboard code in `app/dashboard.py` is heavily commented

---

## 📝 License & Attribution

This is an educational project for fraud detection using machine learning. Feel free to use it for learning, but be aware that model predictions should always be validated by human review in production environments.

---

## 🚀 Next Steps

Once you have the dashboard running:

1. **Explore the data** — Visit the Overview and Fraud Patterns pages
2. **Check model performance** — See the Model Results page
3. **Test single predictions** — Try the Check a Transaction page with sample data
4. **Process batch files** — Upload the included test files to see predictions
5. **Customize the model** — Replace the dataset and retrain for your own data

---

**Made with ❤️ using Python, Machine Learning, and Streamlit**
