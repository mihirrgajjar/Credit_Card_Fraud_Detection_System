import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_data(filepath):
    """Load dataset from path."""
    return pd.read_csv(filepath)

def preprocess_categorical(df):
    """
    Encode categorical columns.
    - day_of_week: custom ordered mapping (Mon -> 0 ... Sun -> 6)
    - merchant_category: scikit-learn LabelEncoder
    """
    df = df.copy()
    
    # Custom mapping for days of week
    day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    df['day_of_week'] = df['day_of_week'].map(day_mapping)
    
    # Label encoding for merchant_category
    le_merchant = LabelEncoder()
    df['merchant_category'] = le_merchant.fit_transform(df['merchant_category'])
    
    return df, le_merchant, day_mapping

def prepare_features(df, target_col='is_fraud', drop_cols=None):
    """Drop identifier columns and separate features from target."""
    if drop_cols is None:
        drop_cols = ['transaction_id']
        
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    return X, y

def split_dataset(X, y, test_size=0.2, random_state=42):
    """Split features and target into stratified train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train, X_test):
    """Fit a standard scaler on train set and transform both train and test partitions."""
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler

def balance_training_data(X_train, y_train, random_state=42):
    """Apply SMOTE on training set only to handle target class imbalance."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def save_preprocessing_artifacts(scaler, le_merchant, day_mapping, output_dir='models'):
    """Serialize the scaling and encoding objects for dashboard prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save encoders
    encoders = {
        'merchant_category_encoder': le_merchant,
        'day_of_week_mapping': day_mapping
    }
    joblib.dump(encoders, os.path.join(output_dir, 'label_encoder.pkl'))

def run_pipeline(data_path, models_dir='models'):
    """Run full preprocessing workflow and return processed training and testing partitions."""
    print("Loading data...")
    df = load_data(data_path)
    
    print("Encoding categorical features...")
    df_encoded, le_merchant, day_mapping = preprocess_categorical(df)
    
    print("Splitting features and target...")
    X, y = prepare_features(df_encoded)
    
    print("Splitting into train/test partitions...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    print("Standardizing numerical features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("Balancing training data with SMOTE...")
    X_train_res, y_train_res = balance_training_data(X_train_scaled, y_train)
    
    print("Saving scaling and encoding objects...")
    save_preprocessing_artifacts(scaler, le_merchant, day_mapping, models_dir)
    
    print("Preprocessing completed successfully.")
    return X_train_res, X_test_scaled, y_train_res, y_test

if __name__ == '__main__':
    run_pipeline('credit_card_fraud_dataset.csv')
