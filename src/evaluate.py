import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Import other steps from our package
from preprocess import run_pipeline
from train import train_models

def evaluate_models(trained_models, X_test, y_test, plots_dir='plots'):
    """
    Evaluate trained models on the test partition.
    Prints classification reports, saves heatmap plots of confusion matrices,
    and returns a comparison DataFrame of accuracy, precision, recall, and f1.
    """
    os.makedirs(plots_dir, exist_ok=True)
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, model) in enumerate(trained_models.items()):
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }
        
        print(f"\n=== {name} Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, 'model_confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    print(f"\nConfusion matrices saved to {cm_path}")
    plt.close()
    
    df_results = pd.DataFrame(results).T
    return df_results

def select_and_save_best_model(trained_models, X_test, y_test, output_dir='models'):
    """
    Select the best performing model.
    Prioritizes Recall first (minimizing missed fraud), followed by F1-Score.
    Saves the selected classifier to models/best_model.pkl.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    best_recall = -1
    best_model_name = None
    best_model = None
    best_metrics = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Prioritize recall, tiebreak with F1
        if rec > best_recall:
            best_recall = rec
            best_model_name = name
            best_model = model
            best_metrics = {'Recall': rec, 'F1-Score': f1}
        elif rec == best_recall:
            if best_model_name is not None and f1 > best_metrics['F1-Score']:
                best_model_name = name
                best_model = model
                best_metrics = {'Recall': rec, 'F1-Score': f1}
                
    print(f"\nSelected Best Model: {best_model_name}")
    print(f"Recall: {best_metrics['Recall']:.4f} | F1-Score: {best_metrics['F1-Score']:.4f}")
    
    model_path = os.path.join(output_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    return best_model_name, best_metrics

def main():
    # 1. Run preprocessing pipeline
    X_train, X_test, y_train, y_test = run_pipeline('credit_card_fraud_dataset.csv')
    
    # 2. Train models
    print("\nTraining models...")
    trained_models = train_models(X_train, y_train)
    
    # 3. Evaluate models
    print("\nEvaluating models...")
    df_results = evaluate_models(trained_models, X_test, y_test)
    print("\n=== Model Performance Comparison ===")
    print(df_results.round(4))
    
    # 4. Select and save best model
    select_and_save_best_model(trained_models, X_test, y_test)

if __name__ == '__main__':
    main()
