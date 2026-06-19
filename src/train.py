from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def init_models(random_state=42):
    """Initialize classifier candidates."""
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state)
    }

def train_models(X_train, y_train, random_state=42):
    """
    Train model candidates on the resampled training partition.
    Returns a dictionary of trained classifiers.
    """
    classifiers = init_models(random_state)
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
    print("All models trained successfully.")
    return trained_models
