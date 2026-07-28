import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def main():
    print("=" * 60)
    print("      TELECOM CUSTOMER CHURN PREDICTION PIPELINE       ")
    print("=" * 60)

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    if not os.path.exists(data_path):
        data_path = 'data.csv'
    
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Data Preprocessing
    print("\n[2] Data Preprocessing & Cleaning...")
    
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # TotalCharges is object type with spaces for missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    if missing_count > 0:
        print(f"  - Found {missing_count} missing values in 'TotalCharges'. Imputing with median.")
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode Churn (Target)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"  - Target distribution (Churn): {df['Churn'].value_counts(normalize=True).to_dict()}")

    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # One-hot encoding for categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"  - Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"  - Numerical columns ({len(numerical_cols)}): {numerical_cols}")

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"  - Features shape after One-Hot Encoding: {X_encoded.shape}")

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[3] Train-Test Split completed (80/20):")
    print(f"  - Train shape: {X_train.shape}")
    print(f"  - Test shape:  {X_test.shape}")

    # 4. Model Training & Evaluation
    print("\n[4] Training & Evaluating Classifiers...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('gbc', models['Gradient Boosting']),
            ('lr', models['Logistic Regression']),
            ('abc', models['AdaBoost'])
        ],
        voting='soft'
    )
    models['Voting Classifier'] = voting_clf

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # K-Fold CV Score
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        
        # Fit on train data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        results.append({
            'Model': name,
            'CV Accuracy (Mean)': cv_scores.mean(),
            'CV Accuracy (Std)': cv_scores.std(),
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        print(f"  * {name:<22} | CV Acc: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) | Test Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    results_df = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    
    print("\n" + "=" * 85)
    print("                              MODEL EVALUATION SUMMARY                              ")
    print("=" * 85)
    print(results_df.to_string(index=False))

    # 5. Best Model Metrics & Classification Report
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n[5] Best Performing Model: {best_model_name}")
    best_model = models[best_model_name]
    best_y_pred = best_model.predict(X_test)

    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_y_pred, target_names=['No Churn', 'Churn']))

    cm = confusion_matrix(y_test, best_y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Plot & save Confusion Matrix for Voting Classifier / Best Model
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'pipeline_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Plot Model Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x='Test Accuracy', y='Model', palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0.6, 0.9)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'pipeline_model_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to: {comp_path}")

    print("\nPipeline Execution Finished Successfully!")

if __name__ == '__main__':
    main()
