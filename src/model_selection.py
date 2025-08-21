# 6. Model Selection, 7. Model Training, 8. Model Evaluation

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preparation import df_final
from src.feature_engineering import (
    features_train_test,
    features_transformation
)

X_train, X_test, y_train, y_test = features_train_test(df_final)
X_train, X_test = features_transformation(X_train, X_test)


# --------------------------------
# Try several models and Fine-tune
# --------------------------------

def evaluate_model(model, X_test, y_test):
    '''
    Evaluate a classifier with multiple metrics after training (SMOTE applied on train set).
    '''
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # for ROC/PR curves

    # Basic metrics
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print(
        "PR-AUC (Average Precision):",
        round(
            average_precision_score(
                y_test,
                y_prob),
            3))

    # Detailed report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
        'Not Fraud', 'Fraud'], yticklabels=[
        'Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(
        fpr,
        tpr,
        label=f'ROC curve (AUC = {
            roc_auc_score(
                y_test,
                y_prob):.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(
        recall,
        precision,
        label=f'PR curve (AP = {
            average_precision_score(
                y_test,
                y_prob):.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob)
    }


models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'Model name: {name}')
    evaluate_model(model, X_test, y_test)


# ---------- Approach for the decided model ----------

# pipeline = Pipeline(
#     steps=[
#         ('preprocessor', preprocessor),
#         ('smote', SMOTE(random_state=42)), # imbalanced data - generate rows for minority class
#         ('model', RandomForestClassifier(random_state=42))
#     ]
# )
