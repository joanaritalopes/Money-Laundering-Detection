# Anti-Money Laundering detection with Machine Learning

This project applies machine learning to detect fraud transactions from a banking dataset.
It demonstrates a complete ML workflow from data preparation → feature engineering → feature selection → handling imbalance with SMOTE → model training → model evaluation.

# Workflow

1. Data Preparation
	•	Load and clean raw transaction data.
	•	Detect and flag outliers (IQR method).
	•	Create the final dataset (df_final) for modeling.

2. Train/Test Split
	•	Stratified split to preserve class balance.
	•	Target variable: Is Laundering.

3. Feature Engineering
	•	Encode categorical variables with OneHotEncoder.
	•	Scale numerical features using RobustScaler.
	•	Handle outliers with an outlier flag instead of dropping rows.

4. Feature Selection
	•	L1 Logistic Regression (LASSO) for sparse, linear feature selection.
	•	Recursive Feature Elimination (RFE) with Random Forest.
	•	Tree-based Feature Importance using Random Forest.

5. Imbalanced Data Handling
	•	Applied SMOTE (Synthetic Minority Oversampling Technique) on the training set.
	•	Only training data is oversampled, test set remains untouched.

6. Model Training - Trained and evaluated multiple classifiers:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting
	•	SVM (with probability calibration)

7. Model Evaluation - For each model, computed:
	•	Precision, Recall, F1-score
	•	ROC-AUC, PR-AUC
	•	Confusion Matrix
	•	ROC Curve & Precision-Recall Curve

# Main libraries used:
	•	pandas
	•	numpy
	•	scikit-learn
	•	imblearn
	•	matplotlib
	•	seaborn

 # Next Steps / Improvements
	•	Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
    •	Experiment with different features
 
