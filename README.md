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

# Evaluation Metrics usually for Classification models
	•	Accuracy: % of total correct predictions (correct predictions/total predictions) (balanced)
	•	Precison: how many correct out of the predicted positives (when we don't want FP)
	•	Recall: how many correct out of the actual positives (when we don't want FN - example: decease detection, fraud detection) -> TP rate
	•	F1-score: between Precision and Recall (imbalanced) -> a high F1 score, means a good balance between TP avoid FP
	•	Confusion Matrix: TP, TN, FP, FN grid
	•	ROC-AUC: area under ROC curve measures probability of Positivs > Negatives, plot TP rate vs FP rate (binary clasifications -> especially imbalanced data)
	•	PR-AUC: area under plot Precision vs Recall curve -> focuses on minority class performance (highly imbalanced data) 

 # Next Steps / Improvements
     •	Unsupervised learning to detect unknown laudering patterns
	•	Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
    •	Experiment with different features
	•	Create Unit tests for code expected behavior validation
