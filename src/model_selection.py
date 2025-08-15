# 6. Model Selection
# 7. Model Training
# 8. Model Evaluation

from imblearn.pipeline import Pipeline

# Approach for the decided model

# pipeline = Pipeline(
#     steps=[
#         ('preprocessor', preprocessor),
#         ('smote', SMOTE(random_state=42)), # Deal with imbalanced data - generate rows for the minority class
#         ('model', RandomForestClassifier(random_state=42))
#     ]
# )
