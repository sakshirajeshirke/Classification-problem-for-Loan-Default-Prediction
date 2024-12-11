from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Assuming X and y are already preprocessed and loaded
# If not, ensure that X is a 2D numeric array and y is a 1D numeric array

# Scale the features for Logistic Regression (Important for models like Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Compare models
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, scoring='roc_auc', cv=5)  # Use scaled X for LogisticRegression
    results[name] = scores.mean()
    print(f"{name} AUC-ROC: {scores.mean()}")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Train Logistic Regression and Random Forest models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Initialize lists to store results
accuracy_results = {}

# Evaluate models using cross-validation
for name, model in models.items():
    # Get predictions using cross-validation
    y_pred_cv = cross_val_predict(model, X, y, cv=5)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y, y_pred_cv)
    
    # Store the result
    accuracy_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy}")

# Optionally, you can display the accuracy results
print("\nAccuracy Comparison:")
for model, accuracy in accuracy_results.items():
    print(f"{model}: {accuracy:.4f}")
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Ensure X and y are in the correct shape and type
assert X.ndim == 2, "X should be a 2D array"
assert y.ndim == 1, "y should be a 1D array"

# Initialize the RandomForestClassifier
best_model = RandomForestClassifier(random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20]
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(best_model, param_grid, scoring='roc_auc', cv=5, verbose=1)

# Fit the model
grid_search.fit(X, y)

# Display the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best AUC-ROC:", grid_search.best_score_)

# Optionally, evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print AUC-ROC on the test set
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC-ROC: {roc_auc}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Hyperparameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers for small datasets
    'max_iter': [100, 200, 300]  # Maximum number of iterations
}

# Initialize Logistic Regression model
logreg = LogisticRegression(random_state=42)

# GridSearchCV for Logistic Regression with cross-validation
grid_search_lr = GridSearchCV(logreg, param_grid_lr, scoring='roc_auc', cv=5, verbose=1)
grid_search_lr.fit(X, y)

# Best model and score for Logistic Regression
best_logreg = grid_search_lr.best_estimator_
best_logreg_auc = grid_search_lr.best_score_

# Display the best parameters and AUC-ROC score
print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best AUC-ROC for Logistic Regression:", best_logreg_auc)

# Test the model on the test set
y_pred_logreg = best_logreg.predict(X_test)
logreg_auc_test = roc_auc_score(y_test, y_pred_logreg)
print(f"Test AUC-ROC for Logistic Regression: {logreg_auc_test}")


