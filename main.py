import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

# --- 1. Simulate Data for Double Machine Learning (DML) ---

np.random.seed(42)
n = 1000  # number of samples
p = 10    # number of covariates

# Covariates/features
X = np.random.normal(0, 1, size=(n, p))

# Treatment assignment (binary, depends on X)
def propensity_score(x):
    return 1 / (1 + np.exp(-x[:, 0] + 0.5 * x[:, 1]))

ps = propensity_score(X)
D = np.random.binomial(1, ps)

# Outcome (depends on X and D, with treatment effect = 2)
y = 2 * D + X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 1, n)

# --- 2. DML Algorithm Implementation ---

def dml2(X, D, y, ml_g, ml_m, n_splits=2):
    """
    Double Machine Learning (DML2) for treatment effect estimation.
    X: Covariates
    D: Treatment (binary)
    y: Outcome
    ml_g: ML model for outcome regression
    ml_m: ML model for propensity score
    n_splits: Number of folds for cross-fitting
    """
    n = X.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    theta_list = []

    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Fit outcome model g(X)
        g_model = clone(ml_g)
        g_model.fit(np.hstack([X_train, D_train.reshape(-1, 1)]), y_train)
        g0 = g_model.predict(np.hstack([X_test, np.zeros((len(X_test), 1))]))
        g1 = g_model.predict(np.hstack([X_test, np.ones((len(X_test), 1))]))

        # 2. Fit propensity model m(X)
        m_model = clone(ml_m)
        m_model.fit(X_train, D_train)
        m_hat = m_model.predict_proba(X_test)[:, 1]

        # 3. Compute orthogonalized scores
        y_res = y_test - (g0 * (1 - D_test) + g1 * D_test)
        d_res = D_test - m_hat

        # 4. Estimate theta (treatment effect)
        theta = np.sum(d_res * y_res) / np.sum(d_res ** 2)
        theta_list.append(theta)

    # Average over folds
    theta_hat = np.mean(theta_list)
    return theta_hat

# --- 3. Run DML with Different ML Models ---

# Use Lasso for outcome, LogisticRegression for propensity
lasso = LassoCV(cv=3)
logreg = LogisticRegressionCV(cv=3, solver='lbfgs', max_iter=1000)

theta_hat_lasso = dml2(X, D, y, ml_g=lasso, ml_m=logreg, n_splits=2)
print(f"Estimated ATE (Lasso/LogReg): {theta_hat_lasso:.3f}")

# Use Random Forests for both models
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

theta_hat_rf = dml2(X, D, y, ml_g=rf_reg, ml_m=rf_clf, n_splits=2)
print(f"Estimated ATE (Random Forest): {theta_hat_rf:.3f}")

# --- 4. Compare with Naive OLS ---


ols = LinearRegression()
ols.fit(np.hstack([X, D.reshape(-1, 1)]), y)
ate_ols = ols.coef_[-1]
print(f"Naive OLS ATE: {ate_ols:.3f}")

# --- 5. Evaluate Performance ---

print("\nTrue ATE: 2.0")
print(f"DML (Lasso/LogReg) Error: {abs(theta_hat_lasso - 2.0):.3f}")
print(f"DML (Random Forest) Error: {abs(theta_hat_rf - 2.0):.3f}")
print(f"Naive OLS Error: {abs(ate_ols - 2.0):.3f}")

# --- 6. Visualize Results ---

import matplotlib.pyplot as plt

methods = ['DML Lasso/LogReg', 'DML RF', 'Naive OLS']
estimates = [theta_hat_lasso, theta_hat_rf, ate_ols]
errors = [abs(theta_hat_lasso - 2.0), abs(theta_hat_rf - 2.0), abs(ate_ols - 2.0)]

plt.figure(figsize=(8, 5))
plt.bar(methods, estimates, color=['skyblue', 'lightgreen', 'salmon'])
plt.axhline(2.0, color='black', linestyle='--', label='True ATE')
plt.ylabel('Estimated ATE')
plt.title('Double Machine Learning vs Naive OLS')
plt.legend()
plt.show()

# --- 7. Summary ---

print("\nSummary:")
for m, est, err in zip(methods, estimates, errors):
    print(f"{m}: Estimate={est:.3f}, Error={err:.3f}")

# The code above simulates a DML use case, estimates ATE using DML with different ML models,
# compares with naive OLS, and visualizes the results.