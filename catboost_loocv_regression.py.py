import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

import os

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE_DIR, "datasets/capacitance_dataset.csv"))

y = df["Capacitance"]
X = df.drop(columns=["Capacitance"])

cat_features = ["Material", "Synthesis", "Electrolyte"]

# CatBoost model : strong regularization for low data
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="MAE",
    l2_leaf_reg=5,
    random_strength=1.0,
    bagging_temperature=0.5,
    verbose=False,
    random_seed=42
)

loo = LeaveOneOut()

mae_scores = []
y_true_all = []
y_pred_all = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train,y_train,cat_features=cat_features)

    y_pred = model.predict(X_test)
    mae_scores.append(abs(y_pred[0] - y_test.values[0]))

    y_true_all.append(y_test.values[0])
    y_pred_all.append(y_pred[0])

print(f"LOOCV MAE: {np.mean(mae_scores):.2f} F/g")

plt.scatter(y_true_all, y_pred_all)
plt.plot([min(y_true_all), max(y_true_all)],
         [min(y_true_all), max(y_true_all)],
         linestyle='--')
plt.xlabel("Actual Capacitance")
plt.ylabel("Predicted Capacitance")
plt.title("LOOCV: Actual vs Predicted")
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.pdf"), bbox_inches='tight')
plt.show()

residuals = np.array(y_true_all) - np.array(y_pred_all)
plt.scatter(y_pred_all, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(os.path.join(PLOTS_DIR, "residual.pdf"), bbox_inches='tight')
plt.show()

plt.hist(residuals, bins=10)
plt.title("Error Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.savefig(os.path.join(PLOTS_DIR, "error_distribution.pdf"), bbox_inches='tight')
plt.show()

importances = model.get_feature_importance()
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.pdf"), bbox_inches='tight')
plt.show()