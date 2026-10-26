
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Keep the same prediction task: predict G3 without G1 and G2
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(
    include=["object", "str", "category"]
).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=1))
])

param_grid = {
    "regressor__n_estimators": [200, 400],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": [0.7, 1.0]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest parameters:")
print(search.best_params_)
print(f"\nBest CV MAE: {-search.best_score_:.3f}")

# Evaluate the selected model on the untouched test set
predictions = search.best_estimator_.predict(X_test)

print("\nHeld-out test results:")
print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

print("\nActual vs predicted grades:")
print(pd.DataFrame({
    "Actual": y_test.to_numpy()[:10],
    "Predicted": predictions[:10].round(2)
}).to_string(index=False))
