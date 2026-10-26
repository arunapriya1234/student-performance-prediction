
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Predict G3 without using G1 and G2
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

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

# Tune on training data only
search.fit(X_train, y_train)

# Save the best fitted pipeline
joblib.dump(search.best_estimator_, "student_performance_model.pkl")

# Save expected input columns for consistent predictions
joblib.dump(list(X.columns), "model_features.pkl")

print("\nModel saved successfully!")
print("File: student_performance_model.pkl")
print("Best parameters:", search.best_params_)
