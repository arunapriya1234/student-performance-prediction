
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Predict final grade without G1 and G2
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

# Identify feature types
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

# Preprocessing
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

# Models to compare
models = {
    "Average Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# Five-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1
    )

    results.append({
        "Model": name,
        "Mean MAE": -scores["test_mae"].mean(),
        "MAE Std": scores["test_mae"].std(),
        "Mean RMSE": -scores["test_rmse"].mean(),
        "Mean R2": scores["test_r2"].mean()
    })

comparison = pd.DataFrame(results).sort_values("Mean MAE")

print("\n5-FOLD CROSS-VALIDATION RESULTS")
print(comparison.round(3).to_string(index=False))

comparison.to_csv("cross_validation_results.csv", index=False)

print("\nSaved to cross_validation_results.csv")
