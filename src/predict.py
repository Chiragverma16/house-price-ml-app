import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "backend/artifacts/model.pkl"
PIPELINE_FILE = "backend/artifacts/pipeline.pkl"

os.makedirs("backend/artifacts", exist_ok=True)

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

# TRAINING PHASE
housing = pd.read_csv("data/housing.csv")

housing['income_cat'] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    housing = housing.loc[train_index].drop("income_cat", axis=1)

housing_labels = housing["median_house_value"].copy()
housing_features = housing.drop("median_house_value", axis=1)

num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

pipeline = build_pipeline(num_attribs, cat_attribs)
housing_prepared = pipeline.fit_transform(housing_features)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(housing_prepared, housing_labels)

joblib.dump(model, MODEL_FILE, compress=3)
joblib.dump(pipeline, PIPELINE_FILE, compress=3)

print("Model trained and saved successfully.")