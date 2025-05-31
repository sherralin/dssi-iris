"""
This module automates model training.
"""

# training.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
features = X.columns.tolist()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
score = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", score)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Save model and features
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/iris_model.joblib")
joblib.dump(features, "model/iris_features.joblib")
print("Model and features saved to ./model/")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    argparser.add_argument("--f1_criteria", type=float)
    args = argparser.parse_args()
    run(args.data_path, args.f1_criteria)