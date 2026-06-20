import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_data import get_merged

FEATURE_COLS = [
    "CoursePrice", "CourseDuration", "CourseRating",
    "TeacherRating", "YearsOfExperience",
    "CourseCategory", "CourseLevel", "CourseType", "PriceBand",
    "DurationBucket", "RatingTier", "ExperienceBucket"
]

CAT_COLS = ["CourseCategory", "CourseLevel", "CourseType",
            "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"]

def prepare_features(df):
    X = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X

def train_models(target="EnrollmentCount"):
    df = get_merged()
    df = df[df[target] > 0]  # only courses with data
    X = prepare_features(df)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": round(mean_absolute_error(y_test, preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "R2": round(r2_score(y_test, preds), 3),
        }
        trained[name] = model

    return trained, results, X_test, y_test

def get_feature_importance(model, feature_names=FEATURE_COLS):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()
    return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False)

def predict_single(input_dict, model):
    """Predict for a single course given user inputs."""
    row = pd.DataFrame([input_dict])
    for col in CAT_COLS:
        le = LabelEncoder()
        # fit on known values to avoid unseen label issues
        known = ["Data Science","Web Development","Business","Design","Marketing",
                 "Cybersecurity","AI & ML","Finance",
                 "Beginner","Intermediate","Advanced",
                 "Video","Live","Hybrid",
                 "Low","Medium","High",
                 "Short","Long",
                 "Junior","Mid","Senior"]
        le.fit(known)
        try:
            row[col] = le.transform(row[col].astype(str))
        except Exception:
            row[col] = 0
    row = row.reindex(columns=FEATURE_COLS, fill_value=0)
    return float(model.predict(row)[0])
