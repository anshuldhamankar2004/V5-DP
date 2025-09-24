# --- routes/modeling.py ---
import io
from flask import Blueprint, request, jsonify, render_template, send_file
import os, traceback, joblib ,io
import numpy as np
import pandas as pd
import polars as pl
import json
from core import load_session
from config import MODEL_DIR
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, mean_squared_error,
    accuracy_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from services.session_manager import load_session
from io import BytesIO
import base64
modeling_bp = Blueprint('modeling', __name__)

# --- Model Map ---
def get_model_instance(name, objective=None):
    name = name.lower()
    model_map = {
        "randomforestclassifier": RandomForestClassifier,
        "xgboostclassifier": XGBClassifier,
        "lgbmclassifier": LGBMClassifier,
        "catboostclassifier": lambda: CatBoostClassifier(verbose=0),
        "svc": SVC,
        "randomforestregressor": RandomForestRegressor,
        "xgboostregressor": XGBRegressor,
        "lgbmregressor": LGBMRegressor,
        "catboostregressor": lambda: CatBoostRegressor(verbose=0),
        "linearregression": LinearRegression,
        "kmeans": lambda: KMeans(n_clusters=3, random_state=42),
    }
    model_cls = model_map.get(name)
    return model_cls() if callable(model_cls) else model_cls()

# --- Routes ---
@modeling_bp.route('/modeldev/<file_id>', methods=['GET'])
def model_dev_page(file_id):
    session_data = load_session(file_id)
    if not session_data:
        return "Error: File ID not found", 404
    return render_template("modeldev.html", file_id=file_id)

@modeling_bp.route('/modeldev/<file_id>/columns', methods=['GET'])
def get_columns(file_id):
    try:
        session_data = load_session(file_id)

        # If parquet is stored in session (decoded as bytes already in load_session)
        if "modified_parquet" in session_data:
            buffer = io.BytesIO(session_data["modified_parquet"])
            df = pd.read_parquet(buffer)
        elif "modified" in session_data:
            df = pd.DataFrame(session_data["modified"])
        else:
            return jsonify({"error": "No modified data found in session."}), 400

        return jsonify({
            "columns": list(df.columns),
            "types": df.dtypes.astype(str).to_dict()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@modeling_bp.route('/modeldev/<file_id>/recommend', methods=['POST'])
def recommend_models(file_id):
    objective = request.json.get("objective", "classification").lower()
    models = {
        "classification": ["RandomForestClassifier", "XGBoostClassifier", "LGBMClassifier", "CatBoostClassifier", "SVC"],
        "regression": ["RandomForestRegressor", "XGBoostRegressor", "LGBMRegressor", "CatBoostRegressor", "LinearRegression"],
        "clustering": ["KMeans"]
    }
    return jsonify({"models": models.get(objective, [])})




@modeling_bp.route('/modeldev/<file_id>/train', methods=['POST'])
def train_model(file_id):
    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({"error": "File ID not found"}), 404

        # Try to load modified data from JSON or fallback to Parquet
        if "modified" in session_data and isinstance(session_data["modified"], list):
            df = pd.DataFrame(session_data["modified"])
        elif "modified_parquet" in session_data:
            try:
                buffer = BytesIO(session_data["modified_parquet"])
                df = pd.read_parquet(buffer)
            except Exception as e:
                return jsonify({"error": f"Failed to decode modified_parquet: {str(e)}"}), 400
        else:
            return jsonify({"error": "No modified data found"}), 400

        if df.empty:
            return jsonify({"error": "Modified data is empty"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "No training parameters provided"}), 400

        # Required fields
        objective = data.get('objective', '').lower()
        model_name = data.get('model')
        target_col = data.get('target')
        selected_features = data.get('features', [])
        train_split = float(data.get('train_split', 80)) / 100.0
        cv = int(data.get('cv', 5))

        # Tuning options
        enable_tuning = data.get('enable_tuning', False)
        tuning_method = data.get('tuning_method', 'gridsearch')
        tuning_iterations = int(data.get('tuning_iterations') or 10)
        min_depth = int(data.get('min_depth') or 3)
        max_depth = int(data.get('max_depth') or 10)
        min_lr = float(data.get('min_lr') or 0.001)
        max_lr = float(data.get('max_lr') or 0.1)
        min_estimators = int(data.get('min_estimators') or 50)
        max_estimators = int(data.get('max_estimators') or 300)
        optimize_metric = data.get('optimize_metric') or 'accuracy'
        enable_early_stopping = bool(data.get('enable_early_stopping', False))
        custom_params = data.get('custom_params')

        if custom_params:
            try:
                search_space = json.loads(custom_params)
            except Exception:
                return jsonify({"error": "Invalid custom_params JSON"}), 400
        else:
            search_space = {
                "max_depth": list(range(min_depth, max_depth + 1)),
                "learning_rate": [round(x, 5) for x in np.linspace(min_lr, max_lr, 10)],
                "n_estimators": list(range(min_estimators, max_estimators + 1, 10))
            }

        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
        if not selected_features and objective != "clustering":
            return jsonify({"error": "Selected features are required for training"}), 400
        if objective != "clustering" and not target_col:
            return jsonify({"error": "Target column is required for supervised learning"}), 400

        model = get_model_instance(model_name, objective)
        if model is None:
            return jsonify({"error": f"Model '{model_name}' not supported"}), 400

        best_params = {}
        scores = None
        metrics = {}

        if objective == 'clustering':
            X = df.select_dtypes(include=['number']).dropna()
            if X.empty:
                return jsonify({"error": "No numeric data available for clustering"}), 400

            model.fit(X)
            metrics = {'clusters': model.labels_.tolist()}

        else:
            X = df[selected_features].dropna()
            y = df[target_col].loc[X.index]

            if X.empty or y.empty:
                return jsonify({"error": "Insufficient data after dropping NA values"}), 400

            X = pd.get_dummies(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 - train_split, random_state=42)

            if enable_tuning:
                if tuning_method == 'gridsearch':
                    tuner = GridSearchCV(
                        model, search_space, cv=cv,
                        scoring=optimize_metric,
                        n_jobs=-1
                    )
                elif tuning_method == 'randomsearch':
                    tuner = RandomizedSearchCV(
                        model, search_space, n_iter=tuning_iterations, cv=cv,
                        scoring=optimize_metric,
                        random_state=42, n_jobs=-1
                    )
                else:
                    return jsonify({"error": f"Tuning method '{tuning_method}' not supported"}), 400

                tuner.fit(X_train, y_train)
                model = tuner.best_estimator_
                best_params = tuner.best_params_
            else:
                model.fit(X_train, y_train)

            scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring=optimize_metric
            )

            y_pred = model.predict(X_test)

            if objective == "classification":
                metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                try:
                    if hasattr(model, "predict_proba"):
                        metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    else:
                        metrics['roc_auc'] = None
                except Exception:
                    metrics['roc_auc'] = None

            elif objective == "regression":
                metrics['mse'] = mean_squared_error(y_test, y_pred)

        filename = f"{file_id}_{model_name}.pkl"
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, path)

        return jsonify({
            "success": True,
            "cv_score": scores.mean() if scores is not None else None,
            "metrics": metrics,
            "model_path": filename,
            "best_params": best_params
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@modeling_bp.route("/modeldev/<file_id>/automl-recommend", methods=["POST"])
def automl_recommend(file_id):
    try:
        data = request.get_json()
        target = data.get("target")
        features = data.get("features", [])
        objective = data.get("objective", "classification")

        result = recommend_best_model(file_id, target, features, objective)

        return jsonify({
            "success": True,
            "best_model": result["best_model"],
            "message": result["message"],
            "top_models": result["top_models"]
        })

    except Exception as e:
        print(f"❌ AutoML recommendation error: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# --- Helper Functions ---
def recommend_best_model(file_id, target_col, features, objective):
    session_data = load_session(file_id)

    # ✅ Use Parquet-based loading
    if "modified_parquet" in session_data:
        df = pd.read_parquet(io.BytesIO(session_data["modified_parquet"]))
    elif "modified" in session_data:
        df = pd.DataFrame(session_data["modified"])
    else:
        raise ValueError("No modified data available in session")

    X = df[features].dropna()
    y = df.loc[X.index, target_col]

    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if objective == "classification":
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
    else:
        clf = LazyRegressor(verbose=0, ignore_warnings=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)

    models.reset_index(inplace=True)

    top_models = []
    for _, row in models.head(5).iterrows():
        model_info = {"Model": row["Model"]}
        for metric in ["Accuracy", "F1 Score", "ROC AUC", "R2", "Adjusted R2", "RMSE", "MAE"]:
            if metric in row:
                model_info[metric] = row.get(metric)
        top_models.append(model_info)

    if not top_models:
        return {
            "best_model": None,
            "message": "❌ AutoML couldn't find any suitable models.",
            "top_models": []
        }

    best_model = top_models[0]["Model"]
    return {
        "best_model": best_model,
        "message": f"✅ AutoML suggests: {best_model}",
        "top_models": top_models
    }

def get_param_grids():
    return {
        "RandomForestClassifier": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "XGBClassifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "LGBMClassifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "CatBoostClassifier": {"iterations": [100, 200], "depth": [4, 6, 8]},
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear", "saga"]},
        "RandomForestRegressor": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "XGBRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "LGBMRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
        "CatBoostRegressor": {"iterations": [100, 200], "depth": [4, 6, 8]},
    }
