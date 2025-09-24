from flask import Blueprint, request ,render_template
from core import load_session, jsonify
from config import MODEL_DIR
import joblib, os, traceback

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Create the blueprint
modeling_bp = Blueprint('modeling', __name__)



@modeling_bp.route('/modeldev/<file_id>', methods=['GET'])
def model_development_page(file_id):
    try:
        # ✅ Load user's session data
        session_data = load_session(file_id)

        if not session_data:
            return "Invalid file ID", 404

        df = pd.DataFrame(session_data.get('modified', {}))  # Convert dict back to DataFrame

        if df.empty:
            return "Modified dataset not found", 404

        return render_template("modeldev.html", file_id=file_id)

    except Exception as e:
        print(f"❌ Error loading model dev page: {e}")
        traceback.print_exc()
        return "Error loading model development page", 500



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
    if model_cls is None:
        raise ValueError(f"Unknown model: {name}")
    return model_cls() if callable(model_cls) else model_cls()


@modeling_bp.route('/train/<file_id>', methods=['POST'])
def train_model(file_id):
    try:
        session_data = load_session(file_id)
        if not session_data:
            return jsonify({"error": "File ID not found"}), 404

        df = pd.DataFrame(session_data.get("modified", []))
        data = request.get_json()

        objective = data.get('objective', '').lower()
        model_name = data.get('model')
        target_col = data.get('target')
        selected_features = data.get('features', [])
        train_split = float(data.get('train_split', 80)) / 100.0
        cv = int(data.get('cv', 5))

        if not model_name:
            return jsonify({"error": "Model name is required"}), 400

        model = get_model_instance(model_name, objective)

        if df.empty:
            return jsonify({"error": "Dataset is empty"}), 400

        if objective != 'clustering':
            if not target_col or target_col not in df.columns:
                return jsonify({"error": f"Target column '{target_col}' not found in dataset"}), 400
            if not selected_features or not all(col in df.columns for col in selected_features):
                return jsonify({"error": "Selected features are invalid or missing"}), 400

        if objective == 'clustering':
            X = df.select_dtypes(include=['number']).dropna()
            model.fit(X)
            y_pred = model.labels_
            metrics = {'clusters': y_pred.tolist()}
            scores = None

        else:
            X = df[selected_features].dropna()
            y = df[target_col].loc[X.index]

            X = pd.get_dummies(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=1 - train_split, random_state=42
            )

            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv)
            y_pred = model.predict(X_test)

            metrics = {}
            if objective == "classification":
                metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                try:
                    if hasattr(model, "predict_proba"):
                        metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                except:
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
            "model": model_name,
        })

    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
