import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def make_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

models = {}
preprocessors = {}
rf_components = {}
target_scaler = None


def load_models():
    """Load all trained models and preprocessors"""
    base = Path(__file__).resolve().parent
    model_dir = base / "model" / "saved_models"

    try:
        m1_path = model_dir / "model1.keras"
        if m1_path.exists():
            models["persistence"] = keras.models.load_model(str(m1_path))
            print("Loaded Model 1: First Year Persistence (Hybrid RF+NN)")

        rf_path = model_dir / "rf_leaf_model.pkl"
        encoder_path = model_dir / "rf_leaf_encoder.pkl"
        prep_path = model_dir / "preprocessor.pkl"

        if rf_path.exists():
            rf_components["rf_model"] = joblib.load(str(rf_path))
            print("Loaded RF leaf model")
        if encoder_path.exists():
            rf_components["leaf_encoder"] = joblib.load(str(encoder_path))
            print("Loaded RF leaf encoder")
        if prep_path.exists():
            preprocessors["persistence"] = joblib.load(str(prep_path))
            print("Loaded persistence preprocessor")

        m2_path = model_dir / "model2.keras"
        if m2_path.exists():
            models["dropout"] = keras.models.load_model(str(m2_path))
            print("Loaded Model 2: Dropout Prediction")

        prep2_path = model_dir / "preprocessor_model2.pkl"
        if prep2_path.exists():
            preprocessors["dropout"] = (
                joblib.load("model/saved_models/preprocessor.pkl")
                if os.path.exists("model/saved_models/preprocessor.pkl")
                else None
            )
            print("Loaded preprocessor for Model 2")
        m3_path = model_dir / "model3.keras"
        if m3_path.exists():
            models["gpa"] = keras.models.load_model(str(m3_path))
            print("Loaded Model 3: Academic Performance (GPA)")

        prep3_path = model_dir / "preprocessor_model3.pkl"
        scaler3_path = model_dir / "target_scaler_model3.pkl"
        if prep3_path.exists():
            preprocessors["gpa"] = (
                joblib.load("model/saved_models/preprocessor.pkl")
                if os.path.exists("model/saved_models/preprocessor.pkl")
                else None
            )
            print("Loaded preprocessor for Model 3")

        if scaler3_path.exists():
            global target_scaler
            target_scaler = joblib.load(str(scaler3_path))
            print("Loaded target scaler for Model 3")
        else:
            target_scaler = MinMaxScaler()
            target_scaler.fit(np.array([[0.0], [4.5]]))
            print("Target scaler for Model 3 not found - initialized new one")

        m4_path = model_dir / "model4.keras"
        if m4_path.exists():
            models["success"] = keras.models.load_model(str(m4_path))
            print("Loaded Model 4: Program Success")

        prep4_path = model_dir / "preprocessor_model4.pkl"
        if prep4_path.exists():
            preprocessors["success"] = (
                joblib.load("model/saved_models/preprocessor.pkl")
                if os.path.exists("model/saved_models/preprocessor.pkl")
                else None
            )
            print("Loaded preprocessor for Model 4")

    except Exception as e:
        print(f"Error loading models: {e}")


load_models()


def create_preprocessor(include_second_term=True, include_engineered=False):
    """
    Create preprocessor (only used for training; do NOT use in production prediction).
    """

    if include_engineered:
        num_cols = ["First_Term_GPA", "HS_Average", "Math_Score", "GPA_Delta"]
        bin_cols = ["Low_GPA", "High_HS"]
    elif include_second_term:
        num_cols = [
            "First_Term_GPA",
            "Second_Term_GPA",
            "HS_Average",
            "Math_Score",
        ]
        bin_cols = []
    else:
        num_cols = ["First_Term_GPA", "HS_Average", "Math_Score"]
        bin_cols = []

    cat_cols = [
        "First_Language",
        "Funding",
        "FastTrack",
        "Coop",
        "Residency",
        "Gender",
        "Prev_Education",
        "Age_Group",
        "English_Grade",
    ]

    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant", fill_value="Unknown"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]

    if bin_cols:
        transformers.append(("bin", "passthrough", bin_cols))

    return ColumnTransformer(transformers=transformers)


def preprocess_input(data):
    """Convert numeric codes to categorical strings and handle missing values"""
    processed = {}

    processed["First_Term_GPA"] = float(data.get("First_Term_GPA", 3.0))
    processed["Second_Term_GPA"] = (
        float(data.get("Second_Term_GPA", 3.0))
        if "Second_Term_GPA" in data
        else None
    )
    processed["HS_Average"] = float(data.get("HS_Average", 80.0))
    processed["Math_Score"] = float(data.get("Math_Score", 25.0))

    processed["First_Language"] = str(data.get("First_Language", "1"))
    processed["Funding"] = str(data.get("Funding", "2"))
    processed["FastTrack"] = str(data.get("FastTrack", "2"))
    processed["Coop"] = str(data.get("Coop", "2"))
    processed["Residency"] = str(data.get("Residency", "1"))
    processed["Gender"] = str(data.get("Gender", "2"))
    processed["Prev_Education"] = str(data.get("Prev_Education", "1"))
    processed["Age_Group"] = str(data.get("Age_Group", "3"))
    processed["English_Grade"] = str(data.get("English_Grade", "7"))

    for key in [
        "Prev_Education",
        "Age_Group",
        "First_Language",
        "English_Grade",
    ]:
        if processed[key] in ["?", "0"]:
            processed[key] = "Unknown"

    return processed


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": {
                k: k in models
                for k in ["persistence", "dropout", "gpa", "success"]
            },
            "message": "Neural Networks API is running",
        }
    )


@app.route("/api/predict/persistence", methods=["POST"])
def predict_persistence():
    """Predict first year persistence using Model 1 (Hybrid RF+NN)"""
    try:
        if "persistence" not in models:
            return (
                jsonify(
                    {"success": False, "error": "Persistence model not loaded"}
                ),
                500,
            )
        if "persistence" not in preprocessors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Persistence preprocessor not loaded",
                    }
                ),
                500,
            )

        data = request.json or {}
        input_data = preprocess_input(data)
        df = pd.DataFrame([input_data])

        preprocessor = preprocessors["persistence"]
        X_proc = preprocessor.transform(df)
        X_proc = np.asarray(
            X_proc.todense() if hasattr(X_proc, "todense") else X_proc
        )
        if "rf_model" in rf_components and "leaf_encoder" in rf_components:
            rf_model = rf_components["rf_model"]
            leaf_encoder = rf_components["leaf_encoder"]
            X_leaves = rf_model.apply(X_proc)
            X_leaves_enc = leaf_encoder.transform(X_leaves)
            X_final = np.hstack([X_proc, X_leaves_enc])
        else:
            X_final = X_proc

        print(
            f"predict_persistence: X_proc.shape={X_proc.shape}, X_final.shape={X_final.shape}"
        )
        print("First:", X_final)

        prediction = models["persistence"].predict([X_final], verbose=0)
        probability = float(prediction[0][0])

        result = "Will Persist" if probability > 0.5 else "At Risk"
        risk_level = (
            "Low"
            if probability > 0.7
            else "Medium" if probability > 0.4 else "High"
        )

        return jsonify(
            {
                "success": True,
                "prediction": result,
                "probability": round(probability, 4),
                "risk_level": risk_level,
                "confidence": round(abs(probability - 0.5) * 2, 4),
                "model": "Hybrid RF+NN (Model 1)",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict/dropout", methods=["POST"])
def predict_dropout():
    """Predict dropout using Model 2"""
    try:
        if "dropout" not in models or "dropout" not in preprocessors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Model or preprocessor not loaded",
                    }
                ),
                500,
            )

        data = request.json or {}
        df = pd.DataFrame([data])

        preprocessor = preprocessors["dropout"]

        X_proc = preprocessor.transform(df)
        if hasattr(X_proc, "todense"):
            X_proc = np.asarray(X_proc.todense())
        expected_shape = models["dropout"].input_shape[1]
        if X_proc.shape[1] < expected_shape:
            X_proc = np.pad(
                X_proc,
                ((0, 0), (0, expected_shape - X_proc.shape[1])),
                "constant",
            )
        elif X_proc.shape[1] > expected_shape:
            X_proc = X_proc[:, :expected_shape]

        print(f"predict_dropout: X_proc.shape={X_proc.shape}")

        prediction = models["dropout"].predict(X_proc, verbose=0)
        probability = float(prediction[0][0])

        result = "Will Dropout" if probability > 0.5 else "Will Continue"
        risk_level = (
            "High"
            if probability > 0.7
            else "Medium" if probability > 0.4 else "Low"
        )

        return jsonify(
            {
                "success": True,
                "prediction": result,
                "dropout_probability": round(probability, 4),
                "risk_level": risk_level,
                "recommendation": get_dropout_recommendation(probability),
                "model": "Dropout NN (Model 2)",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict/gpa", methods=["POST"])
def predict_gpa():
    """Predict second term GPA using Model 3 (Regression)"""
    try:
        if (
            "gpa" not in models
            or "gpa" not in preprocessors
            or target_scaler is None
        ):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Model, preprocessor or scaler not loaded",
                    }
                ),
                500,
            )

        data = request.json or {}
        input_data = preprocess_input(data)

        first_term_gpa = input_data.get("First_Term_GPA", 0)
        hs_average = input_data.get("HS_Average", 0)
        input_data["GPA_Delta"] = hs_average - first_term_gpa
        input_data["Low_GPA"] = int(first_term_gpa < 2.0)
        input_data["High_HS"] = int(hs_average > 80)

        df = pd.DataFrame([input_data])

        preprocessor = preprocessors["gpa"]
        X_proc = preprocessor.transform(df)
        if hasattr(X_proc, "todense"):
            X_proc = np.asarray(X_proc.todense())
        expected_shape = models["gpa"].input_shape[1]
        if X_proc.shape[1] < expected_shape:
            X_proc = np.pad(
                X_proc,
                ((0, 0), (0, expected_shape - X_proc.shape[1])),
                "constant",
            )
        elif X_proc.shape[1] > expected_shape:
            X_proc = X_proc[:, :expected_shape]

        print(f"predict_gpa: X_proc.shape={X_proc.shape}")
        prediction_scaled = models["gpa"].predict(X_proc, verbose=0)
        prediction_scaled = np.asarray(prediction_scaled).reshape(-1, 1)
        predicted_gpa = target_scaler.inverse_transform(prediction_scaled)[0][
            0
        ]
        predicted_gpa = max(0.0, min(4.5, predicted_gpa))
        improvement = predicted_gpa - first_term_gpa
        response = {
            "success": True,
            "predicted_gpa": predicted_gpa,
            "grade_letter": gpa_to_letter(predicted_gpa),
            "improvement": improvement,
            "trend": (
                "Improving"
                if improvement > 0.1
                else "Declining" if improvement < -0.1 else "Stable"
            ),
            "recommendation": get_gpa_recommendation(
                predicted_gpa, improvement
            ),
            "model": "Academic Performance NN (Model 3)",
        }

        return jsonify(make_json_serializable(response))

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict/success", methods=["POST"])
def predict_success():
    """Predict program success using Model 4 (Neural Network)"""
    try:
        if "success" not in models or "success" not in preprocessors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Model or preprocessor not loaded",
                    }
                ),
                500,
            )

        data = request.json or {}
        expected_cols = [
            "First_Term_GPA",
            "HS_Average",
            "Math_Score",
            "First_Language",
            "Funding",
            "FastTrack",
            "Coop",
            "Residency",
            "Gender",
            "Prev_Education",
            "Age_Group",
            "English_Grade",
        ]

        for col in expected_cols:
            if col not in data:
                data[col] = (
                    0
                    if col in ["First_Term_GPA", "HS_Average", "Math_Score"]
                    else "Unknown"
                )

        df = pd.DataFrame([data])
        preprocessor = preprocessors["success"]
        X_proc = preprocessor.transform(df)
        if hasattr(X_proc, "todense"):
            X_proc = X_proc.todense()
        X_proc = np.array(X_proc)

        expected_input_dim = models["success"].input_shape[1]
        if X_proc.shape[1] < expected_input_dim:
            X_proc = np.hstack(
                [
                    X_proc,
                    np.zeros(
                        (X_proc.shape[0], expected_input_dim - X_proc.shape[1])
                    ),
                ]
            )
        elif X_proc.shape[1] > expected_input_dim:
            X_proc = X_proc[:, :expected_input_dim]
        prediction = models["success"].predict(X_proc, verbose=0)
        probability = float(prediction[0][0])

        result = (
            "Will Complete Program"
            if probability > 0.5
            else "At Risk of Not Completing"
        )
        confidence_level = (
            "High"
            if abs(probability - 0.5) > 0.3
            else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
        )

        return jsonify(
            {
                "success": True,
                "prediction": result,
                "probability": round(probability, 4),
                "confidence_level": confidence_level,
                "recommendation": get_success_recommendation(probability),
                "model": "Program Success NN (Model 4)",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict/comprehensive", methods=["POST"])
def predict_comprehensive():
    """Run all predictions for comprehensive student analysis"""
    try:
        data = request.json or {}
        results = {}

        endpoints = [
            ("persistence", predict_persistence),
            ("dropout", predict_dropout),
            ("gpa", predict_gpa),
            ("success", predict_success),
        ]

        for name, endpoint_func in endpoints:
            try:
                with app.test_request_context(json=data):
                    resp = endpoint_func()
                    if isinstance(resp, tuple):
                        response_obj = resp[0]
                    else:
                        response_obj = resp

                    if (
                        hasattr(response_obj, "status_code")
                        and response_obj.status_code == 200
                    ):
                        results[name] = response_obj.get_json()
                    else:
                        try:
                            results[name] = response_obj.get_json()
                        except:
                            results[name] = {
                                "error": f"{name} endpoint returned status {getattr(response_obj, 'status_code', 'unknown') }"
                            }
            except Exception as e:
                results[name] = {"error": str(e)}

        return jsonify(
            {
                "success": True,
                "comprehensive_results": results,
                "student_profile": {
                    "First_Term_GPA": data.get("First_Term_GPA", "N/A"),
                    "HS_Average": data.get("HS_Average", "N/A"),
                },
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/models/info", methods=["GET"])
def models_info():
    model_info = {}

    for name, model in models.items():
        try:
            model_info[name] = {
                "loaded": True,
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "layers": len(model.layers),
            }
        except Exception:
            model_info[name] = {
                "loaded": True,
                "info": "Model details unavailable",
            }

    return jsonify(
        {
            "success": True,
            "models": model_info,
            "rf_components": list(rf_components.keys()),
            "preprocessors": list(preprocessors.keys()),
        }
    )


def gpa_to_letter(gpa):
    if gpa >= 3.7:
        return "A"
    elif gpa >= 3.3:
        return "A-"
    elif gpa >= 3.0:
        return "B+"
    elif gpa >= 2.7:
        return "B"
    elif gpa >= 2.3:
        return "B-"
    elif gpa >= 2.0:
        return "C+"
    elif gpa >= 1.7:
        return "C"
    else:
        return "D"


def get_dropout_recommendation(probability):
    if probability > 0.7:
        return "High risk - Immediate intervention required. Consider academic counseling and support services."
    elif probability > 0.4:
        return "Moderate risk - Monitor closely and offer additional support resources."
    else:
        return "Low risk - Student showing good engagement. Continue current support."


def get_gpa_recommendation(gpa, improvement):
    if gpa < 2.0:
        return "Academic probation risk. Recommend tutoring and study skills workshops."
    elif gpa < 2.5:
        return (
            "Below average performance. Consider additional academic support."
        )
    elif improvement > 0.3:
        return "Strong improvement trend. Encourage continued effort."
    elif improvement < -0.3:
        return "Declining performance. Investigate challenges and provide support."
    else:
        return "Stable performance. Maintain current strategies."


def get_success_recommendation(probability):
    if probability > 0.7:
        return "High likelihood of success. Continue supporting current trajectory."
    elif probability > 0.4:
        return "Moderate success probability. Provide additional guidance and resources."
    else:
        return "Low success probability. Urgent intervention and comprehensive support needed."


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Neural Networks Backend API Starting...")
    print("Endpoints available:")
    print("  - /api/predict/persistence (Model 1 - Hybrid RF+NN)")
    print("  - /api/predict/dropout (Model 2)")
    print("  - /api/predict/gpa (Model 3)")
    print("  - /api/predict/success (Model 4)")
    print("  - /api/predict/comprehensive (All models)")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
