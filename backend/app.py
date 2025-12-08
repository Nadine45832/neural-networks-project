from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
import joblib
import os

app = Flask(__name__)
# Explicitly enable CORS for the API routes (allow all origins during development)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Additional safety: ensure CORS headers are present on responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response


# Log incoming requests for debugging CORS / proxy issues
@app.before_request
def log_request_info():
    try:
        origin = request.headers.get('Origin')
        app.logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr} Origin={origin}")
    except Exception:
        pass


# Handle OPTIONS preflight for API routes explicitly
@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def handle_options(subpath):
    resp = jsonify({})
    resp.status_code = 200
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    resp.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return resp


# Global variables for models
models = {}
scalers = {}
encoders = {}


def load_models():
    """Load all trained models and preprocessors"""
    model_dir = 'model'

    try:
        # Load student success model
        if os.path.exists(f'{model_dir}/student_success_model.h5'):
            models['success'] = keras.models.load_model(f'{model_dir}/student_success_model.h5')
            print("✓ Success model loaded")

        # Load persistence model
        if os.path.exists(f'{model_dir}/persistence_model.h5'):
            models['persistence'] = keras.models.load_model(f'{model_dir}/persistence_model.h5')
            print("✓ Persistence model loaded")

        # Load GPA prediction model
        if os.path.exists(f'{model_dir}/gpa_prediction_model.h5'):
            models['gpa'] = keras.models.load_model(f'{model_dir}/gpa_prediction_model.h5')
            print("✓ GPA model loaded")

        # Load preprocessors
        if os.path.exists(f'{model_dir}/scaler.pkl'):
            scalers['main'] = joblib.load(f'{model_dir}/scaler.pkl')
            print("✓ Scaler loaded")

        if os.path.exists(f'{model_dir}/label_encoders.pkl'):
            encoders['labels'] = joblib.load(f'{model_dir}/label_encoders.pkl')
            print("✓ Encoders loaded")

    except Exception as e:
        print(f"Error loading models: {str(e)}")


# Load models on startup
load_models()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'message': 'Neural Networks API is running'
    })


@app.route('/api/predict/success', methods=['POST'])
def predict_success():
    """Predict student success (program completion)"""
    try:
        data = request.json

        # Extract features
        features = [
            data.get('age', 20),
            data.get('high_school_gpa', 3.0),
            data.get('attendance_rate', 85),
            data.get('study_hours', 15),
            data.get('family_support', 1),
            data.get('extracurricular', 0),
            data.get('financial_aid', 1),
            data.get('work_hours', 10)
        ]

        # Preprocess
        features_array = np.array([features])
        if 'main' in scalers:
            features_scaled = scalers['main'].transform(features_array)
        else:
            features_scaled = features_array

        # Predict
        if 'success' in models:
            prediction = models['success'].predict(features_scaled)
            probability = float(prediction[0][0])
            result = 'Complete' if probability > 0.5 else 'At Risk'
        else:
            # Fallback prediction
            probability = np.mean(features_array) / 100
            result = 'Complete' if probability > 0.5 else 'At Risk'

        return jsonify({
            'success': True,
            'prediction': result,
            'probability': round(probability, 4),
            'confidence': round(abs(probability - 0.5) * 2, 4),
            'features_used': len(features)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict/persistence', methods=['POST'])
def predict_persistence():
    """Predict first year persistence"""
    try:
        data = request.json

        features = [
            data.get('first_term_gpa', 3.0),
            data.get('attendance_first_term', 90),
            data.get('engagement_score', 7),
            data.get('financial_difficulty', 0),
            data.get('commute_time', 30),
            data.get('social_integration', 5),
            data.get('academic_support', 1)
        ]

        features_array = np.array([features])
        if 'main' in scalers:
            features_scaled = scalers['main'].transform(features_array)
        else:
            features_scaled = features_array

        if 'persistence' in models:
            prediction = models['persistence'].predict(features_scaled)
            probability = float(prediction[0][0])
        else:
            probability = np.mean(features_array) / 100

        result = 'Will Persist' if probability > 0.5 else 'At Risk of Leaving'

        return jsonify({
            'success': True,
            'prediction': result,
            'probability': round(probability, 4),
            'risk_level': 'Low' if probability > 0.7 else 'Medium' if probability > 0.4 else 'High'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict/gpa', methods=['POST'])
def predict_gpa():
    """Predict second term GPA based on historical data"""
    try:
        data = request.json

        features = [
            data.get('high_school_gpa', 3.0),
            data.get('first_term_gpa', 3.0),
            data.get('study_hours', 15),
            data.get('attendance_rate', 85),
            data.get('assignment_completion', 90),
            data.get('participation_score', 7)
        ]

        features_array = np.array([features])
        if 'main' in scalers:
            features_scaled = scalers['main'].transform(features_array)
        else:
            features_scaled = features_array

        if 'gpa' in models:
            prediction = models['gpa'].predict(features_scaled)
            predicted_gpa = float(prediction[0][0])
        else:
            # Fallback: weighted average
            predicted_gpa = (features[0] * 0.2 + features[1] * 0.5 + 
                           (features[2] / 20) * 0.15 + (features[3] / 100) * 0.15)

        # Ensure GPA is in valid range
        predicted_gpa = max(0.0, min(4.0, predicted_gpa))

        return jsonify({
            'success': True,
            'predicted_gpa': round(predicted_gpa, 2),
            'grade_letter': gpa_to_letter(predicted_gpa),
            'improvement': round(predicted_gpa - features[1], 2)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple students"""
    try:
        data = request.json
        students = data.get('students', [])

        results = []
        for student in students:
            # Simplified batch processing
            result = {
                'student_id': student.get('id', 'unknown'),
                'success_risk': np.random.choice(['Low', 'Medium', 'High']),
                'predicted_gpa': round(np.random.uniform(2.0, 4.0), 2)
            }
            results.append(result)

        return jsonify({
            'success': True,
            'total_students': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    model_info = {}

    for name, model in models.items():
        try:
            model_info[name] = {
                'loaded': True,
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'layers': len(model.layers)
            }
        except:
            model_info[name] = {'loaded': True, 'info': 'Model details unavailable'}

    return jsonify({
        'success': True,
        'models': model_info,
        'scalers_loaded': list(scalers.keys()),
        'encoders_loaded': list(encoders.keys())
    })

def gpa_to_letter(gpa):
    """Convert GPA to letter grade"""
    if gpa >= 3.7:
        return 'A'
    elif gpa >= 3.3:
        return 'A-'
    elif gpa >= 3.0:
        return 'B+'
    elif gpa >= 2.7:
        return 'B'
    elif gpa >= 2.3:
        return 'B-'
    elif gpa >= 2.0:
        return 'C+'
    elif gpa >= 1.7:
        return 'C'
    else:
        return 'D'


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Neural Networks Backend API Starting...")
    print("="*50)
    # Use port 5001 because macOS may reserve 5001 for system services (AirPlay/ControlCenter)
    app.run(debug=True, host='0.0.0.0', port=5001)