from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables
models = {}
model_scores = {}
scaler = None
feature_importance = None
predictions_history = []

def train_multiple_models():
    """Train multiple ML models and compare performance"""
    global models, model_scores, scaler, feature_importance
    
    if not os.path.exists("bodyfat.csv"):
        print("ERROR: bodyfat.csv not found!")
        return False
    
    # Load dataset
    data = pd.read_csv("bodyfat.csv")
    data = data.drop("Density", axis=1)
    
    X = data.drop("BodyFat", axis=1)
    y = data["BodyFat"]
    
    # Feature scaling for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train multiple models
    model_configs = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'Linear Regression': LinearRegression()
    }
    
    best_model_name = None
    best_score = -float('inf')
    
    for name, model in model_configs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        models[name] = model
        model_scores[name] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2)
        }
        
        print(f"{name} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
    
    # Get feature importance from Random Forest
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_names = data.drop("BodyFat", axis=1).columns
        importance = rf_model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        feature_importance = {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
    
    print(f"\n✅ Best Model: {best_model_name} with R² = {best_score:.4f}")
    return True

def get_health_classification(body_fat, age, weight, height):
    """Enhanced health classification with age and BMI consideration"""
    bmi = (weight / (height ** 2)) * 703  # BMI calculation
    
    # Adjust categories based on age
    if age < 30:
        thresholds = (10, 18, 25)
    elif age < 50:
        thresholds = (12, 20, 27)
    else:
        thresholds = (15, 23, 30)
    
    if body_fat < thresholds[0]:
        category = "Athlete"
        risk = "Very Low"
        advice = "Excellent body composition. Maintain your healthy lifestyle and regular training."
    elif body_fat < thresholds[1]:
        category = "Fit"
        risk = "Low"
        advice = "Good health status. Continue regular exercise and balanced nutrition."
    elif body_fat < thresholds[2]:
        category = "Average"
        risk = "Moderate"
        advice = "Consider improving diet quality and increasing physical activity to 150+ minutes weekly."
    else:
        category = "Obese"
        risk = "High"
        advice = "Higher health risk detected. Consult healthcare provider for personalized weight management plan."
    
    # Add BMI-based insights
    bmi_status = ""
    if bmi < 18.5:
        bmi_status = "underweight"
    elif bmi < 25:
        bmi_status = "normal weight"
    elif bmi < 30:
        bmi_status = "overweight"
    else:
        bmi_status = "obese"
    
    return {
        "category": category,
        "risk": risk,
        "advice": advice,
        "bmi": round(bmi, 2),
        "bmi_status": bmi_status
    }

def generate_personalized_recommendations(body_fat, age, weight, health_info):
    """AI-generated personalized health recommendations"""
    recommendations = []
    
    # Exercise recommendations
    if body_fat > 25:
        recommendations.append({
            "type": "Exercise",
            "icon": "🏃",
            "title": "Cardio Training",
            "description": "Aim for 30-45 minutes of moderate cardio 5 days/week (walking, jogging, cycling)"
        })
        recommendations.append({
            "type": "Exercise",
            "icon": "💪",
            "title": "Strength Training",
            "description": "Include resistance training 3x/week to build lean muscle and boost metabolism"
        })
    else:
        recommendations.append({
            "type": "Exercise",
            "icon": "🎯",
            "title": "Maintain Activity",
            "description": "Continue current exercise routine with mix of cardio and strength training"
        })
    
    # Nutrition recommendations
    if body_fat > 20:
        recommendations.append({
            "type": "Nutrition",
            "icon": "🥗",
            "title": "Caloric Deficit",
            "description": "Reduce daily calories by 300-500 through portion control and nutrient-dense foods"
        })
    
    recommendations.append({
        "type": "Nutrition",
        "icon": "🥤",
        "title": "Hydration",
        "description": f"Drink at least {round(weight * 0.5)} oz of water daily"
    })
    
    # Lifestyle recommendations
    recommendations.append({
        "type": "Lifestyle",
        "icon": "😴",
        "title": "Sleep Quality",
        "description": "Aim for 7-9 hours of quality sleep to support metabolism and recovery"
    })
    
    if age > 40:
        recommendations.append({
            "type": "Health",
            "icon": "🏥",
            "title": "Regular Checkups",
            "description": "Schedule annual physical exams and metabolic health screenings"
        })
    
    return recommendations

def calculate_health_metrics(user_data):
    """Calculate additional health metrics"""
    weight = user_data['Weight']
    height = user_data['Height']
    age = user_data['Age']
    
    # BMI
    bmi = (weight / (height ** 2)) * 703
    
    # Ideal weight range (using BMI 18.5-24.9)
    ideal_weight_min = 18.5 * (height ** 2) / 703
    ideal_weight_max = 24.9 * (height ** 2) / 703
    
    # Waist-to-hip ratio (using abdomen and hip)
    whr = user_data['Abdomen'] / user_data['Hip']
    
    return {
        "bmi": round(bmi, 2),
        "ideal_weight_range": f"{round(ideal_weight_min, 1)} - {round(ideal_weight_max, 1)} lbs",
        "waist_hip_ratio": round(whr, 2),
        "whr_status": "Healthy" if whr < 0.9 else "At Risk"
    }

@app.route('/')
def home():
    return jsonify({
        "message": "AI Body Fat Prediction API v2.0 is running!",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "model_scores": model_scores
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global models, scaler, predictions_history
    
    if not models:
        return jsonify({"error": "Models not trained. Please restart the server."}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            "Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
            "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Prepare user data
        user_data = {key: float(data[key]) for key in required_fields}
        user_df = pd.DataFrame([user_data])
        
        # Scale features
        user_scaled = scaler.transform(user_df)
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            pred = model.predict(user_scaled)[0]
            predictions[name] = round(pred, 2)
        
        # Use best model (Random Forest) for main prediction
        body_fat = predictions.get('Random Forest', list(predictions.values())[0])
        
        # Get health classification with age and weight consideration
        health_info = get_health_classification(
            body_fat, 
            user_data['Age'], 
            user_data['Weight'], 
            user_data['Height']
        )
        
        # Calculate additional metrics
        health_metrics = calculate_health_metrics(user_data)
        
        # Generate personalized recommendations
        recommendations = generate_personalized_recommendations(
            body_fat,
            user_data['Age'],
            user_data['Weight'],
            health_info
        )
        
        # Calculate confidence score based on model agreement
        pred_values = list(predictions.values())
        confidence = 100 - (np.std(pred_values) * 10)  # Lower std = higher confidence
        confidence = max(min(confidence, 100), 50)  # Keep between 50-100
        
        # Store prediction history
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "body_fat": body_fat,
            "category": health_info["category"]
        }
        predictions_history.append(prediction_record)
        if len(predictions_history) > 100:  # Keep last 100 predictions
            predictions_history.pop(0)
        
        # Return comprehensive results
        return jsonify({
            "body_fat_percentage": body_fat,
            "confidence_score": round(confidence, 1),
            "all_model_predictions": predictions,
            "category": health_info["category"],
            "risk": health_info["risk"],
            "advice": health_info["advice"],
            "bmi": health_metrics["bmi"],
            "bmi_status": health_info["bmi_status"],
            "ideal_weight_range": health_metrics["ideal_weight_range"],
            "waist_hip_ratio": health_metrics["waist_hip_ratio"],
            "whr_status": health_metrics["whr_status"],
            "recommendations": recommendations,
            "input_data": user_data
        })
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "models": model_scores,
        "feature_importance": feature_importance,
        "total_predictions": len(predictions_history)
    })

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    if feature_importance:
        return jsonify({"feature_importance": feature_importance})
    return jsonify({"error": "Feature importance not available"}), 404

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    if not predictions_history:
        return jsonify({"message": "No predictions yet"})
    
    body_fats = [p['body_fat'] for p in predictions_history]
    
    stats = {
        "total_predictions": len(predictions_history),
        "average_body_fat": round(np.mean(body_fats), 2),
        "min_body_fat": round(min(body_fats), 2),
        "max_body_fat": round(max(body_fats), 2),
        "std_body_fat": round(np.std(body_fats), 2),
        "recent_predictions": predictions_history[-5:]
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("🚀 Training multiple AI models...")
    if train_multiple_models():
        print("✅ All models trained successfully!")
        print("🌐 Starting Flask server...")
        app.run(debug=True, port=5000)
    else:
        print("❌ Failed to train models. Make sure bodyfat.csv is in the backend directory.")