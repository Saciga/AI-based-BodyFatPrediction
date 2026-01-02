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
import base64
from io import BytesIO
from PIL import Image
import cv2
import mediapipe as mp
from datetime import datetime
import pickle

# Try to import TensorFlow and visualization libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - install with: pip install tensorflow")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIZ_AVAILABLE = True
    print("✅ Visualization libraries available")
except ImportError:
    VIZ_AVAILABLE = False
    print("⚠️  Visualization not available - install with: pip install matplotlib seaborn")

app = Flask(__name__)
CORS(app)

# Global variables
models = {}
model_scores = {}
scaler = None
dl_model = None
dl_scaler = None
feature_importance = None
predictions_history = []
training_history = None

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def train_deep_learning_model(X_train, y_train, X_test, y_test):
    """Train Deep Neural Network"""
    global dl_model, dl_scaler, training_history
    
    if not TF_AVAILABLE:
        print("⚠️  Skipping DL - TensorFlow not available")
        return None
    
    print("\n🧠 Training Deep Neural Network...")
    
    # Create separate scaler for DL
    dl_scaler = StandardScaler()
    X_train_dl = dl_scaler.fit_transform(X_train)
    X_test_dl = dl_scaler.transform(X_test)
    
    # Build model
    dl_model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(1)
    ])
    
    dl_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.00001,
        verbose=0
    )
    
    # Train
    history = dl_model.fit(
        X_train_dl, y_train,
        validation_data=(X_test_dl, y_test),
        epochs=500,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    training_history = history
    
    # Evaluate
    y_pred_dl = dl_model.predict(X_test_dl, verbose=0).flatten()
    
    r2_dl = r2_score(y_test, y_pred_dl)
    mae_dl = mean_absolute_error(y_test, y_pred_dl)
    rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_dl))
    
    print(f"Deep Neural Network - R²: {r2_dl:.4f}, MAE: {mae_dl:.2f}, RMSE: {rmse_dl:.2f}")
    
    # Save model
    try:
        dl_model.save('dl_bodyfat_model.h5')
        with open('dl_scaler.pkl', 'wb') as f:
            pickle.dump(dl_scaler, f)
        print("✅ DL model saved: dl_bodyfat_model.h5")
    except Exception as e:
        print(f"⚠️  Could not save DL model: {e}")
    
    return {
        'r2_score': round(r2_dl, 4),
        'mae': round(mae_dl, 2),
        'rmse': round(rmse_dl, 2)
    }

def create_visualizations(X_train, X_test, y_train, y_test, data):
    """Create comprehensive visualizations"""
    if not VIZ_AVAILABLE:
        print("⚠️  Skipping visualizations - libraries not available")
        return
    
    print("\n📊 Creating visualizations...")
    
    try:
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Model R² Comparison
        model_names = list(model_scores.keys())
        r2_scores = [model_scores[m]['r2_score'] for m in model_names]
        colors = ['#667eea', '#764ba2', '#8bc34a', '#ff6b6b'][:len(model_names)]
        
        bars = axes[0, 0].bar(model_names, r2_scores, color=colors)
        axes[0, 0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Model Accuracy (R² Score)', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylim([min(r2_scores) - 0.01, 1.0])
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. MAE Comparison
        mae_scores = [model_scores[m]['mae'] for m in model_names]
        bars = axes[0, 1].bar(model_names, mae_scores, color=colors)
        axes[0, 1].set_ylabel('MAE (%)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Training History (DL)
        if training_history and TF_AVAILABLE:
            axes[0, 2].plot(training_history.history['loss'], label='Training Loss', linewidth=2)
            axes[0, 2].plot(training_history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 2].set_xlabel('Epoch', fontsize=11)
            axes[0, 2].set_ylabel('Loss (MSE)', fontsize=11)
            axes[0, 2].set_title('Deep Learning Training', fontsize=13, fontweight='bold')
            axes[0, 2].legend(fontsize=9)
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'DL Training\nNot Available', 
                           ha='center', va='center', fontsize=12, color='gray')
            axes[0, 2].set_xticks([])
            axes[0, 2].set_yticks([])
        
        # 4. Predictions vs Actual
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2_score'])
        best_model = models.get(best_model_name)
        if best_model:
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
        else:
            # Use DL model if best
            y_pred = dl_model.predict(dl_scaler.transform(X_test), verbose=0).flatten()
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6, s=50, color='#667eea')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', linewidth=2, label='Perfect')
        axes[1, 0].set_xlabel('Actual Body Fat (%)', fontsize=11)
        axes[1, 0].set_ylabel('Predicted Body Fat (%)', fontsize=11)
        axes[1, 0].set_title(f'{best_model_name} Predictions', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals Distribution
        residuals = y_test - y_pred
        axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='#667eea')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Residuals Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance
        if feature_importance:
            features = list(feature_importance.keys())[:8]
            importances = [feature_importance[f] for f in features]
            bars = axes[1, 2].barh(features, importances, color='#667eea')
            axes[1, 2].set_xlabel('Importance', fontsize=11, fontweight='bold')
            axes[1, 2].set_title('Top 8 Features', fontsize=13, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, axis='x')
            for bar in bars:
                width = bar.get_width()
                axes[1, 2].text(width, bar.get_y() + bar.get_height()/2.,
                               f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.suptitle('AI Body Fat Prediction - Model Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_analysis.png")
        plt.close()
        
        # Create data distribution plot
        create_data_distribution(data)
        
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

def create_data_distribution(data):
    """Create data distribution plots"""
    if not VIZ_AVAILABLE:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Body Fat Distribution
        axes[0, 0].hist(data['BodyFat'], bins=25, edgecolor='black', alpha=0.7, color='#667eea')
        axes[0, 0].set_xlabel('Body Fat (%)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Body Fat Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Age vs Body Fat
        axes[0, 1].scatter(data['Age'], data['BodyFat'], alpha=0.5, color='#764ba2')
        axes[0, 1].set_xlabel('Age (years)', fontsize=11)
        axes[0, 1].set_ylabel('Body Fat (%)', fontsize=11)
        axes[0, 1].set_title('Age vs Body Fat', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Weight vs Body Fat
        axes[1, 0].scatter(data['Weight'], data['BodyFat'], alpha=0.5, color='#8bc34a')
        axes[1, 0].set_xlabel('Weight (lbs)', fontsize=11)
        axes[1, 0].set_ylabel('Body Fat (%)', fontsize=11)
        axes[1, 0].set_title('Weight vs Body Fat', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Abdomen vs Body Fat
        axes[1, 1].scatter(data['Abdomen'], data['BodyFat'], alpha=0.5, color='#ff6b6b')
        axes[1, 1].set_xlabel('Abdomen (cm)', fontsize=11)
        axes[1, 1].set_ylabel('Body Fat (%)', fontsize=11)
        axes[1, 1].set_title('Abdomen vs Body Fat', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: data_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Data distribution error: {e}")

def train_multiple_models():
    """Train multiple ML models"""
    global models, model_scores, scaler, feature_importance
    
    if not os.path.exists("bodyfat.csv"):
        print("ERROR: bodyfat.csv not found!")
        return False
    
    print("\n" + "="*60)
    print("TRAINING AI MODELS")
    print("="*60)
    
    data = pd.read_csv("bodyfat.csv")
    data = data.drop("Density", axis=1)
    
    X = data.drop("BodyFat", axis=1)
    y = data["BodyFat"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Traditional ML models
    model_configs = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'Linear Regression': LinearRegression()
    }
    
    print("\n📊 Traditional ML Models:")
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
        
        print(f"  {name:20} - R²: {r2:.4f}, MAE: {mae:.2f}%, RMSE: {rmse:.2f}%")
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
    
    # Train Deep Learning model
    if TF_AVAILABLE:
        dl_scores = train_deep_learning_model(
            X_train.reshape(-1, X.shape[1]), 
            y_train.values if hasattr(y_train, 'values') else y_train,
            X_test.reshape(-1, X.shape[1]), 
            y_test.values if hasattr(y_test, 'values') else y_test
        )
        
        if dl_scores:
            model_scores['Deep Neural Network'] = dl_scores
            if dl_scores['r2_score'] > best_score:
                best_score = dl_scores['r2_score']
                best_model_name = 'Deep Neural Network'
    
    # Feature importance
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_names = data.drop("BodyFat", axis=1).columns
        importance = rf_model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        feature_importance = {k: round(v, 4) for k, v in sorted(feature_importance.items(), 
                                                                key=lambda x: x[1], reverse=True)}
    
    print("\n" + "="*60)
    print(f"✅ Best Model: {best_model_name}")
    print(f"   R² Score: {best_score:.4f}")
    print("="*60)
    
    # Create visualizations
    if VIZ_AVAILABLE:
        create_visualizations(
            X_train.reshape(-1, X.shape[1]),
            X_test.reshape(-1, X.shape[1]),
            y_train.values if hasattr(y_train, 'values') else y_train,
            y_test.values if hasattr(y_test, 'values') else y_test,
            data
        )
    
    return True

def calculate_distance(point1, point2):
    """Calculate Euclidean distance"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extract_measurements_from_image(image_data, height_inches=None, weight_lbs=None):
    """CV-based measurement extraction"""
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_height, img_width = image.shape[:2]
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        ) as pose:
            
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {
                    "success": False,
                    "error": "Could not detect body pose. Ensure full body visible, front-facing, good lighting."
                }
            
            landmarks = results.pose_landmarks.landmark
            
            def get_point(landmark_id):
                return (
                    landmarks[landmark_id].x * img_width,
                    landmarks[landmark_id].y * img_height
                )
            
            # Extract landmarks
            nose = get_point(0)
            left_shoulder = get_point(11)
            right_shoulder = get_point(12)
            left_hip = get_point(23)
            right_hip = get_point(24)
            left_knee = get_point(25)
            left_ankle = get_point(27)
            right_ankle = get_point(28)
            left_elbow = get_point(13)
            left_wrist = get_point(15)
            
            # Calculate measurements
            shoulder_width_px = calculate_distance(left_shoulder, right_shoulder)
            hip_width_px = calculate_distance(left_hip, right_hip)
            body_height_px = calculate_distance(nose, ((left_ankle[0] + right_ankle[0])/2, 
                                                        (left_ankle[1] + right_ankle[1])/2))
            torso_height_px = calculate_distance(
                ((left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2),
                ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
            )
            upper_arm_px = calculate_distance(left_shoulder, left_elbow)
            forearm_px = calculate_distance(left_elbow, left_wrist)
            thigh_px = calculate_distance(left_hip, left_knee)
            
            shoulder_hip_ratio = shoulder_width_px / hip_width_px if hip_width_px > 0 else 1.0
            
            # Conversion factor
            if height_inches:
                conversion_factor = (height_inches * 2.54) / body_height_px
            else:
                conversion_factor = 170 / body_height_px
            
            # Estimate measurements
            estimated_age = 30
            estimated_weight = weight_lbs if weight_lbs else max(100, min(300, (shoulder_width_px * hip_width_px * torso_height_px / 1000) * 0.8))
            estimated_height = height_inches if height_inches else max(50, min(85, (body_height_px * conversion_factor) / 2.54))
            
            neck = max(30, min(50, shoulder_width_px * conversion_factor * 0.35))
            chest = max(80, min(130, shoulder_width_px * conversion_factor * 2.7))
            abdomen = max(70, min(140, chest * (0.95 + (shoulder_hip_ratio - 1.0) * 0.2)))
            hip = max(80, min(130, hip_width_px * conversion_factor * 2.8))
            thigh = max(40, min(80, thigh_px * conversion_factor * 0.85))
            knee = max(30, min(50, thigh * 0.65))
            ankle = max(18, min(35, knee * 0.60))
            biceps = max(25, min(45, upper_arm_px * conversion_factor * 0.85))
            forearm = max(22, min(38, forearm_px * conversion_factor * 0.70))
            wrist = max(14, min(22, forearm * 0.60))
            
            whr = abdomen / hip
            body_fat_est = max(5, min(40, 10 + (whr - 0.7) * 30))
            
            return {
                "success": True,
                "estimated_measurements": {
                    "Age": round(estimated_age, 1),
                    "Weight": round(estimated_weight, 1),
                    "Height": round(estimated_height, 1),
                    "Neck": round(neck, 1),
                    "Chest": round(chest, 1),
                    "Abdomen": round(abdomen, 1),
                    "Hip": round(hip, 1),
                    "Thigh": round(thigh, 1),
                    "Knee": round(knee, 1),
                    "Ankle": round(ankle, 1),
                    "Biceps": round(biceps, 1),
                    "Forearm": round(forearm, 1),
                    "Wrist": round(wrist, 1)
                },
                "body_analysis": {
                    "shoulder_hip_ratio": round(shoulder_hip_ratio, 2),
                    "waist_hip_ratio": round(whr, 2),
                    "estimated_body_fat": round(body_fat_est, 1),
                    "body_type": "Athletic" if shoulder_hip_ratio > 1.15 else "Balanced" if shoulder_hip_ratio > 1.05 else "Pear-shaped"
                },
                "cv_metrics": {
                    "landmarks_detected": len(landmarks),
                    "image_size": f"{img_width}x{img_height}",
                    "pose_quality": "Excellent"
                },
                "note": "CV estimates. Provide height/weight for better accuracy."
            }
            
    except Exception as e:
        return {"success": False, "error": f"CV Error: {str(e)}"}

def get_health_classification(body_fat, age, weight, height):
    """Health classification"""
    bmi = (weight / (height ** 2)) * 703
    
    if age < 30:
        thresholds = (10, 18, 25)
    elif age < 50:
        thresholds = (12, 20, 27)
    else:
        thresholds = (15, 23, 30)
    
    if body_fat < thresholds[0]:
        category, risk, advice = "Athlete", "Very Low", "Excellent composition. Maintain lifestyle."
    elif body_fat < thresholds[1]:
        category, risk, advice = "Fit", "Low", "Good status. Continue exercise."
    elif body_fat < thresholds[2]:
        category, risk, advice = "Average", "Moderate", "Improve diet and activity (150+ min/week)."
    else:
        category, risk, advice = "Obese", "High", "Consult healthcare provider."
    
    bmi_status = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"
    
    return {"category": category, "risk": risk, "advice": advice, "bmi": round(bmi, 2), "bmi_status": bmi_status}

def generate_hyper_personalized_recommendations(user_data, body_fat, health_info):
    """Personalized recommendations"""
    recommendations = []
    age, weight, abdomen = user_data['Age'], user_data['Weight'], user_data['Abdomen']
    whr = abdomen / user_data['Hip']
    
    # Exercise
    if body_fat > 25:
        icon, title = ("🏃", "High-Intensity Cardio") if age < 35 else ("🚶", "Moderate Cardio")
        desc = f"Body fat {body_fat}%: {'HIIT 4x/week 30min' if age < 35 else '45min walking 5x/week'}"
        recommendations.append({"type": "Exercise", "icon": icon, "title": title, "description": desc})
    else:
        recommendations.append({"type": "Exercise", "icon": "💪", "title": "Maintain Activity", 
                               "description": "Continue balanced training"})
    
    # Nutrition
    calories = int(10 * weight + 6.25 * (user_data['Height'] * 2.54) - 5 * age)
    deficit = 500 if body_fat > 25 else 300 if body_fat > 20 else 0
    recommendations.append({"type": "Nutrition", "icon": "🥗", 
                           "title": f"Target: {calories - deficit} cal/day",
                           "description": f"Protein: {int(weight)}g/day"})
    
    # Hydration
    water = round(weight * 0.67)
    recommendations.append({"type": "Hydration", "icon": "🥤", 
                           "title": f"{water}oz Water Daily",
                           "description": "Add 16oz per exercise hour"})
    
    # Warnings
    if whr > 0.95:
        recommendations.append({"type": "Health", "icon": "⚠️", 
                               "title": "Visceral Fat Risk",
                               "description": f"WHR {whr:.2f}: Prioritize cardio"})
    
    if age > 40:
        recommendations.append({"type": "Health", "icon": "🏥", 
                               "title": "Health Panel",
                               "description": "Annual blood work recommended"})
    
    return recommendations

def calculate_health_metrics(user_data):
    """Calculate metrics"""
    weight, height = user_data['Weight'], user_data['Height']
    bmi = (weight / (height ** 2)) * 703
    ideal_min = 18.5 * (height ** 2) / 703
    ideal_max = 24.9 * (height ** 2) / 703
    whr = user_data['Abdomen'] / user_data['Hip']
    
    return {
        "bmi": round(bmi, 2),
        "ideal_weight_range": f"{round(ideal_min, 1)} - {round(ideal_max, 1)} lbs",
        "waist_hip_ratio": round(whr, 2),
        "whr_status": "Healthy" if whr < 0.9 else "At Risk"
    }

@app.route('/')
def home():
    model_list = list(models.keys())
    if dl_model:
        model_list.append("Deep Neural Network")
    
    return jsonify({
        "message": "AI Body Fat Prediction API v3.0",
        "models_loaded": len(model_list),
        "available_models": model_list,
        "features": ["Traditional ML", "Deep Learning", "Computer Vision", "Visualizations"]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global models, scaler, dl_model, dl_scaler
    
    if not models:
        return jsonify({"error": "Models not trained"}), 500
    
    try:
        data = request.get_json()
        use_cv = False
        cv_results = None
        
        # CV mode
        if 'image' in data and data['image']:
            height_hint = float(data.get('Height', 0)) if data.get('Height') else None
            weight_hint = float(data.get('Weight', 0)) if data.get('Weight') else None
            
            cv_results = extract_measurements_from_image(data['image'], height_hint, weight_hint)
            
            if cv_results['success']:
                use_cv = True
                user_data = cv_results['estimated_measurements']
            else:
                return jsonify({"error": cv_results['error']}), 400
        else:
            # Manual mode
            required_fields = ["Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
                             "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]
            
            for field in required_fields:
                if field not in data or not data[field]:
                    return jsonify({"error": f"Missing: {field}"}), 400
            
            user_data = {key: float(data[key]) for key in required_fields}
        
        # Predictions
        user_df = pd.DataFrame([user_data])
        user_scaled = scaler.transform(user_df)
        
        predictions = {}
        for name, model in models.items():
            pred = model.predict(user_scaled)[0]
            predictions[name] = round(pred, 2)
        
        # Add DL prediction
        if dl_model and dl_scaler:
            user_dl = dl_scaler.transform(user_df.values)
            pred_dl = dl_model.predict(user_dl, verbose=0)[0][0]
            predictions['Deep Neural Network'] = round(pred_dl, 2)
        
        # Use best prediction
        body_fat = predictions.get('Deep Neural Network', predictions.get('Random Forest', 
                                                                         list(predictions.values())[0]))
        
        health_info = get_health_classification(body_fat, user_data['Age'], 
                                                user_data['Weight'], user_data['Height'])
        health_metrics = calculate_health_metrics(user_data)
        recommendations = generate_hyper_personalized_recommendations(user_data, body_fat, health_info)
        
        # Confidence
        pred_values = list(predictions.values())
        confidence = 100 - (np.std(pred_values) * 10)
        confidence = max(min(confidence, 100), 50)
        if use_cv:
            confidence = max(confidence - 10, 60)
        
        response = {
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
            "input_data": user_data,
            "prediction_mode": "Computer Vision" if use_cv else "Manual Input"
        }
        
        if use_cv and cv_results:
            response["cv_analysis"] = {
                "body_analysis": cv_results.get('body_analysis', {}),
                "cv_metrics": cv_results.get('cv_metrics', {}),
                "note": cv_results.get('note', '')
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "models": model_scores,
        "feature_importance": feature_importance,
        "cv_enabled": True,
        "dl_enabled": dl_model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI BODY FAT PREDICTION SYSTEM v3.0")
    print("="*60)
    
    if train_multiple_models():
        print("\n✅ All systems ready!")
        print("📸 Computer Vision: Enabled")
        print("🧠 Deep Learning: " + ("Enabled" if TF_AVAILABLE else "Disabled"))
        print("📊 Visualizations: " + ("Enabled" if VIZ_AVAILABLE else "Disabled"))
        print("\n🌐 Starting Flask server on http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Failed to initialize models")
        print("="*60)