import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    Age: '',
    Weight: '',
    Height: '',
    Neck: '',
    Chest: '',
    Abdomen: '',
    Hip: '',
    Thigh: '',
    Knee: '',
    Ankle: '',
    Biceps: '',
    Forearm: '',
    Wrist: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [showAllModels, setShowAllModels] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [cvMode, setCvMode] = useState(false);

  const API_URL = 'http://localhost:5000/api';

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/model-info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Preview image
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        setImageData(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setImagePreview(null);
    setImageData(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      let payload;
      
      if (cvMode && imageData) {
        // CV Mode: Only image required, optional height/weight for calibration
        payload = {
          image: imageData,
          Height: formData.Height || null,
          Weight: formData.Weight || null
        };
      } else if (imageData && !cvMode) {
        // Hybrid mode: Image + manual measurements
        payload = {
          ...formData,
          image: imageData
        };
      } else {
        // Manual mode: All measurements required
        payload = formData;
      }

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      console.log('API Response:', data);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    const colors = {
      'Very Low': '#4caf50',
      'Low': '#8bc34a',
      'Moderate': '#ff9800',
      'High': '#f44336'
    };
    return colors[risk] || '#9e9e9e';
  };

  const getCategoryColor = (category) => {
    const colors = {
      'Athlete': '#4caf50',
      'Fit': '#8bc34a',
      'Average': '#ff9800',
      'Obese': '#f44336'
    };
    return colors[category] || '#667eea';
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🏋️ AI Body Fat Predictor v2.0</h1>
          <p>Advanced multi-model prediction with personalized health insights</p>
          {modelInfo && modelInfo.models && (
            <div className="model-badge">
              <span>🤖 {Object.keys(modelInfo.models).length} AI Models Active</span>
            </div>
          )}
        </header>

        <form onSubmit={handleSubmit} className="prediction-form">
          
          {/* NEW: Image Upload Section */}
          <div className="form-section image-upload-section">
            <h3>📸 Body Photo Analysis</h3>
            
            <div className="cv-mode-toggle">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={cvMode}
                  onChange={(e) => setCvMode(e.target.checked)}
                />
                <span className="toggle-text">
                  {cvMode ? '🤖 AI Vision Mode (Photo Only)' : '📝 Manual + Photo Mode'}
                </span>
              </label>
              <p className="mode-description">
                {cvMode 
                  ? '✨ Upload photo only - AI will extract all measurements automatically!' 
                  : '📏 Upload photo for reference, enter measurements manually'}
              </p>
            </div>
            
            {!imagePreview ? (
              <div className="image-upload-area">
                <input
                  type="file"
                  id="image-upload"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={{ display: 'none' }}
                />
                <label htmlFor="image-upload" className="upload-label">
                  <div className="upload-icon">📷</div>
                  <div className="upload-text">Click to upload full-body photo</div>
                  <div className="upload-subtext">
                    {cvMode 
                      ? 'Front-facing, full body visible, good lighting - AI will do the rest!' 
                      : 'PNG, JPG up to 10MB'}
                  </div>
                </label>
              </div>
            ) : (
              <div className="image-preview-container">
                <img src={imagePreview} alt="Body preview" className="image-preview" />
                <button type="button" onClick={removeImage} className="remove-image-btn">
                  ✕ Remove Photo
                </button>
              </div>
            )}
          </div>

          {!cvMode && (
            <div className="form-section">
              <h3>📊 Basic Information</h3>
              <div className="form-row">
                <div className="form-group">
                  <label>Age (years)</label>
                  <input
                    type="number"
                    name="Age"
                    value={formData.Age}
                    onChange={handleChange}
                    required
                    min="1"
                    max="120"
                    step="1"
                    placeholder="e.g., 25"
                  />
                </div>
                <div className="form-group">
                  <label>Weight (lbs)</label>
                  <input
                    type="number"
                    name="Weight"
                    value={formData.Weight}
                    onChange={handleChange}
                    required
                    min="1"
                    step="0.1"
                    placeholder="e.g., 154.25"
                  />
                </div>
                <div className="form-group">
                  <label>Height (inches)</label>
                  <input
                    type="number"
                    name="Height"
                    value={formData.Height}
                    onChange={handleChange}
                    required
                    min="1"
                    step="0.1"
                    placeholder="e.g., 67.75"
                  />
                </div>
              </div>
            </div>
          )}

          {cvMode && (
            <div className="form-section cv-optional-section">
              <h3>📊 Optional: Calibration Data</h3>
              <p className="optional-note">Providing these improves accuracy by 20-30%</p>
              <div className="form-row">
                <div className="form-group">
                  <label>Age (years) - Optional</label>
                  <input
                    type="number"
                    name="Age"
                    value={formData.Age}
                    onChange={handleChange}
                    min="1"
                    max="120"
                    step="1"
                    placeholder="Optional - helps personalize advice"
                  />
                </div>
                <div className="form-group">
                  <label>Weight (lbs) - ⭐ Recommended</label>
                  <input
                    type="number"
                    name="Weight"
                    value={formData.Weight}
                    onChange={handleChange}
                    min="1"
                    step="0.1"
                    placeholder="Helps calibrate measurements"
                  />
                </div>
                <div className="form-group">
                  <label>Height (inches) - ⭐ Recommended</label>
                  <input
                    type="number"
                    name="Height"
                    value={formData.Height}
                    onChange={handleChange}
                    min="1"
                    step="0.1"
                    placeholder="Helps calibrate measurements"
                  />
                </div>
              </div>
            </div>
          )}

          {!cvMode && (
            <div className="form-section">
              <h3>📏 Body Circumferences (cm)</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Neck</label>
                <input
                  type="number"
                  name="Neck"
                  value={formData.Neck}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 36.2"
                />
              </div>
              <div className="form-group">
                <label>Chest</label>
                <input
                  type="number"
                  name="Chest"
                  value={formData.Chest}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 93.1"
                />
              </div>
              <div className="form-group">
                <label>Abdomen</label>
                <input
                  type="number"
                  name="Abdomen"
                  value={formData.Abdomen}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 85.2"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Hip</label>
                <input
                  type="number"
                  name="Hip"
                  value={formData.Hip}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 94.5"
                />
              </div>
              <div className="form-group">
                <label>Thigh</label>
                <input
                  type="number"
                  name="Thigh"
                  value={formData.Thigh}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 59.0"
                />
              </div>
              <div className="form-group">
                <label>Knee</label>
                <input
                  type="number"
                  name="Knee"
                  value={formData.Knee}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 37.3"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Ankle</label>
                <input
                  type="number"
                  name="Ankle"
                  value={formData.Ankle}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 21.9"
                />
              </div>
              <div className="form-group">
                <label>Biceps</label>
                <input
                  type="number"
                  name="Biceps"
                  value={formData.Biceps}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 32.0"
                />
              </div>
              <div className="form-group">
                <label>Forearm</label>
                <input
                  type="number"
                  name="Forearm"
                  value={formData.Forearm}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 27.4"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Wrist</label>
                <input
                  type="number"
                  name="Wrist"
                  value={formData.Wrist}
                  onChange={handleChange}
                  required
                  step="0.1"
                  placeholder="e.g., 17.1"
                />
              </div>
            </div>
          </div>
          )}

          <button type="submit" className="submit-btn" disabled={loading || (cvMode && !imageData)}>
            {loading ? '🔄 Analyzing with AI...' : cvMode ? '🤖 Analyze Photo with AI' : '🔬 Predict Body Fat'}
          </button>
          
          {cvMode && !imageData && (
            <p className="cv-warning">⚠️ Please upload a photo to use AI Vision Mode</p>
          )}
        </form>

        {error && (
          <div className="error-box">
            <strong>⚠️ Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-card">
            <h2>🎯 AI Health Assessment Result</h2>
            
            {result.personalization_note && (
              <div className="personalization-badge">
                ✨ {result.personalization_note}
              </div>
            )}

            {result.prediction_mode && (
              <div className="prediction-mode-badge">
                {result.prediction_mode === 'Computer Vision' ? '🤖' : '📝'} {result.prediction_mode}
              </div>
            )}

            {result.cv_analysis && (
              <div className="cv-analysis-box">
                <h4>🤖 Computer Vision Analysis</h4>
                {result.cv_analysis.body_analysis && (
                  <div className="cv-body-analysis">
                    <p><strong>Body Type:</strong> {result.cv_analysis.body_analysis.body_type}</p>
                    <p><strong>Shoulder/Hip Ratio:</strong> {result.cv_analysis.body_analysis.shoulder_hip_ratio}</p>
                    <p><strong>Waist/Hip Ratio:</strong> {result.cv_analysis.body_analysis.waist_hip_ratio}</p>
                    <p><strong>CV Body Fat Estimate:</strong> {result.cv_analysis.body_analysis.estimated_body_fat}%</p>
                  </div>
                )}
                {result.cv_analysis.cv_metrics && (
                  <div className="cv-metrics">
                    <p><strong>Pose Quality:</strong> {result.cv_analysis.cv_metrics.pose_quality}</p>
                    <p><strong>Landmarks Detected:</strong> {result.cv_analysis.cv_metrics.landmarks_detected}</p>
                  </div>
                )}
                {result.cv_analysis.note && (
                  <p className="cv-note">{result.cv_analysis.note}</p>
                )}
              </div>
            )}
            
            <div className="result-main">
              <div className="bodyfat-display">
                <span className="bodyfat-value">{result.body_fat_percentage}%</span>
                <span className="bodyfat-label">Body Fat Percentage</span>
                {result.confidence_score && (
                  <div className="confidence-badge">
                    Confidence: {result.confidence_score}%
                  </div>
                )}
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-icon">🏆</span>
                <span className="metric-label">Category</span>
                <span 
                  className="metric-value"
                  style={{ color: getCategoryColor(result.category) }}
                >
                  {result.category}
                </span>
              </div>
              
              <div className="metric-card">
                <span className="metric-icon">⚠️</span>
                <span className="metric-label">Risk Level</span>
                <span 
                  className="metric-value"
                  style={{ color: getRiskColor(result.risk) }}
                >
                  {result.risk}
                </span>
              </div>

              {result.bmi && (
                <div className="metric-card">
                  <span className="metric-icon">📊</span>
                  <span className="metric-label">BMI</span>
                  <span className="metric-value">{result.bmi}</span>
                  {result.bmi_status && (
                    <span className="metric-sub">{result.bmi_status}</span>
                  )}
                </div>
              )}

              {result.waist_hip_ratio && (
                <div className="metric-card">
                  <span className="metric-icon">📏</span>
                  <span className="metric-label">Waist/Hip Ratio</span>
                  <span className="metric-value">{result.waist_hip_ratio}</span>
                  {result.whr_status && (
                    <span className="metric-sub">{result.whr_status}</span>
                  )}
                </div>
              )}
            </div>

            {result.all_model_predictions && Object.keys(result.all_model_predictions).length > 0 && (
              <div className="model-predictions">
                <button 
                  className="toggle-models-btn"
                  onClick={() => setShowAllModels(!showAllModels)}
                  type="button"
                >
                  {showAllModels ? '▼' : '▶'} View All Model Predictions
                </button>
                
                {showAllModels && (
                  <div className="model-predictions-list">
                    {Object.entries(result.all_model_predictions).map(([model, prediction]) => (
                      <div key={model} className="model-prediction-item">
                        <span className="model-name">🤖 {model}</span>
                        <span className="model-prediction">{prediction}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="advice-box">
              <h4>💡 Primary Recommendation</h4>
              <p>{result.advice}</p>
            </div>

            {result.ideal_weight_range && (
              <div className="ideal-weight-box">
                <h4>🎯 Ideal Weight Range</h4>
                <p>{result.ideal_weight_range}</p>
              </div>
            )}

            {result.recommendations && result.recommendations.length > 0 && (
              <div className="recommendations-section">
                <h3>📋 Hyper-Personalized Action Plan</h3>
                <p className="recommendations-subtitle">Based on your specific measurements and body composition</p>
                <div className="recommendations-grid">
                  {result.recommendations.map((rec, index) => (
                    <div key={index} className="recommendation-card">
                      <div className="rec-header">
                        <span className="rec-icon">{rec.icon}</span>
                        <span className="rec-type">{rec.type}</span>
                      </div>
                      <h4>{rec.title}</h4>
                      <p>{rec.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {modelInfo && modelInfo.models && Object.keys(modelInfo.models).length > 0 && (
          <div className="model-info-section">
            <h3>🤖 AI Model Performance</h3>
            <div className="model-info-grid">
              {Object.entries(modelInfo.models).map(([model, scores]) => (
                <div key={model} className="model-info-card">
                  <h4>{model}</h4>
                  <div className="model-scores">
                    <div className="score-item">
                      <span>R² Score:</span>
                      <strong>{(scores.r2_score * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="score-item">
                      <span>MAE:</span>
                      <strong>{scores.mae}</strong>
                    </div>
                    <div className="score-item">
                      <span>RMSE:</span>
                      <strong>{scores.rmse}</strong>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;