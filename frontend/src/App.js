import React, { useEffect, useState } from 'react';
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      console.log('API Response:', data); // Debug log
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

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? '🔄 Analyzing with AI...' : '🔬 Predict Body Fat'}
          </button>
        </form>

        {error && (
          <div className="error-box">
            <strong>⚠️ Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-card">
            <h2>🎯 AI Health Assessment Result</h2>
            
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
                <h3>📋 Personalized Action Plan</h3>
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