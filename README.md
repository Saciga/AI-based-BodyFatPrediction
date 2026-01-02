# 🏋️ AI Body Fat Prediction System

AI-powered body fat prediction using Machine Learning, Deep Learning (TensorFlow), and Computer Vision (MediaPipe).

## 🌟 Features

- **Traditional ML Models:** Random Forest, Gradient Boosting, Linear Regression
- **Deep Learning:** TensorFlow Neural Network (98.75% R² accuracy)
- **Computer Vision:** MediaPipe pose detection for automatic measurement extraction
- **Image-Only Prediction:** Upload photo, get instant body fat prediction
- **Personalized Health Recommendations:** Age-based advice, nutrition, exercise plans
- **Full-Stack Web App:** React.js frontend + Flask backend

## 📊 Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Deep Neural Network | 0.9875 | 0.75% | 0.86% |
| Random Forest | 0.9851 | 0.82% | 1.03% |
| Gradient Boosting | 0.9802 | 0.95% | 1.19% |
| Linear Regression | 0.9654 | 1.21% | 1.57% |

## 🚀 Installation

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🛠️ Technologies Used

**Backend:**
- Python 3.x
- Flask
- TensorFlow/Keras
- scikit-learn
- MediaPipe
- OpenCV
- pandas, numpy

**Frontend:**
- React.js
- CSS3

**Visualization:**
- matplotlib
- seaborn

## 📸 Computer Vision Features

- Automatic body measurement extraction from photos
- MediaPipe pose detection (33 landmarks)
- No manual measurements needed
- Works with single front-facing photo

## 📄 License

Educational Project

## 👨‍💻 Author
SACIGA R V  - AI/ML Project