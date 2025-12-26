# Body Fat Prediction App

This project is a web application that predicts body fat percentage based on user input of various body measurements. It consists of a Flask backend that handles the prediction logic and a React.js frontend for user interaction.

## Project Structure

```
bodyfat-prediction-app
├── backend
│   ├── app.py                # Flask application code
│   ├── requirements.txt      # Python dependencies
│   ├── models
│   │   └── model.pkl         # Pre-trained model for predictions
│   ├── static                # Static files (images, CSS, JS)
│   └── templates             # HTML templates for dynamic content
├── frontend
│   ├── public
│   │   ├── index.html        # Main HTML file for React app
│   │   └── manifest.json     # Metadata for the web application
│   ├── src
│   │   ├── App.js            # Main React component
│   │   ├── components
│   │   │   ├── Form.js       # User input form component
│   │   │   └── Result.js     # Component to display prediction results
│   │   ├── index.js          # Entry point for React app
│   │   └── styles
│   │       └── App.css       # Styles for the React application
│   ├── package.json          # npm configuration file
│   └── README.md             # Documentation for the frontend
├── README.md                 # Documentation for the entire project
└── .gitignore                # Files and directories to ignore by Git
```

## Features

- **User Input**: Collects body measurements through a user-friendly form.
- **Prediction**: Utilizes a pre-trained RandomForestRegressor model to predict body fat percentage.
- **Health Assessment**: Provides health category, risk level, and recommendations based on the predicted body fat percentage.
- **Logging**: Records user input data for future analysis and improvements.

## Setup Instructions

### Backend

1. Navigate to the `backend` directory.
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```
   python app.py
   ```

### Frontend

1. Navigate to the `frontend` directory.
2. Install the required npm packages:
   ```
   npm install
   ```
3. Start the React application:
   ```
   npm start
   ```

## Usage

1. Open your web browser and go to `http://localhost:3000` to access the application.
2. Enter your body measurements in the form and submit to receive your body fat prediction and health assessment.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License.