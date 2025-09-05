# CarDekho Price Prediction App

A machine learning-powered web application that predicts used car prices based on various features like brand, model, year, mileage, and more. Built with Python, Scikit-learn, and Streamlit.

## Features

- **Accurate Price Prediction**: Utilizes a trained Random Forest model for reliable price estimation
- **User-friendly Interface**: Simple and intuitive web interface built with Streamlit
- **Feature Engineering**: Includes smart feature engineering like car age calculation and luxury brand detection
- **Responsive Design**: Works seamlessly on both desktop and mobile devices
- **Sample Data**: Try the app with pre-loaded sample data

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/carsdekho-prediction.git
   cd carsdekho-prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Required Packages

All required packages are listed in `requirements.txt`:

```
streamlit==1.29.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
plotly==5.15.0
```

## How to Run

1. Ensure you have all the required packages installed
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your web browser and navigate to `http://localhost:8501`

## Model Details

The prediction model uses a Random Forest Regressor with the following features:
- **Brand**: Car manufacturer (e.g., Maruti, Hyundai, Honda)
- **Year**: Manufacturing year
- **Kilometers Driven**: Total distance covered by the car
- **Fuel Type**: Petrol, Diesel, CNG, etc.
- **Seller Type**: Individual or Dealer
- **Transmission**: Manual or Automatic
- **Owner**: Number of previous owners
- **Engine**: Engine displacement in CC
- **Max Power**: Maximum power in bhp
- **Seats**: Number of seats
- **Mileage**: Fuel efficiency in kmpl
- **Car Age**: Calculated as current year - manufacturing year
- **Luxury Brand**: Boolean indicating if the brand is considered luxury

## Performance Metrics

The model has been evaluated with the following metrics:
- R² Score: 0.76

## Usage

1. Fill in the car details in the input form
2. Click "Predict Price" to get the estimated price
3. Use the "Load Sample Data" button to try with example values
4. Adjust the sliders and dropdowns to see how different features affect the price

## Project Structure

```
carsdekho_pred/
├── app.py                 # Main Streamlit application
├── carsDekho.py           # Model training and preprocessing code
├── best_model.pkl         # Trained model
├── encoder.pkl            # Feature encoder
├── scaler.pkl             # Feature scaler
├── feature_names.pkl      # List of feature names
├── model_columns.pkl      # Model columns information
├── cardekho.csv           # Dataset
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [CarDekho](https://www.cardekho.com/)
- Built with ❤️ using Python and Streamlit
