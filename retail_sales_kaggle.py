import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import os
import logging
import argparse
from flask import Flask, jsonify, request, render_template, redirect, url_for

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("retail_forecast.log"), logging.StreamHandler()]
)
logger = logging.getLogger("retail_sales_predictor")

# Flask app setup
app = Flask(__name__)

class RetailSalesPredictor:
    def __init__(self, data_path=None):
        """Initialize the retail sales predictor with optional data path."""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.predictions = None
        
    def load_data(self, data_path=None):
        """Load the retail sales data from CSV."""
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("No data path provided!")
            
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Convert column names to lowercase
            self.data.columns = self.data.columns.str.lower()
            
            # Log available columns for debugging
            logger.info(f"Available columns: {self.data.columns.tolist()}")
            
            # Check if 'date' column exists
            if 'date' not in self.data.columns:
                raise KeyError("'date' column is missing in the dataset.")
            
            # Convert date column to datetime
            self.data['date'] = pd.to_datetime(self.data['date'])
                
            logger.info(f"Data loaded successfully with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, store_id=None, item_id=None):
        """Preprocess the data for time series analysis."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        try:
            logger.info("Preprocessing data")
            
            # Check if 'store' and 'item' columns exist
            if store_id is not None and 'store' in self.data.columns:
                self.data = self.data[self.data['store'] == store_id]
            if item_id is not None and 'item' in self.data.columns:
                self.data = self.data[self.data['item'] == item_id]
            
            # Set date as index
            self.data.set_index('date', inplace=True)
            
            # Sort by date
            self.data = self.data.sort_index()
            
            # Store the processed data
            self.processed_data = self.data
            
            logger.info(f"Data preprocessing completed - shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def visualize_data(self):
        """Visualize the sales data."""
        if not hasattr(self, 'processed_data'):
            raise ValueError("No processed data available. Please preprocess data first.")
            
        try:
            logger.info("Visualizing sales data")
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.processed_data.index, self.processed_data['total amount'])
            plt.title('Store-Item Sales Over Time')
            plt.xlabel('Date')
            plt.ylabel('Total Amount')
            plt.grid(True)
            plt.savefig('sales_data_visualization.png')
            plt.close()
            
            # Plot seasonal decomposition
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Resample to monthly if needed
            monthly_data = self.processed_data['total amount'].resample('M').sum()
            
            # Only decompose if we have enough data
            if len(monthly_data) >= 24:
                decomposition = seasonal_decompose(monthly_data, model='multiplicative', period=12)
                
                plt.figure(figsize=(12, 10))
                plt.subplot(4, 1, 1)
                plt.plot(decomposition.observed)
                plt.title('Observed')
                plt.subplot(4, 1, 2)
                plt.plot(decomposition.trend)
                plt.title('Trend')
                plt.subplot(4, 1, 3)
                plt.plot(decomposition.seasonal)
                plt.title('Seasonal')
                plt.subplot(4, 1, 4)
                plt.plot(decomposition.resid)
                plt.title('Residual')
                plt.tight_layout()
                plt.savefig('seasonal_decomposition.png')
                plt.close()
            
            logger.info("Data visualization completed")
        except Exception as e:
            logger.error(f"Error visualizing data: {str(e)}")
            raise
    
    def split_data(self, test_size=90):
        """Split data into training and testing sets."""
        if not hasattr(self, 'processed_data'):
            raise ValueError("No processed data available. Please preprocess data first.")
            
        try:
            logger.info(f"Splitting data with test size of {test_size} days")
            
            # Split data
            train_data = self.processed_data.iloc[:-test_size]
            test_data = self.processed_data.iloc[-test_size:]
            
            self.train_data = train_data
            self.test_data = test_data
            
            logger.info(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def train_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMA model on the data."""
        if not hasattr(self, 'train_data'):
            raise ValueError("No training data available. Please split the data first.")
            
        try:
            logger.info(f"Training SARIMA model with order={order}, seasonal_order={seasonal_order}")
            
            model = SARIMAX(
                self.train_data['total amount'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.model_fit = model.fit(disp=False)
            
            logger.info("Model training completed")
            logger.info(f"AIC: {self.model_fit.aic}")
            
            return self.model_fit
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    def evaluate_model(self):
    """Evaluate model performance using test data."""
    if not hasattr(self, 'model_fit') or not hasattr(self, 'test_data'):
        raise ValueError("No trained model or test data available.")
        
    try:
        logger.info("Evaluating model performance")
        
        # Use integer-based indices for predictions
        start = len(self.train_data)
        end = start + len(self.test_data) - 1
        
        predictions = self.model_fit.predict(start=start, end=end, dynamic=False)
        
        # Align predictions with test data index
        predictions = pd.Series(predictions, index=self.test_data.index)
        self.test_predictions = predictions
        predictions = predictions.fillna(0)  # Replace NaN with 0 or use another strategy
        
        # Calculate RMSE
        rmse = sqrt(mean_squared_error(self.test_data['total amount'], predictions))
        
        # Calculate MAPE with handling for zero values
        actual = self.test_data['total amount'].values
        pred = predictions.values
        
        # Avoid division by zero by excluding zero values from MAPE calculation
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - pred[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = float('nan')  # or set to a default value like 0
        
        # Convert to float to ensure it's a standard Python type
        mape = float(mape)
        
        logger.info(f"Model RMSE: {rmse}")
        logger.info(f"Model MAPE: {mape}%")
        
        return {
            'rmse': float(rmse),  # Ensure it's a standard Python float
            'mape': mape,
            'predictions': predictions
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

    
    
    def forecast_future(self, steps=30):
        """Forecast future sales."""
        if not hasattr(self, 'model_fit'):
            raise ValueError("No trained model available. Please train the model first.")
            
        try:
            logger.info(f"Forecasting next {steps} days")
            
            # Get forecast
            forecast = self.model_fit.forecast(steps=steps)
            
            # Create future date index
            last_date = self.processed_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'total amount': forecast
            })
            forecast_df.set_index('date', inplace=True)
            
            self.forecast = forecast_df
            
            logger.info("Forecast generated successfully")
            return forecast_df
        except Exception as e:
            logger.error(f"Error forecasting: {str(e)}")
            raise
    
    def save_model(self, filepath='retail_sales_model.pkl'):
        """Save the trained model to a file."""
        if not hasattr(self, 'model_fit'):
            raise ValueError("No trained model available. Please train the model first.")
            
        try:
            logger.info(f"Saving model to {filepath}")
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.model_fit, f)
                
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath='retail_sales_model.pkl'):
        """Load a trained model from a file."""
        try:
            logger.info(f"Loading model from {filepath}")
            
            with open(filepath, 'rb') as f:
                self.model_fit = pickle.load(f)
                
            logger.info("Model loaded successfully")
            return self.model_fit
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# API route for programmatic access
@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        # Get the parameters from the request
        data = request.get_json()
        data_path = data.get('data_path', 'train.csv')
        store_id = data.get('store_id', 1)
        item_id = data.get('item_id', 1)
        test_days = data.get('test_days', 90)
        forecast_days = data.get('forecast_days', 30)

        # Initialize the predictor
        predictor = RetailSalesPredictor(data_path)

        # Load and preprocess data
        predictor.load_data(data_path)
        predictor.preprocess_data(store_id=store_id, item_id=item_id)

        # Split data and train model
        predictor.split_data(test_size=test_days)
        predictor.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))

        # Evaluate model
        evaluation = predictor.evaluate_model()

        # Forecast future sales
        future_sales = predictor.forecast_future(steps=forecast_days)

        # Return the forecast as a JSON response
        response = {
            'evaluation': {
                'RMSE': evaluation['rmse'],
                'MAPE': evaluation['mape']
            },
            'forecast': future_sales.reset_index().to_dict(orient='records')
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Web routes for browser access
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/web-predict', methods=['POST'])
def web_predict():
    try:
        # Get form data
        store_id = int(request.form.get('store_id', 1))
        item_id = int(request.form.get('item_id', 1))
        test_days = int(request.form.get('test_days', 90))
        forecast_days = int(request.form.get('forecast_days', 30))
        
        # Use a default data path that's included in your deployment
        data_path = 'train.csv'
        
        # Initialize the predictor
        predictor = RetailSalesPredictor(data_path)
        
        # Load and preprocess data
        predictor.load_data(data_path)
        predictor.preprocess_data(store_id=store_id, item_id=item_id)
        
        # Split data and train model
        predictor.split_data(test_size=test_days)
        predictor.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        
        # Evaluate model
        evaluation = predictor.evaluate_model()
        
        # Forecast future sales
        future_sales = predictor.forecast_future(steps=forecast_days)
        
        # Format for template
        forecast_data = future_sales.reset_index().to_dict(orient='records')
        
        # Render template with results
        return render_template('index.html', 
                               evaluation=evaluation, 
                               forecast=forecast_data)
    
    except Exception as e:
        logger.error(f"Error in web prediction: {str(e)}")
        error_message = f"Error generating forecast: {str(e)}"
        return render_template('index.html', error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
