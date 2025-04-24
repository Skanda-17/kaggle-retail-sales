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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("retail_forecast.log"), logging.StreamHandler()]
)
logger = logging.getLogger("retail_sales_predictor")

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
            
            # Convert date column to datetime
           # Normalize column names to lowercase (if needed)
            self.data.columns = self.data.columns.str.strip().str.lower()  # This will convert 'Date' to 'date'

# Now convert the 'date' column to datetime
            self.data['date'] = pd.to_datetime(self.data['date'])

                
            logger.info(f"Data loaded successfully with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, store_id=1, item_id=1):
        """Preprocess the data for time series analysis for a specific store and item."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        try:
            logger.info(f"Preprocessing data for store {store_id} and item {item_id}")
            
            # Filter data for specific store and item
            filtered_data = self.data[(self.data['store'] == store_id) & (self.data['item'] == item_id)]
            
            # Set date as index
            filtered_data.set_index('date', inplace=True)
            
            # Sort by date
            filtered_data = filtered_data.sort_index()
            
            # Store the processed data
            self.processed_data = filtered_data
            
            logger.info(f"Data preprocessing completed - shape: {filtered_data.shape}")
            return filtered_data
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
            plt.plot(self.processed_data.index, self.processed_data['sales'])
            plt.title('Store-Item Sales Over Time')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.grid(True)
            plt.savefig('sales_data_visualization.png')
            plt.close()
            
            # Plot seasonal decomposition
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Resample to monthly if needed
            monthly_data = self.processed_data['sales'].resample('M').sum()
            
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
            
            # Train SARIMA model
            model = SARIMAX(
                self.train_data['sales'],
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
            
            # Generate predictions for test period
            start = len(self.train_data)
            end = len(self.train_data) + len(self.test_data) - 1
            
            predictions = self.model_fit.predict(start=start, end=end, dynamic=False)
            self.test_predictions = predictions
            
            # Calculate evaluation metrics
            rmse = sqrt(mean_squared_error(self.test_data['sales'], predictions))
            mape = np.mean(np.abs((self.test_data['sales'] - predictions) / self.test_data['sales'])) * 100
            
            logger.info(f"Model RMSE: {rmse}")
            logger.info(f"Model MAPE: {mape}%")
            
            # Visualize actual vs predicted
            plt.figure(figsize=(12, 6))
            plt.plot(self.test_data.index, self.test_data['sales'], label='Actual Sales')
            plt.plot(self.test_data.index, predictions, label='Predicted Sales', color='red')
            plt.title('Actual vs Predicted Sales')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.savefig('model_evaluation.png')
            plt.close()
            
            return {
                'rmse': rmse,
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
                'sales': forecast
            })
            forecast_df.set_index('date', inplace=True)
            
            self.forecast = forecast_df
            
            # Visualize forecast
            plt.figure(figsize=(12, 6))
            plt.plot(self.processed_data.index[-90:], self.processed_data['sales'][-90:], label='Historical Sales')
            plt.plot(forecast_df.index, forecast_df['sales'], label='Forecasted Sales', color='red')
            plt.title(f'Sales Forecast for Next {steps} Days')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.savefig('sales_forecast.png')
            plt.close()
            
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
            
    def analyze_product_categories(self):
        """Analyze sales patterns across different product categories."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        try:
            logger.info("Analyzing product categories")
            
            self.data.columns = self.data.columns.str.strip().str.lower()  # This will convert 'Store', 'Item', 'Sales' to lowercase

# Print the column names to verify
            print(self.data.columns)

# Now group by store and item and sum the sales
            store_item_sales = self.data.groupby(['store', 'item'])['sales'].sum().reset_index()

# Sort the store-item combinations by total sales and find the top 5
            top_combinations = store_item_sales.sort_values('sales', ascending=False).head(5)

# Display the top combinations
            print(top_combinations)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='store', y='sales', hue='item', data=top_combinations)
            plt.title('Top 5 Store-Item Combinations by Total Sales')
            plt.xlabel('Store ID')
            plt.ylabel('Total Sales')
            plt.savefig('top_store_item_combinations.png')
            plt.close()
            
            # Analyze sales by store
            store_sales = self.data.groupby('store')['sales'].sum().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            store_sales.plot(kind='bar')
            plt.title('Total Sales by Store')
            plt.xlabel('Store ID')
            plt.ylabel('Total Sales')
            plt.savefig('sales_by_store.png')
            plt.close()
            
            logger.info("Product category analysis completed")
            return store_item_sales
        except Exception as e:
            logger.error(f"Error analyzing product categories: {str(e)}")
            raise

def main():
    """Main function to run the retail sales prediction."""
    parser = argparse.ArgumentParser(description='Retail Sales Prediction')
    parser.add_argument('--data_path', type=str, default='train.csv', help='Path to the dataset CSV file')
    parser.add_argument('--store_id', type=int, default=1, help='Store ID to analyze')
    parser.add_argument('--item_id', type=int, default=1, help='Item ID to analyze')
    parser.add_argument('--test_days', type=int, default=90, help='Number of days for testing')
    parser.add_argument('--forecast_days', type=int, default=30, help='Number of days to forecast')
    
    args = parser.parse_args()
    
    logger.info("Starting retail sales prediction")
    logger.info(f"Parameters: Store ID={args.store_id}, Item ID={args.item_id}, Test days={args.test_days}, Forecast days={args.forecast_days}")
    
    # Initialize predictor
    predictor = RetailSalesPredictor(args.data_path)
    
    # Load data
    predictor.load_data()
    
    # Analyze overall patterns
    predictor.analyze_product_categories()
    
    # Preprocess data for specific store and item
    predictor.preprocess_data(store_id=args.store_id, item_id=args.item_id)
    
    # Visualize the data
    predictor.visualize_data()
    
    # Split data
    predictor.split_data(test_size=args.test_days)
    
    # Train model
    # For the Kaggle retail dataset, a SARIMA model with these parameters often works well
    predictor.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    
    # Evaluate model
    evaluation = predictor.evaluate_model()
    logger.info(f"Model evaluation: RMSE={evaluation['rmse']:.2f}, MAPE={evaluation['mape']:.2f}%")
    
    # Forecast future
    future_sales = predictor.forecast_future(steps=args.forecast_days)
    
    # Save model
    predictor.save_model()
    
    logger.info(f"Forecasted sales for next {args.forecast_days} days:")
    logger.info(future_sales.head())
    
    logger.info("Retail sales prediction completed successfully")

if __name__ == "__main__":
    main()
