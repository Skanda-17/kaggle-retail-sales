from flask import Flask, request, jsonify
from retail_sales_kaggle import RetailSalesPredictor

app = Flask(__name__)

@app.route('/')
def home():
    return "Retail Sales Forecasting API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        data_path = data.get("data_path", "train.csv")
        store_id = data.get("store_id", 1)
        item_id = data.get("item_id", 1)
        test_days = data.get("test_days", 90)
        forecast_days = data.get("forecast_days", 30)

        predictor = RetailSalesPredictor(data_path)
        predictor.load_data()
        predictor.analyze_product_categories()
        predictor.preprocess_data(store_id=store_id, item_id=item_id)
        predictor.split_data(test_size=test_days)
        predictor.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        evaluation = predictor.evaluate_model()
        forecast = predictor.forecast_future(steps=forecast_days)
        
        forecast_json = forecast.reset_index().to_dict(orient='records')
        return jsonify({
            "rmse": evaluation["rmse"],
            "mape": evaluation["mape"],
            "forecast": forecast_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
