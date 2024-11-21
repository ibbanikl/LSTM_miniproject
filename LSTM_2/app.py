from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to create datasets for LSTM input
def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])  # Past 60 days as input
        y.append(data[i + time_step, 0])      # Next day as output
    return np.array(x), np.array(y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the ticker from the request
        data = request.get_json()
        ticker = data.get('ticker')

        # Download stock data
        stock_data = yf.download(ticker, start='2015-01-01', end='2023-01-01')

        # Get 'Close' price and reshape for LSTM
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        dates = stock_data.index  # Get the dates for x-axis

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Split data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # Create datasets for LSTM
        time_step = 60
        x_train, y_train = create_dataset(train_data, time_step)
        x_test, y_test = create_dataset(test_data, time_step)

        # Reshape for LSTM input
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compile and train the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=64, epochs=5)

        # Make predictions
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        # Inverse scaling to get actual stock prices
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Inverse transform y_train and y_test for evaluation
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_train_scaled = scaler.inverse_transform(y_train)
        y_test_scaled = scaler.inverse_transform(y_test)

        # Plotting the results
        plt.figure(figsize=(14, 8))

        # Add date labels for the x-axis
        all_dates = pd.to_datetime(dates)
        plt.plot(all_dates, scaler.inverse_transform(scaled_data), label='Actual Stock Price')

        # Compute the dates for train and test predictions
        train_dates = all_dates[time_step:len(train_predict) + time_step]
        test_dates = all_dates[len(train_data) + time_step:len(train_data) + time_step + len(test_predict)]

        plt.plot(train_dates, train_predict, label='Train Predict')
        plt.plot(test_dates, test_predict, label='Test Predict')

        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()

        # Save plot to a BytesIO object (in-memory file)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Return the plot as an image
        return send_file(img, mimetype='image/png', as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)

