# Stock Price Predictor

## Overview

This project utilizes PyTorch to implement a deep learning model for predicting stock prices. The code includes all the necessary preprocessing steps, defining the neural network architecture, training the model, evaluating it, and making predictions.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- Transformers
- Matplotlib

## Installation

To install the required dependencies, run:

\`\`\`bash
pip install pandas numpy scikit-learn torch transformers matplotlib
\`\`\`

## Usage

The code can be run directly by executing the script. It performs the following tasks:

1. **Data Loading**: Reads the stock price data from `AAPL.csv`.
2. **Preprocessing**: Converts dates to datetime objects and scales the features.
3. **Data Splitting**: Splits the data into training and testing sets.
4. **Model Definition**: Defines a simple feed-forward neural network with four hidden layers.
5. **Training**: Trains the model using Mean Squared Error (MSE) loss and the Adam optimizer.
6. **Evaluation**: Evaluates the model on the test data and visualizes the actual vs. predicted prices.
7. **Prediction**: Makes a prediction for the next day's stock price.

## Code Structure

### Importing Libraries

The essential libraries and modules are imported at the beginning.

### Data Preprocessing

- `pd.read_csv('AAPL.csv')`: Reads the stock price data.
- `pd.to_datetime(df['date'])`: Converts the 'date' column to datetime objects.
- `MinMaxScaler()`: Scales the features using Min-Max scaling.

### Splitting Data

- `train_test_split()`: Splits the data into training and testing sets.

### Model Definition

The `StockPredictor` class defines the architecture of the neural network, consisting of four hidden layers and ReLU activation functions.

### Training

The model is trained for 200 epochs using the Adam optimizer.

### Evaluation

- `plt.plot()`: Plots the actual and predicted stock prices.
- `plt.show()`: Displays the plot.

### Prediction

- `scaler.transform()`: Scales the last row of the data.
- `model(input_tensor)`: Makes a prediction using the trained model.


