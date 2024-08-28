# Stock Price Prediction Using LSTM

This project involves developing a stock price prediction model using Long Short-Term Memory (LSTM) networks in Python. The goal is to predict the future prices of selected technology stocks based on their historical data.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Skills and Tools Used](#skills-and-tools-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project is focused on predicting stock prices for well-known tech companies like Apple, Google, Microsoft, and Amazon. The model leverages LSTM, a type of recurrent neural network (RNN) capable of learning from sequential data, making it well-suited for time series forecasting.

## Dataset
The dataset consists of historical stock prices for the following companies:
- Apple (AAPL)
- Google (GOOG)
- Microsoft (MSFT)
- Amazon (AMZN)

The data was sourced from Yahoo Finance using the `yFinance` library.

## Installation
To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required packages using the following command:

```bash 
pip install -r requirements.txt
```
Alternatively, you can manually install the required libraries:
pip install pandas numpy matplotlib seaborn yfinance scikit-learn tensorflow keras

## Project Structure
Stock_Price_Prediction.ipynb: The Jupyter notebook containing the code for the entire project.
README.md: Project overview and instructions.
requirements.txt: A list of Python libraries required to run the project.

##Usage
1. Clone this repository to your local machine:
--> git clone https://github.com/yourusername/stock-price-prediction-lstm.git
   
2. Navigate to the project directory:
--> cd stock-price-prediction-lstm
   
3. Open the Jupyter notebook:
--> jupyter notebook Stock_Price_Prediction using lstm.ipynb
   
4. Run the notebook to see the results of the stock price predictions.

## Model Architecture
The LSTM model is built using the Keras library. The architecture includes:

1. Two LSTM layers with 128 and 64 units respectively.
2. Two Dense layers for the final prediction.
3. The model is trained using the Adam optimizer and mean squared error loss function.
   
## Results
The model was trained and tested on historical stock data, with predictions visualized against actual prices. 
The Root Mean Squared Error (RMSE) was used to evaluate the model's performance.


## Skills and Tools Used
--> Python Programming
--> Data Analysis and Visualization: Pandas, Matplotlib, Seaborn
--> Machine Learning: LSTM networks, Sequential model building, Data preprocessing
--> Libraries: Pandas, NumPy, yFinance, Scikit-learn, TensorFlow, Keras
--> Version Control: Git and GitHub
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

This `README.md` file provides a comprehensive overview of your project, including instructions on how to run it and details about the model and tools used. Make sure to customize the links and sections like "Contributing" and "License" as needed.
