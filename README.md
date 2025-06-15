Stock Price Prediction using LSTM/GRU

This project builds a deep learning model to predict stock prices using historical stock data (specifically Google/Alphabet stock data). It involves data preprocessing, exploratory data analysis (EDA), time series generation, and training LSTM/GRU neural networks.

Project Structure

##  Features & Techniques

- **Data Source**: Historical stock prices (Google - GOOG)
- **Libraries Used**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`, `keras`, `tensorflow`, `sklearn`
- **EDA**:
  - Line plots of all price attributes (`open`, `close`, `high`, `low`, etc.)
  - Comparative analysis between 2016 and 2021 monthly trends
- **Preprocessing**:
  - Drop unused columns (`symbol`, `volume`, `splitFactor`, etc.)
  - Date parsing and setting datetime index
  - Normalization using MinMaxScaler
- **Modeling**:
  - LSTM and GRU deep learning models
  - Early stopping to prevent overfitting
  - Evaluation using MAE, MSE, RMSE, R², and Explained Variance

---

##  Models Used

- **LSTM (Long Short-Term Memory)**:
  - Multiple layers with dropout regularization
  - Trained on historical price sequences
- **GRU (Gated Recurrent Unit)**:
  - Alternative to LSTM with fewer parameters

---

##  Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Explained Variance Score

---
