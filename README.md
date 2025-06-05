# Beijing Air Quality Forecasting Project

This repository contains code and a report for forecasting Beijing's PM2.5 air quality using time series analysis and deep learning.

## Table of Contents
- [Introduction](#introduction)
- [Data Exploration, Preprocessing, & Feature Engineering](#data-exploration-preprocessing--feature-engineering)
- [Model Design & Architecture](#model-design--architecture)
- [Experiment Table](#experiment-table)
- [Results & Discussion](#results--discussion)
- [Code Quality & GitHub Submission](#code-quality--github-submission)
- [Conclusion](#conclusion)
- [Report Citations & Originality](#report-citations--originality)

## Introduction

This project aims to forecast Beijing's PM2.5 air quality using time series forecasting techniques, specifically leveraging a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. Air quality forecasting is crucial for public health and environmental management, and time series models are well-suited for capturing the temporal dependencies inherent in air quality data. The approach involves loading historical air quality data, performing exploratory data analysis, preprocessing the data, building and training an LSTM model, and evaluating its performance.

## Data Exploration, Preprocessing, & Feature Engineering

The dataset contains hourly air quality and meteorological data for Beijing. Initial exploration involved inspecting the first few rows, examining column names, and reviewing descriptive statistics using `df.head()`, `df.columns`, and `df.describe()`. These steps provided an initial understanding of the data structure, variable types, and the range and distribution of numerical features.

The `datetime` column was converted to a datetime object and set as the index, which is essential for time series analysis.

Missing values were identified using `train.isnull().sum()`, revealing missing data primarily in the `pm2.5` column.

Skewness and Kurtosis were calculated for the `pm2.5` column to understand its distribution shape, indicating a skewed distribution with heavy tails, suggesting the presence of outliers or extreme values.

Percentile analysis using `train['pm2.5'].quantile()` further supported the presence of outliers, showing a significant difference between the 75th percentile and the 99th percentile.

Visualizations were used to gain deeper insights:
- A **Correlation Heatmap** (`sns.heatmap`) was generated to visualize the relationships between different features. This helps identify features that are strongly correlated with `pm2.5` and potentially with each other. The heatmap showed correlations between meteorological features and `pm2.5`.
- **Box Plots** (`sns.boxplot`) were plotted for numerical columns to visually detect outliers. This confirmed the presence of outliers in several features, including `pm2.5`, `Iws`, `Is`, and `Ir`.
- A **Line Plot of PM2.5 over time** (`train.plot(y='pm2.5')`) showed the temporal patterns and trends in PM2.5 concentration. This visualization clearly depicted the seasonality and irregular spikes in air pollution.
- A **Missing Value Heatmap** (`msno.matrix`) provided a visual representation of the location and extent of missing values across the dataset.

Missing values in the `pm2.5` column were handled by first converting the column to numeric with `errors='coerce'` to handle any non-numeric entries gracefully. Then, linear interpolation (`interpolate(method='linear')`) was applied to fill missing values based on surrounding data points, followed by forward-fill (`fillna(method='ffill')`) and backward-fill (`fillna(method='bfill')`) to handle any remaining NaNs at the beginning or end of the series.

Seasonal decomposition (`seasonal_decompose`) was performed on the `pm2.5` data to separate the time series into its trend, seasonal, and residual components. This helps in understanding the underlying patterns and seasonality of PM2.5 levels.

Temporal feature engineering was performed by extracting `hour`, `day`, `month`, `weekday`, and `season` from the datetime index. These features can capture daily, weekly, monthly, and seasonal patterns in air quality. Plotting the average PM2.5 by hour of day revealed a clear diurnal pattern.

Features were separated from the target variable (`pm2.5`). The `No` column was dropped as it is an identifier and not a predictive feature.

Normalization was applied using `StandardScaler` to scale both the features (`X_train`) and the target (`y_train`). Scaling is important for neural networks as it helps in faster convergence and prevents features with larger scales from dominating the learning process.

Time series sequences were created using the `create_sequences` function. This function generates input sequences (past observations) and corresponding target values (the next observation) based on a defined `timesteps` window. A `timesteps` of 24 was chosen, meaning the model will use the previous 24 hours of data to predict the next hour's PM2.5 level.

The sequenced data was then split into training and validation sets using `train_test_split` with `shuffle=False` to maintain the temporal order of the data.

## Model Design & Architecture

The model used for this time series forecasting task is a Sequential model built with Keras, incorporating LSTM layers. LSTMs are particularly well-suited for sequence prediction problems like time series forecasting because they can learn long-term dependencies in the data, mitigating the vanishing gradient problem often encountered in traditional RNNs.

The current architecture is a relatively simple one, serving as a baseline for experimentation:

- **Input Layer:** The model expects input sequences with a shape corresponding to `(timesteps, number_of_features)`. In this case, with `timesteps = 24` and 14 features after feature engineering, the input shape is `(24, 14)`.
- **LSTM Layer:** A single LSTM layer with 64 units and `tanh` activation is used. The `return_sequences=False` argument means this layer only returns the output for the last timestep of the sequence, which is appropriate for a many-to-one prediction task (predicting the next single PM2.5 value). The `tanh` activation is commonly used in LSTM gates.
- **Dense Layers:**
    - A Dense layer with 32 units and `relu` activation follows the LSTM layer. This layer helps in further processing the features extracted by the LSTM.
    - A final Dense layer with 1 unit and no activation function (linear activation) is used for the output, as PM2.5 is a continuous value.

The model is compiled with the `Adam` optimizer, which is an adaptive learning rate optimization algorithm known for its efficiency. The loss function is `mse` (Mean Squared Error), a standard metric for regression tasks, and `rmse` (Root Mean Squared Error) is included as a metric for monitoring during training.

## Experiment Table

To find the best performing model, a systematic approach to experimentation is crucial.

| Experiment No. | Learning Rate | Batch Size | Optimizer | LSTM Layers | LSTM Units | Dense Layers | Dense Units | Activation Functions | Dropout | Regularization | Timesteps | Validation RMSE | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (Baseline) | 0.001 | 64 | Adam | 1 | 64 | 2 | 32, 1 | tanh (LSTM), relu (Dense) | None | None | 24 | 0.9032 | Initial model |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 15+ |  |  |  |  |  |  |  |  |  |  |  |  |  |

*Fill in the table with the parameters and resulting validation RMSE for each experiment.*

Discuss how variations in these parameters impacted the model's performance and how you arrived at the best configuration. Note any observations regarding overfitting or underfitting during these experiments.

## Results & Discussion

The performance of the model was evaluated using the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics.

**Root Mean Squared Error (RMSE):** RMSE is a widely used metric to measure the difference between predicted values and actual values. It represents the square root of the average of the squared differences between the predictions and the actual observations. A lower RMSE indicates a better fit of the model to the data.

The formula for RMSE is:

$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} $

Where:
- $N$ is the number of observations.
- $y_i$ is the actual value for the i-th observation.
- $\hat{y}_i$ is the predicted value for the i-th observation.

During training, the model's performance was monitored on both the training and validation sets. The training history plot shows the loss (MSE on scaled data) decreasing over epochs for both sets. The validation loss serves as an indicator of how well the model generalizes to unseen data.

The final training loss (MSE on Original Scale) was calculated to be approximately **2319.57**, and the final training loss (MSE on Scaled Data) was approximately **0.277**. The final validation RMSE (on scaled data), based on the best performing epoch (epoch 7 in this case, as indicated by Early Stopping), was approximately **0.9032**.

The difference between the training and validation loss curves can provide insights into overfitting. If the training loss continues to decrease while the validation loss increases or plateaus, it suggests that the model is overfitting to the training data. In this particular run, the validation loss fluctuated but generally remained higher than the training loss, indicating some potential for overfitting, which was partially addressed by Early Stopping.

Further experimentation with different model architectures, hyperparameters, and regularization techniques (as outlined in the Experiment Table section) would be necessary to systematically improve the model's performance and potentially reduce the gap between training and validation loss. Techniques to mitigate vanishing and exploding gradients in RNNs/LSTMs include using appropriate activation functions (like `tanh` or `relu`), employing gradient clipping, and utilizing architectures like LSTMs or GRUs that are specifically designed to handle these issues. In this model, the use of an LSTM layer helps in addressing the vanishing gradient problem.

Visualizing the model's predictions against the actual values on a portion of the validation or test set would provide a clearer picture of its performance and where it makes errors.

## Code Quality

The code in this notebook is organized into logical sections for data loading, exploration, preprocessing, model building, training, and prediction. Comments are included to explain key steps.

## Conclusion

In this project, I tackled the task of forecasting Beijing's PM2.5 air quality using an LSTM model. We performed comprehensive data exploration to understand the dataset's characteristics, including missing values, distributions, and temporal patterns. The data was preprocessed by handling missing values through interpolation and imputation, engineering temporal features, and scaling the data.

A baseline LSTM model was built and trained, demonstrating the feasibility of using LSTMs for this forecasting problem. The initial results, as indicated by the validation RMSE, show room for improvement.

Future work should focus on extensive experimentation with different model architectures and hyperparameters as outlined in the Experiment Table section. Potential improvements include:

- **Exploring more complex LSTM or GRU architectures:** Experiment with stacked LSTM layers, Bidirectional LSTMs, or GRU networks.
- **Hyperparameter tuning:** Systematically tune the learning rate, batch size, number of units, dropout rates, and regularization parameters.
- **Feature Engineering:** Explore additional relevant features, such as lagged values of PM2.5 or other meteorological variables, or interaction terms.
- **Ensemble Methods:** Combine predictions from multiple models to potentially improve accuracy.
- **Advanced time series techniques:** Investigate other time series forecasting methods like Prophet, ARIMA, or Transformer networks for comparison.

Addressing the challenges of time series forecasting, such as capturing long-term dependencies and handling non-stationarity, remains a key focus for achieving better accuracy.

## Report Citations & Originality

This report is based on the analysis and code developed for the Beijing Air Quality Forecasting project. While standard libraries and common machine learning techniques were used, the specific implementation and analysis presented are original to this project.
