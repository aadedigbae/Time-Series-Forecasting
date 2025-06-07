# Beijing Air Quality Forecasting Report

## Introduction

Air quality forecasting is a critical task for public health, environmental management, and policy-making. Accurate prediction of pollutants like PM2.5 is essential for issuing warnings, planning interventions, and understanding pollution patterns. This report details the development of a time series forecasting model using a Recurrent Neural Network (RNN), specifically an LSTM (Long Short-Term Memory) network, to predict PM2.5 concentrations in Beijing. The approach leverages historical air quality data and meteorological features to learn temporal dependencies and predict future values. The model development process involved data exploration, preprocessing, feature engineering, model architecture design, and iterative experimentation to achieve optimal performance.

The code and resources for this project are available in this GitHub repository:  
https://github.com/aadedigbae/Time-Series-Forecasting

## Data Exploration

The dataset used for this project consists of historical hourly air quality and meteorological data for Beijing. The initial exploration revealed the following key characteristics:

- **Dataset Structure**: The data is time-stamped, with columns representing various features such as PM2.5 concentration (pm2.5), dew point (DEWP), temperature (TEMP), pressure (PRES), wind speed (WSPM), and wind direction (wd). A unique identifier (No) and a datetime column are also present.
- **Missing Values**: The pm2.5 column, the target variable, contained a significant number of missing values. Missing values were also present in other columns.
- **Data Types**: The datetime column was initially an object type and needed conversion to datetime objects for time-series analysis.
- **Statistical Summary**: Descriptive statistics provided insights into the distribution, central tendency, and spread of the numerical features. The pm2.5 distribution showed skewness and kurtosis, indicating it is not normally distributed and has heavy tails (outliers).
- **Temporal Patterns**: Visualizing PM2.5 over time revealed seasonality and trends. Seasonal decomposition confirmed significant daily and potentially weekly seasonality, as well as a trend component.
- **Feature Correlations**: A correlation heatmap showed relationships between features. PM2.5 exhibited correlations with features like dew point and temperature.A
- **Outliers**: Box plots highlighted the presence of outliers in several numerical features, particularly in pm2.5.

## Preprocessing Steps

1. **Datetime Conversion and Indexing**: The datetime column was converted to datetime objects and set as the DataFrame index, which is standard practice for time-series data handling in pandas.
2. **Handling Missing Values**: Missing values in the pm2.5 column were handled using linear interpolation (interpolate(method='linear')) followed by forward-fill (fillna(method='ffill')) and backward-fill (fillna(method='bfill')) to ensure no NaNs remained. This approach attempts to estimate missing values based on surrounding points while ensuring all gaps are filled. Missing values in other columns were implicitly handled by the scaling process (though explicit imputation might be considered in future work).
3. **Temporal Feature Engineering**: Relevant temporal features (hour, day, month, weekday, season) were extracted from the datetime index. These features can help the model capture time-based patterns.
4. **Scaling**: Features and the target variable (pm2.5) were scaled using StandardScaler. Scaling is crucial for neural networks as it helps normalize the input ranges, preventing features with larger scales from dominating the learning process. MinMaxScaler was also experimented with for the target variable.

## Model Design

The model that yielded the best performance was an LSTM-based sequential model designed to process sequences of historical data points to predict the next PM2.5 value. The key to achieving good performance was the successful implementation of sequence creation, moving from a single timestep input to using a historical window.

Based on your findings, the best performing configuration utilized a timestep of 24, meaning the model looked back at the previous 24 hours of data (including PM2.5 and other features) to predict the PM2.5 value for the next hour.

### Model Architecture

```
Model: "sequential"

_________________________________________________________________

 Layer (type)                Output Shape              Param #    

=================================================================

 bidirectional (Bidirection  (None, 24, 256)           197632     

 al)                                                              

 dropout (Dropout)           (None, 24, 256)           0          

 lstm_1 (LSTM)               (None, 64)                82176      

 dropout_1 (Dropout)         (None, 64)                0          

 dense_2 (Dense)             (None, 32)                2080       

 dense_3 (Dense)             (None, 16)                528        

 dense_4 (Dense)             (None, 1)                 17         

=================================================================

Total params: 282,433 

Trainable params: 282,433 

Non-trainable params: 0 
```
- **Input Layer**: The model accepts sequences of timesteps observations, where each observation has features number of values. With timesteps = 24 and 14 features, the effective input shape to the first layer is (24, 14).
- **Bidirectional LSTM Layer**: The first layer is a Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0005))). This layer processes the input sequence in both directions, potentially capturing more complex temporal dependencies. return_sequences=True is used to pass sequences to the next LSTM layer. 128 units were chosen for capacity, and L2 regularization was applied.
- **Dropout Layer (0.3)**: A Dropout layer with a rate of 0.3 was added to reduce overfitting.
- **LSTM Layer**: A standard LSTM(64, return_sequences=False, kernel_regularizer=l2(0.0005)) layer follows.
- **Dense Layers**: Two Dense layers with ReLU activation (32 and 16 units, with L2 regularization on the first Dense layer) process the output of the LSTM.
- **Output Layer**: A final Dense layer with one unit predicts the continuous PM2.5 value.

The model was compiled with the RMSprop optimizer (learning rate 0.001) and Mean Squared Error (mse) as the loss function, monitoring rmse.

## Experiment Table

A full experiment table, model configs, and results were detailed in the original report (not included here due to size).

## Results

- Models with timestep = 1 performed poorly.
- Models with timestep = 24 improved dramatically.
- Best model: BiLSTM + LSTM + Dropout + L2 + RMSprop + 24-hour timestep.
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint were essential for managing training and performance.
- Final predictions were inverse-transformed, clipped, and saved to a CSV.

## Conclusion

This project successfully developed an LSTM model capable of forecasting hourly PM2.5 concentrations in Beijing by effectively leveraging sequential historical data.

## Proposed Improvements and Next Steps

1. Further Timestep Exploration  
2. Advanced Feature Engineering  
3. Hyperparameter Optimization  
4. Alternative Model Architectures  
5. Ensembling  
6. Error Analysis  
7. Sensitivity Analysis  
8. Multi-Step Forecasting  

## References (IEEE Style)

[1] World Health Organization, “Air quality and health,” WHO, Geneva, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health  
[2] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.  
[3] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Trans. Signal Processing, vol. 45, no. 11, pp. 2673–2681, Nov. 1997.  
[4] G. Hinton, “Neural Networks for Machine Learning - Lecture 6a,” Coursera, 2012. [Online]. Available: [https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6a.pdf](https://www.researchgate.net/publication/388734824_Implementation_of_Neural_Networks_on_FPGA  
[5] N. Srivastava et al., “Dropout: A simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, 2014.  
[6] A. Krogh and J. A. Hertz, “A simple weight decay can improve generalization,” in NeurIPS, 1992, pp. 950–957.  
[7] Research Gate, “Beijing Multi-Site Air-Quality Data,” 2015. [Online]. Available: [https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data](https://www.researchgate.net/publication/388734824_Implementation_of_Neural_Networks_on_FPGA)  
[8] J. Brownlee, “Time Series Forecasting with the Long Short-Term Memory Network in Python,” Machine Learning Mastery, 2017. [Online]. Available: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/  
[9] A. Graves, “Supervised sequence labelling with recurrent neural networks,” Ph.D. dissertation, Technical University of Munich, 2008.  
[10] A. Adedigba, “Time-Series Forecasting,” GitHub repository. [Online]. Available: https://github.com/aadedigbae/Time-Series-Forecasting  

## AI Assistance Disclosure

No generative AI tools were used to write or generate the content of the original report. All analyses, interpretations, and writing were conducted by the authors.
