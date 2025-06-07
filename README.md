# Beijing Air Quality Forecasting using LSTM

## Introduction

Air quality forecasting is a critical task for public health, environmental management, and policy-making. Accurate prediction of pollutants like PM2.5 is essential for issuing warnings, planning interventions, and understanding pollution patterns [1]. This project develops a time series forecasting model using a Recurrent Neural Network (RNN), specifically a Long Short-Term Memory (LSTM) network [2], to predict PM2.5 concentrations in Beijing. The model leverages historical air quality and meteorological data to learn temporal dependencies and forecast future values.  

The complete code and resources are available in the [GitHub repository](https://github.com/aadedigbae/Time-Series-Forecasting) [10].

---

## Data Exploration

The dataset used consists of historical hourly air quality and meteorological data for Beijing [7]. Key characteristics:

- **Features**: Includes PM2.5, dew point (DEWP), temperature (TEMP), pressure (PRES), wind speed (WSPM), wind direction (wd), and a timestamp.
- **Missing Values**: Significant gaps in PM2.5 values were handled via interpolation.
- **Data Types**: The datetime column was converted for time series indexing.
- **Temporal Patterns**: Seasonality and trends were detected.
- **Feature Correlations**: PM2.5 correlated with meteorological variables.
- **Outliers**: Present in several numerical features, especially PM2.5.

---

## Preprocessing Steps

1. **Datetime Indexing**: Converted the datetime column and set it as the DataFrame index.
2. **Missing Values**: Handled using `interpolate()`, forward fill, and backward fill methods.
3. **Temporal Features**: Extracted hour, day, month, weekday, and season [8].
4. **Scaling**: Applied `StandardScaler` and experimented with `MinMaxScaler`.

---

## Model Design

The optimal model was an LSTM-based sequential architecture using a timestep of 24 (i.e., the past 24 hours of data) [2][8]. The architecture:

- **Bidirectional LSTM Layer**: Captures patterns from both past and future in the sequence [3].
- **Dropout (0.3)**: Reduces overfitting [5].
- **LSTM Layer (64 units)**: Encodes temporal structure [2].
- **Dense Layers**: Fully connected layers with ReLU activation.
- **Output Layer**: Predicts a continuous PM2.5 value.
- **Regularization**: Applied L2 weight decay [6].

Compiled with RMSprop optimizer [4], learning rate 0.001, and `mse` loss.

---

## Experimentation

Multiple experiments were conducted, varying timesteps, model depth, dropout, L2 regularization, and optimizer. Key findings:

- **Timestep = 24** dramatically improved performance over single-step models.
- **Bidirectional LSTM** and **Dropout + L2** improved generalization.
- **RMSprop** with a batch size of 128 and early stopping yielded the best results.

---

## Results

- **Training vs Validation Loss**: Showed good convergence using early stopping and learning rate reduction.
- **Best Configuration**: Timestep 24, Bidirectional + LSTM, Dropout 0.3, L2=0.0005, RMSprop optimizer, batch size 128.
- **Forecasting**: Final predictions were inverse-transformed, clipped, and submitted in CSV format.

---

## Future Improvements

1. Tune timesteps further (e.g., 20, 28, 36).
2. Add lag features, rolling stats, and interaction terms.
3. Use hyperparameter tuning frameworks (Optuna, KerasTuner).
4. Explore GRUs [9] or TCNs.
5. Ensemble multiple models.
6. Conduct error and sensitivity analyses.
7. Extend to multi-step forecasting.

---

## AI Assistance Disclosure

No generative AI tools were used to write or generate the content of this report. All analyses, interpretations, and writing were conducted by the authors.

---

## References

```text
[1]    World Health Organization, “Air quality and health,” WHO, Geneva, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health

[2]    S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[3]    M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Trans. Signal Processing, vol. 45, no. 11, pp. 2673–2681, Nov. 1997.

[4]    G. Hinton, “Neural Networks for Machine Learning - Lecture 6a,” Coursera, 2012. [Online]. Available: https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6a.pdf

[5]    N. Srivastava et al., “Dropout: A simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, 2014.

[6]    A. Krogh and J. A. Hertz, “A simple weight decay can improve generalization,” in Advances in Neural Information Processing Systems (NeurIPS), 1992, pp. 950–957.

[7]    UCI Machine Learning Repository, “Beijing Multi-Site Air-Quality Data,” 2017. [Online]. Available: https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

[8]    J. Brownlee, “Time Series Forecasting with the Long Short-Term Memory Network in Python,” Machine Learning Mastery, 2017. [Online]. Available: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

[9]    A. Graves, “Supervised sequence labelling with recurrent neural networks,” Ph.D. dissertation, Technical University of Munich, 2008.

[10]   A. Dedigba, “Time-Series Forecasting,” GitHub repository. [Online]. Available: https://github.com/aadedigbae/Time-Series-Forecasting
