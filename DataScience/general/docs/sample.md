## Sample Analysis     
usage: `#include <sample.h>` or include as part of `<datascience.h>`

description:
The class `Sample<T>` allows basic statistical analysis on
- 1d / single variable numeric data ("series" or "bag of numbers") or
- 2d / dual variable numberic data (vectors x+y, with equal number of elements)


### class `Sample<T>` (alias: `Sample1d<T>`): for 1d samples (vector x)
    - linear regression
        - slope + y-axis intercept
        - r squared + goodness of fit
        - prediction
    - polynomial regression
        - r squared + goodness of fit
        - mean squared error (MSE)
        - prediction
    - stationary transformation
    - Dickey-Fuller test (for stationarity)
    - logarithmic transformation
    - exponential smoothing
    - mean, median, weighted average
    - variance, standard deviation
    - histogram
    - ranking
    - find (=number of occurences of a given value)
    - shuffle
    - sort

### class `Sample2d<T>`: for two 2d samples (vectors x+y)
    - correlation
        - Pearson
        - Spearman
        - Student's t-score
        - z-score
    - covariance
    - Engle-Granger test (for cointegration)
    - x_mean, y_mean