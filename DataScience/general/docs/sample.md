[[return to main page]](../../../README.md)
# Sample Analysis     
usage: `#include <sample.h>` or include as part of `<datascience.h>`

description:
The library allows basic statistical analysis on
- 1d numeric data ("series" or "bag of numbers"): `class Sample<T>`, alias `class Sample1d<T>`
- 2d numberic data (vectors x+y, with equal number of elements): `class Sample2d<T>`

## class `Sample<T>` (alias: `Sample1d<T>`)

Constructor:
- `Sample<T>::Sample(const T* data)` or
- `Sample<T>::Sample(const std::vector<T>& data)`

  note that the first method has less overhead, because the second will
  create an internal copy of the vector as a stack-allocated static array,
  whilst the first method takes an array as it is 'by reference',
  without making a copy

Public Methods:
|method | description |
|-------|-------------|
| `Sample<T>::mean()` | returns the arrithmetic mean as type double|
| `Sample<T>::median()`| returns the median as type double |
| `Sample<T>::weighted_average(bool as_series=false)` | returns the weighted average as type double |
| `Sample<T>::ranking(bool ascending=true)` | returns a pointer to an integer array that holds the ranks corresponding to the values of the sample |
| `Sample<T>::exponential_smoothing(bool as_series=false)` | returns a pointer to an array of type `<T>` that holds the exponential smoothing of the sample |
| `Sample<T>::variance()` | returns the variance of the sample as a double value |
| `Sample<T>::stddev()` | returns the standard deviation of the sample as type double |    
| `Sample<T>::find(T value,int index_from,int index_to)` | returns the number of occurrences of the given value |
| `Sample<T>::Dickey_Fuller()` | performs a Dickey-Fuller test as a unit root test for stationarity and returns a p-value |
| `Sample<T>::stationary(DIFFERENCING method, double degree, double fract_exponent)` | returns a stationary transformation as a pointer to an array of type `<T>` |
| `Sample<T>::sort(bool ascending=true)` | returns a pointer to an array of type `<T>`as a sorted version of the sample |
| `Sample<T>::shuffle()` | returns a pointer to an array of type `<T>`as a shuffled version of the sample |
| `Sample<T>::log_transform()` | returns a pointer to an array of type `<T>`as a log transformation of the sample |
| `Sample<T>::polynomial_predict(T x)` | predicts a y value from a new x value based on polynomial regression (run regression first!) |
| `Sample<T>::polynomial_MSE()` | returns the Mean Squared Error from polynomial regression (run the latter first!) |
| `Sample<T>::isGoodFit(double threshold=0.95)` | returns true if r_squared of regression is within the specified confidence interval; run regression first! |
| `Sample<T>::polynomial_regression(int power=5)` | performs polynomial regression to the specified power; no return value |
| `Sample<T>::linear_regression()` | performs linear regression; no return value |
| `Sample<T>::linear_predict(T x)` | predicts a new y value for a new x value, based on linear regression |
| `Sample<T>::get_slope()` | returns the slope from linear regression as type `<double>` |
| `Sample<T>::y_intercept()` | returns the y axis intercept from linear regression as type `<double>` |
| `Sample<T>::get_r_squared()` | returns the r_squared value from linear or polynomial regression (run regression first!) |
| `Sample<T>::histogram(uint bars)` | returns a `struct Histogram<T>` object that holds the data of a histogram of the given sample with the specified number of bars |

## class `Sample2d<T>`
Constructor:
- `Sample2d<T>::Sample2d(const T* x_data, const T* y_data)` or
- `Sample2d<T>::Sample2d(const std::vector<T>& x_data, const std::vector<T>& y_data)`

  note that the first method has less overhead, because the second will
  create an internal copy of the vector as a stack-allocated static array,
  whilst the first method takes an array as it is 'by reference',
  without making a copy

Methods:
|method | description |
|-------|-------------|
| `Sample2d::Engle_Granger()` | performs an Engle-Granger test for cointegration|
| `Sample2d::correlation()` | performs a correlation analysis of x and y |
| `Sample2d::get_Pearson_R()` | returns Pearson's R value (coefficient of correlation) of x and y |
| `Sample2d::get_Spearman_Rho()` | returns Spearman's Rho value (rank correlation coefficient) of x and y|
| `Sample2d::get_z_score()` | returns the z-score |
| `Sample2d::get_t_score()` | return Students t-score |
| `Sample2d::get_x_mean()` | returns the arrithmetic mean of the x values |
| `Sample2d::get_y_mean()` | returns the arrithmetic mean of the y values |
| `Sample2d::get_covariance()` | returns the corvariance |
| `Sample2d::get_slope()` | returns the slope of an assumed linear correlation |
| `Sample2d::get_y_intercept()` | returns the y-axis intercept of an assumed linear correlation |
| `Sample2d::get_r_squared()` | returns the r_squared (=coefficient of determination) of correlation | 

[[return to main page]](../../../README.md)