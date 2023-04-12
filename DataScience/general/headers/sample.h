#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <memory>
#include "../../distributions/headers/random_distributions.h"

// list of time series differencing methods for stationarity transformation
enum DIFFERENCING
  {
   integer=1,
   logreturn=2,
   fractional=3,
   deltamean=4,
   original=5
  };

// data structure for sample histograms
// histo_x: right(=upper) border of given bar 
template<typename T>
struct Histogram_bar{
    T lower_boundary;
    T upper_boundary;
    int abs_count;
    double rel_count;
};

// histogram structure
template<typename T>
struct Histogram{
    T min;
    T max;
    T bar_width;
    std::vector<Histogram_bar<T>> bar;
    Histogram(int bars){
        for (int n=0;n<bars;n++){
            bar.push_back(Histogram_bar<T>());
        }
    }
};

// Sample class declaration
template<typename T>
class Sample{  
    public:
        double mean();
        double median();   
        double weighted_average(bool as_series=false);
        std::vector<int> ranking(bool ascending=true);
        std::vector<T> exponential_smoothing(bool as_series=false);
        double variance();
        double stddev();    
        unsigned int find(T value,int index_from=0,int index_to=__INT_MAX__);
        double Dickey_Fuller();
        void correlation();
        std::vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        std::vector<T> sort(bool ascending=true);
        std::vector<T> shuffle();
        std::vector<T> log_transform();
        double Engle_Granger();
        T polynomial_predict(T x);
        double polynomial_MSE();
        bool isGoodFit(double threshold=0.95);
        void polynomial_regression(int power=5);
        void linear_regression();
        T linear_predict(T x);
        double get_Pearson_R();
        double get_slope(); 
        double get_y_intercept();
        double get_r_squared();
        double get_z_score();
        double get_t_score();
        double get_Spearman_Rho();
        double get_x_mean();
        double get_y_mean();
        double get_covariance();
        Histogram<T> histogram(uint bars);
        // delete non-parametric default constructor (a sample without data would be useless!)
        Sample() = delete;
        // parametric constructor for single std::vector-type sample
        // note: T must be a numeric type!
        Sample(const std::vector<T>& data_vect){
            this->data_vect = data_vect;
            elements = data_vect->size();
        }
        // parametric constructor for two std::vector-type samples
        // note: T must be a numeric type!
        Sample(const std::vector<T>& x_vect, const std::vector<T>& y_vect){
            this->x_vect = x_vect;
            this->y_vect = y_vect;
            elements = fmin(x_vect->size(),y_vect->size());
        }       
        // parametric constructor for single array-type sample
        // - T must be a numeric type!
        // - array must be 1-dimensional!
        Sample(const T* data_array){
            // copy as std::vector
            elements = sizeof(data_array)/sizeof(T);
            this->data_vect->reserve(elements);
            for (int i=0;i<elements;i++){
                this->data_vect.push_back(data_array[i]);
            }
        }
        // parametric constructor for two array-type samples
        // - T must be a numeric type!
        // - arrays must be 1-dimensional!
        Sample(T* x_array, T* y_array){
            // copy as std::vector
            elements = std::fmin(sizeof(x_array)/sizeof(T),sizeof(y_array)/sizeof(T));
            this->x_vect->reserve(elements);
            this->y_vect->reserve(elements);
            for (int i=0;i<elements;i++){
                this->x_vect.push_back(x_array[i]);
                this->y_vect.push_back(y_array[i]);
            }
        }            
        // destructor
        ~Sample(){};

    protected:

    private:
        std::vector<T>* x_vect;
        std::vector<T>* y_vect;
        std::vector<T>* data_vect;
        std::vector<double> coefficients; //for polynomial regression
        std::vector<double> y_regression;
        std::vector<double> residuals;        
        int elements;
        double Pearson_R;
        double slope;
        double y_intercept;
        double r_squared;
        double z_score;
        double t_score;
        double Spearman_Rho; 
        double x_mean;
        double y_mean;   
        double covariance;
        double standard_deviation;
        double RSS;        
        bool lin_reg_completed=false;
        bool poly_reg_completed=false;
        bool correlation_completed=false;        
};

// include .cpp resource (required due to this being a template class)
// -> thus mitigating 'undefined reference' compiler errors
#include "../sources/sample.cpp"





/*
Possible improvements:

Add input validation to the constructors to ensure that the sample size is greater than zero and that the input data vector is not empty.

Use templates to allow for different types of data vectors, such as std::array and std::valarray.

Use range-based for loops and iterators instead of index-based loops to make the code more readable.

Use the <numeric> header to compute the sum of squared deviations from the mean in the variance function.

Add a function to compute the mean absolute deviation (MAD) of a sample.

Add a function to compute the autocorrelation function (ACF) of a sample.

Add a function to compute the partial autocorrelation function (PACF) of a sample.

Add a function to perform a Box-Jenkins ARIMA model identification and estimation on a sample.

Add a function to perform a Granger causality test on two or more samples.

Add a function to perform a Kolmogorov-Smirnov test for goodness-of-fit on a sample.
*/