#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <memory>
#include "array.h"
#include "../../distributions/headers/random_distributions.h"

// forward declarations
template<typename T> class Vector;

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
        int* ranking(bool ascending=true);
        T* exponential_smoothing(bool as_series=false);
        double variance();
        double stddev();    
        unsigned int find(T value,int index_from=0,int index_to=__INT_MAX__);
        double Dickey_Fuller();
        T* stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        T* sort(bool ascending=true);
        T* shuffle();
        T* log_transform();
        Histogram<T> histogram(uint bars);
        // delete non-parametric default constructor (a sample without data would be useless!)
        Sample() = delete;

        // parametric constructor for single std::vector-type sample
        // note: T must be a numeric type!
        Sample(const std::vector<T>& data){
            elements = data.size();
            this->data = T(elements);
            for (int n=0;n<elements;n++){
                this->data[n] = data[n];
            }
        }   

        // parametric constructor for custom Vector type (as part of array.h)
        Sample(const Vector<T>& data){
            elements = data.get_elements();
            this->data = T(elements);
            this->data=data._data;           
        }

        // parametric constructor for single array-type sample
        // - T must be a numeric type!
        // - array must be 1-dimensional!
        Sample(const T& data){
            elements = sizeof(data)/sizeof(T);  
            this->data=T(elements);
            this->data=data;        
        }
         
        // destructor
        ~Sample(){};

    protected:
        T* data;      
        int elements;
        double standard_deviation;        
};

// 1d sample alias class
template<typename T>
class Sample1d : public Sample<T> {};

// class for 2d samples (vectors x+y)
template <typename T>
class Sample2d {
    public:
        T polynomial_predict(T x);
        double polynomial_MSE();
        bool isGoodFit(double threshold=0.95);
        void polynomial_regression(int power=5);
        void linear_regression();
        T linear_predict(T x);
        double get_slope(); 
        double get_y_intercept();
        double get_r_squared();    
        double Engle_Granger();
        void correlation();
        double get_Pearson_R();
        double get_Spearman_Rho();
        double get_z_score();
        double get_t_score();
        double get_x_mean();
        double get_y_mean();
        double get_covariance();       

        // parametric constructor for two std::vector-type samples
        // - T must be a numeric type!
        // - the vectors x and y should be of equal size!
        // - if the size is unequal, the surplus elements of the larger vector
        //   will be ignored
        Sample2d(const std::vector<T>& x_data, const std::vector<T>& y_data){
            this->elements = std::fmin(x_data.size(),y_data.size());
            // copy vectors into stack-allocated static arrays
            this->x_data=T(elements);
            this->y_data=T(elements);
            for (int n=0;n<elements;n++){
                this->x_data[n]=x_data[n];
                this->y_data[n]=y_data[n];
            }
            coefficients=double(elements);
            y_regression=double(elements);
            residuals=double(elements);             
        };    

        // parametric constructor for two vectors of
        // custom Vector type (as part of array.h)
        // - T must be a numeric type!
        // - the vectors x and y should be of equal size!
        // - if the size is unequal, the surplus elements of the larger vector
        //   will be ignored        
        Sample2d(const Vector<T>& x_data, const Vector<T>& y_data){
            this->elements = std::fmin(x_data.get_elements(), y_data.get_elements());
            this->x_data=T(elements);
            this->y_data=T(elements);            
            for (int n=0;n<elements;n++){
                this->x_data[n]=x_data.get(n);
                this->y_data[n]=y_data.get(n);
            }
            coefficients=double(elements);
            y_regression=double(elements);
            residuals=double(elements);             
        };

        // parametric constructor for two array-type samples
        // - T must be a numeric type!
        // - arrays must be 1-dimensional!
        // - the arrays x and y should be of equal size!
        // - if the size is unequal, the surplus elements of the larger array
        //   will be ignored        
        Sample2d(const T& x_data, const T& y_data){
            this->elements = std::fmin(sizeof(x_data)/sizeof(T), sizeof(y_data)/sizeof(T));
            this->x_data=T(elements);
            this->y_data=T(elements); 
            for (int n=0;n<this->elements;n++){
                this->x_data[n]=x_data[n];
                this->y_data[n]=y_data[n];
            }
            coefficients=double(elements);
            y_regression=double(elements);
            residuals=double(elements);             
        };
    private:
        int elements;
        T* x_data;
        T* y_data; 
        bool correlation_completed=false;
        double z_score;
        double t_score;
        double Spearman_Rho; 
        double x_mean;
        double y_mean;   
        double covariance;
        double Pearson_R;   
        double slope;
        double r_squared;
        double y_intercept;    
        double RSS;        
        bool lin_reg_completed=false;
        bool poly_reg_completed=false;     
        double* coefficients; //for polynomial regression
        double* y_regression;
        double* residuals;                                  
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