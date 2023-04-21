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
        T get(const int index);
        set (const int index, T value);
        double mean();
        double median();   
        double weighted_average(bool as_series=false);
        std::unique_ptr<Sample<int>> ranking(bool ascending=true);
        std::unique_ptr<Sample<T>> exponential_smoothing(bool as_series=false);
        double variance();
        double stddev();    
        unsigned int find(T value,int index_from=0,int index_to=__INT_MAX__);
        double Dickey_Fuller();
        std::unique_ptr<Sample<T>> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        std::unique_ptr<Sample<T>> sort(bool ascending=true);
        std::unique_ptr<Sample<T>> shuffle();
        std::unique_ptr<Sample<T>> log_transform();
        std::unique_ptr<Histogram<T>> histogram(uint bars);
        // delete non-parametric default constructor (a sample without data would be useless!)
        Sample() = delete;

        // parametric constructor for single std::vector-type sample
        // note: T must be a numeric type!
        Sample(const std::vector<T>& data){
            this->_elements = data.size();
            this->_data = new T(this->_elements);
            for (int n=0;n<this->_elements;n++){
                this->_data[n] = data[n];
            }
        }   

        // parametric constructor for custom Vector type (as part of array.h)
        Sample(const Vector<T>& data){
            this->_elements = data.get_elements();
            this->_data = new T(this->_elements);
            this->_data=data._data;           
        }

        // parametric constructor for single array-type sample
        // - T must be a numeric type!
        // - array must be 1-dimensional!
        Sample(const T (&data)[]){
            this->_elements = sizeof(data)/sizeof(T);  
            this->_data = new T(this->_elements);
            this->_data=data;        
        }

        // parametric onstructor for an un-initialized sample of n elements
        Sample(const int n, T init_value=0){
            this->_elements = n;
            this->_data = new T(n);
            std::fill(this->_data,init_value);
        }
         
        // destructor
        ~Sample(){
            delete[] data;
        };

    protected:
        T* _data;      
        int _elements;
        double _standard_deviation;        
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
            this->_elements = std::fmin(x_data.size(),y_data.size());
            // copy vectors into stack-allocated static arrays
            this->_x_data=new T(this->_elements);
            this->_y_data=new T(this->_elements);
            for (int n=0;n<this->_elements;n++){
                this->_x_data[n]=x_data[n];
                this->_y_data[n]=y_data[n];
            }
            _coefficients=new double(this->_elements);
            _y_regression=new double(this->elements);
            _residuals=new double(this->_elements);            
        };    

        // parametric constructor for two vectors of
        // custom Vector type (as part of array.h)
        // - T must be a numeric type!
        // - the vectors x and y should be of equal size!
        // - if the size is unequal, the surplus elements of the larger vector
        //   will be ignored        
        Sample2d(const Vector<T>& x_data, const Vector<T>& y_data){
            this->_elements = std::fmin(x_data.get_elements(), y_data.get_elements());
            this->_x_data=new T(this->_elements);
            this->_y_data=new T(this->elements);            
            for (int n=0;n<this->_elements;n++){
                this->_x_data[n]=x_data.get(n);
                this->_y_data[n]=y_data.get(n);
            }
            _coefficients=new double(this->_elements);
            _y_regression=new double(this->elements);
            _residuals=new double(this->_elements);            
        };

        // parametric constructor for two array-type samples
        // - T must be a numeric type!
        // - arrays must be 1-dimensional!
        // - the arrays x and y should be of equal size!
        // - if the size is unequal, the surplus elements of the larger array
        //   will be ignored        
        Sample2d(const T (&x_data)[], const T (&y_data)[]){
            this->_elements = std::fmin(sizeof(x_data)/sizeof(T), sizeof(y_data)/sizeof(T));
            this->_x_data=new T(this->_elements);
            this->_y_data=T(this->_elements); 
            for (int n=0;n<this->_elements;n++){
                this->x_data[n]=x_data[n];
                this->y_data[n]=y_data[n];
            }
            _coefficients=new double(this->_elements);
            _y_regression=new double(this->elements);
            _residuals=new double(this->_elements);            
        };
        // destructor
        ~Sample2d(){
            delete[] _x_data;
            delete[] _y_data;
            delete[] _coefficients;
            delete[] _y_regression;
            delete[] _residuals;
        }
    private:
        int _elements;
        T* _x_data;
        T* _y_data; 
        bool _correlation_completed=false;
        double _z_score;
        double _t_score;
        double _Spearman_Rho; 
        double _x_mean;
        double _y_mean;   
        double _covariance;
        double _Pearson_R;   
        double _slope;
        double _r_squared;
        double _y_intercept;    
        double _RSS;        
        bool _lin_reg_completed=false;
        bool _poly_reg_completed=false;     
        double* _coefficients; //for polynomial regression
        double* _y_regression;
        double* _residuals;                                  
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