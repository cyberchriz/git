#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <typeindex>
#include <type_traits>
#include <boost/core/demangle.hpp>
#include "../../distributions/headers/random_distributions.h"
#include "../../distributions/headers/cumulative_distribution_functions.h"
//#define MEMLOG
#include "../../../utilities/headers/memlog.h"
#include "../../../utilities/headers/log.h"
#include "../../../utilities/headers/initlists.h"

// forward declarations
template<typename T> class Array;

// list of time series differencing methods for stationarity transformation
enum DIFFERENCING{
   integer,
   logreturn,
   fractional,
   deltamean,
   original
};

// list of available activation functions
enum ActFunc {
    RELU,       // rectified linear unit (ReLU)
    LRELU,      // leaky rectified linear unit (LReLU)
    ELU,        // exponential linar unit (ELU)
    SIGMOID,    // sigmoid (=logistic)
    TANH,       // hyperbolic tangent (tanh)
    SOFTMAX,    // softmax (=normalized exponential)
    IDENT       // identity function
};

// list of available pooling methods
enum PoolMethod {
    MAX,
    MAXABS,
    MIN,
    MEAN,
    MEDIAN,
    MODE
};

// return struct for correlation results
// (Pearson, Spearman, ANOVA, covariance)
template<typename T>
struct CorrelationResult{ 
    double x_mean, y_mean;  
    double x_stddev, y_stddev;     
    double y_intercept, slope;
    double covariance;
    double Pearson_R, Spearman_Rho;   
    double r_squared;    
    double RSS, SST, SSE, SSR, MSE, MSR = 0; 
    double ANOVA_F, ANOVA_p=0;
    double z_score, t_score;          
    Array<T> y_predict;
    void print(){
        std::cout // print to console
        <<   "=========================================================================="
        << "\nCorrelation Results (this=x vs. other=y):"
        << "\n   - mean value of x = " << x_mean
        << "\n   - mean value of y = " << y_mean
        << "\n   - standard deviation of x = " << x_stddev
        << "\n   - standard deviation of y = " << y_stddev
        << "\n   - regression line y-intercept = " << y_intercept
        << "\n   - regression line slope = " << slope
        << "\n   - covariance between x & y = " << covariance
        << "\n   - Pearson correlation coefficient R = " << Pearson_R
        << "\n   - Spearman correlation coefficient Rho = " << Spearman_Rho
        << "\n   - coefficient of determination (r-squared) = " << r_squared
        << "\n   - residual sum of squares (RSS) = " << RSS
        << "\n   - total sum of squares (SST) = " << SST
        << "\n   - explained sum of squares (SSE) = " << SSE
        << "\n   - regression sum of squares (SSR) = " << SSR
        << "\n   - mean squared error (MSE) = " << MSE
        << "\nANOVA:"
        << "\n   - ANOVA F-statistic = " << ANOVA_F
        << "\n   - ANOVA p-value = " << ANOVA_p
        << "\nHypothesis Testing:"
        << "\n   - z-score = " << z_score
        << "\n   - t-score = " << t_score
        << "\n==========================================================================" << std::endl;
    }

    // constructor
    CorrelationResult(int elements){
        y_predict=Array<T>(elements);
    }  
};

// result struct for histograms
template<typename T>
struct HistogramResult{
    private:
        struct Histogrambar{
            T lower_boundary;
            T upper_boundary;
            int abs_count=0;
            double rel_count;
        };
    public:    
        T min, max;
        T bar_width;
        T _width;
        int bars;
        std::unique_ptr<Histogrambar[]> bar;
        HistogramResult() : bars(0) {};
        HistogramResult(int bars) : bars(bars) {
            bar = std::make_unique<Histogrambar[]>(bars);
        }
        ~HistogramResult(){};
};

// result struct for linear regression
template<typename T>
struct LinRegResult{
    double x_mean, y_mean=0;
    double y_intercept, _slope;
    double r_squared;
    std::unique_ptr<double[]> _y_regression;
    std::unique_ptr<double[]> _residuals;
    double SST=0;
    double SSR=0;
    T predict(const T x){return y_intercept + _slope * x;}
    bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
    void print(){
        std::cout // print to console
        <<   "=========================================================================="
        << "\nLinear Regression Results (with 'this'=x vs. 'other'=y):"
        << "\n   - y = " << _slope << " * x + " << y_intercept
        << "\n   - r_squared = " << r_squared << "(=" << (r_squared>0.95 ? "" : "no ") << "'good fit' with confidence interval of 0.95)"
        << "\n   - SST = " << SST
        << "\n   - SSR = " << SSR
        << "\n==========================================================================" << std::endl;
    }    
    // parametric constructor
    LinRegResult(const int elements){
        _y_regression = std::make_unique<double[]>(elements);
        _residuals = std::make_unique<double[]>(elements);
    }
    // delete default constructor (because only a constructor that passes data makes sense)
    LinRegResult() = delete;
    ~LinRegResult(){};
};

// nested struct for polynomial regression
template<typename T>
struct PolyRegResult{
    public:
        double SS_res=0;
        double SS_tot=0;
        double RSS=0;
        double MSE;      
        double RSE;
        double y_mean=0;
        double x_mean=0;
        double r_squared;
        std::unique_ptr<double[]> coefficient;  
        bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
        T predict(const T x){
            double y_pred = 0;
            for (int p = 0; p<=power;p++) {
                y_pred += coefficient[p] * std::pow(x, p);
            }
            return y_pred;
        };  
        // constructor & destructor
        PolyRegResult() : power(0), coefficient(nullptr) {}; 
        PolyRegResult(const int elements, const int power) : power(power) {
            coefficient = std::make_unique<double[]>(power+1);
        };
        ~PolyRegResult(){}
    private:
        int power;
};

// class for multidimensional arrays
template<typename T>
class Array{
    public:
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value);
        void set(const std::vector<int>& index, const T value);
        void set(const Array<int>& index, const T value);
        void set(const int row, const int col, const T value);
        void set(const int index, const T value);
        T get(const std::initializer_list<int>& index) const;
        T get(const std::vector<int>& index) const;
        T get(const Array<int>& index) const;
        T get(const int row, const int col) const;
        int get_dimensions() const {return int(this->dim_size.size());}
        int get_size(int dimension) const {return dimension<this->get_dimensions() ? this->dim_size[dimension] : 0;};
        int get_size() const {return this->data_elements;};
        int get_elements() const {return this->data_elements;};
        std::vector<int> get_shape() const {return dim_size;};
        std::vector<int> get_convolution_shape(std::vector<int>& filter_shape, const bool padding=false) const;
        std::vector<int> get_stacked_shape() const;
        std::string get_shapestring() const;
        int get_subspace(int dimension) const;
        std::vector<int> subspace() const {return this->subspace_size;};       
        int get_capacity() const {return this->capacity;};    
        int get_element(const std::initializer_list<int>& index) const;
        int get_element(const std::vector<int>& index) const;
        std::vector<int> get_index(int element) const;    
        const std::type_info& get_type() const {return typeid(T);};
        std::string get_typename() const {return boost::core::demangle(std::type_index(typeid(*this)).name());}

        // helper method to confirm valid index
        bool index_isvalid(const std::vector<int>& index) const;
        bool index_isvalid(const int index) const;

        // fill, initialize
        void fill_values(const T value);
        void fill_zeros();
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);
        void fill_random_binary(double ratio=0.5);
        void fill_random_sign(double ratio=0.5);
        void fill_range(const T start=0, const T step=1);
        void fill_dropout(double ratio=0.2);
        void fill_Xavier_normal(int fan_in, int fan_out);
        void fill_Xavier_uniform(int fan_in, int fan_out);
        void fill_Xavier_sigmoid(int fan_in, int fan_out);
        void fill_He_ReLU(int fan_in);
        void fill_He_ELU(int fan_in);

        // basic distribution properties
        T min() const;
        T max() const;
        T maxabs() const;
        T mode() const;
        double mean() const;
        double median() const;
        double variance() const;
        double stddev() const;        
        Array<double> nested_min() const;
        Array<double> nested_max() const; 
        Array<double> nested_mean() const; 
        Array<double> nested_variance() const;                             
        Array<double> nested_stddev() const;      
        double skewness() const;
        double kurtosis() const;

        // addition
        T sum() const;
        Array<T> operator+(const T value) const;
        Array<T> operator+(const Array<T>& other) const;
        Array<T> operator++(int) const; //=postfix increment
        Array<T>& operator++() const; //=prefix increment
        void operator+=(const T value);
        void operator+=(const Array<T>& other);

        // substraction
        Array<T> operator-(const T value) const;
        Array<T> operator-(const Array<T>& other) const;
        Array<T> operator--(int) const; //=postfix decrement
        Array<T>& operator--() const; //=prefix decrement
        void operator-=(const T value);
        void operator-=(const Array<T>& other);

        // multiplication
        T product() const; // product reduction (=multiplying all elements with each other)
        Array<T> operator*(const T factor) const; // elementwise multiplication with a scalar
        void operator*=(const T factor); // alias for elementwise multiplication with a scalar
        Array<T> tensordot(const Array<T>& other, const std::vector<int>& axes) const; //=tensor reduction
        Array<T> tensordot(const Array<T>& other) const; //=tensor reduction
        Array<T> operator*(const Array<T>& other) const; //=alias for tensordot matrix multiplication
        void operator*=(const Array<T>& other); //=alias for tensordot matrix multiplication
        T dotproduct(const Array<T>& other) const; //=scalar product
        Array<T> Hadamard_product(const Array<T>& other) const;
        
        // division
        Array<T> operator/(const T quotient) const;
        void operator/=(const T quotient);
        Array<T> Hadamard_division(const Array<T>& other);

        // modulo
        void operator%=(const double num);
        Array<double> operator%(const double num) const;

        // exponentiation & logarithm
        Array<T> pow(const T exponent) const;
        Array<T> pow(const Array<T>& other) const;
        Array<T> sqrt() const;
        Array<T> log() const;
        Array<T> log10() const;

        // rounding (elementwise)
        Array<T> round() const;
        Array<T> floor() const;
        Array<T> ceil() const;
        Array<T> abs() const;

        // min, max (elementwise comparison)
        Array<T> min(const T value) const;
        Array<T> max(const T value) const;
        Array<T> min(const Array<T>& other) const;
        Array<T> max(const Array<T>& other) const;

        // trigonometric functions (elementwise)
        Array<T> cos() const;
        Array<T> sin() const;
        Array<T> tan() const;
        Array<T> acos() const;
        Array<T> asin() const;
        Array<T> atan() const;  

        // hyperbolic functions (elementwise)
        Array<T> cosh() const;
        Array<T> sinh() const;
        Array<T> tanh() const;
        Array<T> acosh() const;
        Array<T> asinh() const;
        Array<T> atanh() const;                

        // find, replace
        Array<T> replace(const T old_value, const T new_value) const;
        int find(const T value) const;
        Array<char> sign() const;

        // scale
        Array<double> scale_minmax(T min=0,T max=1) const;
        Array<double> scale_mean() const;
        Array<double> scale_standardized() const;
        Array<double> scale_unit_length() const;

        // activation functions
        Array<T> activation(ActFunc activation_function) const;
        Array<T> derivative(ActFunc activation_function) const;

        // custom functions
        Array<T> function(const T (*pointer_to_function)(T)) const;

        // outlier treatment
        Array<T> outliers_truncate(double z_score=3) const;
        Array<T> outliers_winsoring(double z_score=3) const;
        Array<T> outliers_mean_imputation(double z_score=3) const;
        Array<T> outliers_median_imputation(double z_score=3) const;
        Array<T> outliers_value_imputation(T value=0, double z_score=3) const;        

        // assignment
        Array<T>& operator=(const Array<T>& other); // =copy assignment
        Array<T>& operator=(Array<T>&& other) noexcept; // =move assignment
        Array<T>& operator=(const T (&arr)[]); // =copy assignment
        Array<T>& operator=(T (&&arr)[]) noexcept; // =move assignment
        Array<T> operator=(std::vector<T> vector); // std::vector copy assignment

        // elementwise comparison by single value
        Array<bool> operator>(const T value) const;
        Array<bool> operator>=(const T value) const;
        Array<bool> operator==(const T value) const;
        Array<bool> operator!=(const T value) const;
        Array<bool> operator<(const T value) const;
        Array<bool> operator<=(const T value) const;

        // elementwise comparison with second array
        Array<bool> operator>(const Array<T>& other) const;
        Array<bool> operator>=(const Array<T>& other) const;
        Array<bool> operator==(const Array<T>& other) const;
        Array<bool> operator!=(const Array<T>& other) const;
        Array<bool> operator<(const Array<T>& other) const;
        Array<bool> operator<=(const Array<T>& other) const;

        // elementwise logical operations
        Array<bool> operator&&(const bool value) const;
        Array<bool> operator||(const bool value) const;
        Array<bool> operator!() const;
        Array<bool> operator&&(const Array<T>& other) const;
        Array<bool> operator||(const Array<T>& other) const;

        // type casting
        template<typename C> operator Array<C>();

        // pointers
        Array<typename std::remove_pointer<T>::type> operator*() const; // dereference operator
        Array<T*> operator&() const; // 'address-of' operator
        
        // conversion
        Array<T> flatten() const;
        void reshape(std::vector<int> shape, const T init_value=0);
        void reshape(std::initializer_list<int> shape, const T init_value=0);
        void reshape(Array<int> shape, const T init_value=0);
        Array<T> concatenate(const Array<T>& other, const int axis=0) const;
        Array<T> add_dimension(int size, T init_value=0) const;
        Array<T> padding(const int amount, const T value=0) const;
        Array<T> padding_pre(const int amount, const T value=0) const;
        Array<T> padding_post(const int amount, const T value=0) const;
        Array<Array<T>> dissect(int axis) const;
        Array<T> pool(PoolMethod method, const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape) const;
        Array<T> pool(PoolMethod method, const std::vector<int> slider_shape, const std::vector<int> stride_shape) const;
        Array<T> convolution(const Array<T>& filter, bool padding=false) const;    
        Array<T> transpose() const;    
        Array<T> reverse() const;
        T stack() const;
        Array<T> shuffle() const;

        // 1d Array statistics
        CorrelationResult<T> correlation(const Array<T>& other) const;
        LinRegResult<T> regression_linear(const Array<T>& other) const;
        PolyRegResult<T> regression_polynomial(const Array<T>& other, const int power) const;
        HistogramResult<T> histogram(int bars) const;

        // Array dynamic handling
        int push_back(T init_value);
        T pop_last();
        T pop_first();
        T erase(const int index);
        int grow(const int additional_elements, int dimension=0, T init_value=0);
        int shrink(const int remove_amount, int dimension=0);       
        void resize(const int newsize, int dimension=0, T init_value=0);         

        // 1d Array sample analysis
        Array<int> ranking() const;
        Array<T> exponential_smoothing(bool as_series=false) const;
        double weighted_average(bool as_series=true) const;
        double Dickey_Fuller(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        double Engle_Granger(const Array<T>& other) const;
        Array<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        Array<T> sort(bool ascending=true) const;
        double covariance(const Array<T>& other) const;
        Array<T> binning(const int bins) const;                   

        // indexing
        T& operator[](const int index) const;
        T& operator[](const int index);

        // output
        void print(std::string comment="", std::string delimiter=", ", std::string line_break="\n", bool with_indices=false) const;

    protected:

        // protected member variables
        bool equalsize(const Array<T>& other) const;
        int data_elements=0; // total number of data_elements in all _dimensions
        std::vector<int> dim_size; // holds the size (number of data_elements) per individual dimension 
        std::vector<int> subspace_size;       

    public:
        // main data buffer
        std::unique_ptr<T[]> data = nullptr; // 1dimensional array of source data

        // constructor & destructor declarations
        Array(){};
        Array(const std::initializer_list<int>& shape);
        Array(const std::vector<int>& shape);
        Array(const Array<int>& shape);
        Array(const int elements);
        
        //=move constructor
        Array(Array&& other) noexcept;

        // copy constructor
        Array(Array& other);
        ~Array();   
    
    private:
        void resize_array(std::unique_ptr<T[]>& arr, const int oldSize, const int newSize, T init_value=0);
        const float _reserve = 0.5;
        int capacity;        
};

// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/datastructures.cpp"