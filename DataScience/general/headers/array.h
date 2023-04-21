#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "../../distributions/headers/random_distributions.h"
//#define MEMLOG
#include "../../../utilities/headers/memlog.h"

// forward declarations
template<typename T> class Matrix;
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
    T witdh;
    std::vector<Histogram_bar<T>> bar;
    Histogram(int bars){
        bar.reserve(bars);
        for (int n=0;n<bars;n++){
            bar.push_back(Histogram_bar<T>());
        }
    }
};

// Linear Regression struct
template<typename T>
struct LinReg{
    double x_mean=0;
    double y_mean=0;
    double y_intercept;
    double slope;
    double r_squared;
    double* y_regression;
    double* residuals;
    double SST=0;
    double SSR=0;
    T predict(const T x){return y_intercept + slope * x;}
    bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
    // constructor & destructor
    LinReg(const int elements){
        y_regression = new double(elements);
        residuals = new double(elements);
    }
    LinReg() = delete;
    ~LinReg(){
        delete[] y_regression;
        delete[] residuals;
    }
};

// Polynomial Regression struct
template<typename T>
struct PolyReg{
    public:
        double SS_res=0;
        double SS_tot=0;
        double RSS=0;
        double MSE;      
        double RSE;
        double y_mean=0;
        double x_mean=0;
        double r_squared;
        double* coefficient;  
        bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
        T predict(const T x){
            double y_pred = 0;
            for (int p = 0; p<=power;p++) {
                y_pred += coefficient[p] * std::pow(x, p);
            }
            return y_pred;
        };  
        // constructor & destructor
        PolyReg(const int elements, const int power){
            this->elements=elements;
            this->power=power;
            coefficient = new double(power+1);
        }
        ~PolyReg(){
            delete[] coefficient;
        }
    private:
        int elements;
        int power;
};

// Correlation struct
template<typename T>
struct Correlation{
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
    double* y_regression;
    double* residuals;   
};

// class for multidimensional arrays
template<typename T>
class Array{
    public:     
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value);
        void set(const std::vector<int>& index, const T value);
        T get(const std::initializer_list<int>& index);
        T get(const std::vector<int>& index);
        int get_dimensions();
        int get_size(int dimension);
        int get_elements() const;

        // fill, initialize
        void fill_values(T value);
        void fill_zeros();
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);
        virtual void fill_range(const T start=0, const T step=1);

        // distribution properties
        double mean();
        double median();
        double variance();
        double stddev();

        // addition
        T sum();
        std::unique_ptr<Array<T>> operator+(const T value);
        std::unique_ptr<Array<T>> operator+(const Array& other);
        std::unique_ptr<Array<T>> operator++(int); //=postfix increment
        Array<T>& operator++(); //=prefix increment
        void operator+=(const T value);
        void operator+=(const Array& other);

        // substraction
        void substract(const T value);
        std::unique_ptr<Array<T>> operator-(const T value);
        std::unique_ptr<Array<T>> operator-(const Array& other);
        std::unique_ptr<Array<T>> operator--(int); //=postfix decrement
        Array<T>& operator--(); //=prefix decrement
        void operator-=(const T value);
        void operator-=(const Array& other);

        // multiplication
        T product();
        std::unique_ptr<Array<T>> operator*(const T factor);
        void operator*=(const T factor);
        std::unique_ptr<Array<T>> Hadamard(const Array& other);
        
        // division
        std::unique_ptr<Array<T>> operator/(const T quotient);
        void operator/=(const T quotient);

        // modulo
        void operator%=(const double num);
        std::unique_ptr<Array<double>> operator%(const double num);

        // exponentiation
        void pow(const T exponent);
        void pow(const Array& other);
        void sqrt();
        void log();
        void log10();

        // rounding
        void round();
        void floor();
        void ceil();

        // find, replace
        void replace(const T old_value, const T new_value);
        int find(T value);

        // custom functions
        void function(const T (*pointer_to_function)(T));

        // assignment
        void operator=(const Array& other);

        // elementwise comparison by single value
        std::unique_ptr<Array<bool>> operator>(const T value);
        std::unique_ptr<Array<bool>> operator>=(const T value);
        std::unique_ptr<Array<bool>> operator==(const T value);
        std::unique_ptr<Array<bool>> operator!=(const T value);
        std::unique_ptr<Array<bool>> operator<(const T value);
        std::unique_ptr<Array<bool>> operator<=(const T value);

        // elementwise comparison with second array
        std::unique_ptr<Array<bool>> operator>(const Array& other);
        std::unique_ptr<Array<bool>> operator>=(const Array& other);
        std::unique_ptr<Array<bool>> operator==(const Array& other);
        std::unique_ptr<Array<bool>> operator!=(const Array& other);
        std::unique_ptr<Array<bool>> operator<(const Array& other);
        std::unique_ptr<Array<bool>> operator<=(const Array& other);

        // elementwise logical operations
        std::unique_ptr<Array<bool>> operator&&(const bool value);
        std::unique_ptr<Array<bool>> operator||(const bool value);
        std::unique_ptr<Array<bool>> operator!();
        std::unique_ptr<Array<bool>> operator&&(const Array& other);
        std::unique_ptr<Array<bool>> operator||(const Array& other);

        // type casting
        template<typename C> operator Array<C>();

        // conversion
        std::unique_ptr<Vector<T>> flatten();
        virtual std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0);
        virtual std::unique_ptr<Matrix<T>> asMatrix();
        std::unique_ptr<Array<T>> asArray(const std::initializer_list<int>& initlist, T init_value=0);

        // output
        void print(std::string comment="", std::string delimiter=", ", std::string line_break="\n", bool with_indices=false);        

        // constructor & destructor declarations
        Array(){};
        Array(const std::initializer_list<int>& init_list);
        Array(const std::vector<int>& dimensions);
        ~Array();

        // public member variables
        T* _data; // 1dimensional array of source _data
    
    protected:

        // protected member variables
        bool equal_size(const Array& other);
        int _elements=0; // total number of _elements in all _dimensions
        int _dimensions=0;
        std::vector<int> _size; // holds the size (number of _elements) per individual dimension 
        
        // protected methods
        int get_element(const std::initializer_list<int>& index);
        int get_element(const std::vector<int>& index);
        void resizeArray(T*& arr, const int newSize);
        std::initializer_list<int> array_to_initlist(int* arr, int size);
        std::unique_ptr<int[]> initlist_to_array(const std::initializer_list<int>& lst);
};

// derived class from Array<T>, for 2d matrix
template<typename T>
class Matrix : public Array<T>{
    public:
        // getters & setters
        void set(const int row, const int col, T value);
        T get(const int row, const int col);

        // fill, initialize
        void fill_range(const T start=0, const T step=1) override;

        // special matrix operations
        std::unique_ptr<Matrix<T>> dotproduct(const Matrix& other);
        std::unique_ptr<Matrix<T>> operator*(const Matrix& otehr);
        std::unique_ptr<Matrix<T>> transpose();

        // assignment
        std::unique_ptr<Matrix<T>> operator=(const Matrix& other);

        // conversion
        std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0) override;

        // constructor declarations
        Matrix(){};
        Matrix(const int rows, const int cols);
};

// derived class from Array<T>, for 1d vectors
template<typename T>
class Vector : public Array<T>{
    public:
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value) = delete;
        T get(const std::initializer_list<int>& index) = delete;
        void set(const int index, const T value);
        T get(const int index);

        // fill, initialize
        void fill_range(const T start=0, const T step=1) override;

        // dynamic handling
        int push_back(T value);
        T pop();
        T pop_first();
        int get_capacity();
        int size();
        void resize(const int new_size);
        int grow(const int additional_elements,T value=0);
        int shrink(const int remove_amount);
        std::unique_ptr<Vector<T>> flatten()=delete;

        // transpose
        std::unique_ptr<Matrix<T>> transpose();

        // Multiplication
        T dotproduct(const Vector& other);
        T operator*(const Vector& other);

        // sample analysis
        std::unique_ptr<Vector<int>> ranking(bool ascending=true);
        std::unique_ptr<Vector<T>> exponential_smoothing(bool as_series=false);
        double weighted_average(bool as_series=true);
        double Dickey_Fuller();
        std::unique_ptr<Vector<T>> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        std::unique_ptr<Vector<T>> sort(bool ascending=true);
        std::unique_ptr<Vector<T>> shuffle();
        std::unique_ptr<LinReg<T>> linear_regression(const Vector<T>& other);
        std::unique_ptr<PolyReg<T>> polynomial_regression(const Vector<T>& other, const int power=5);
        std::unique_ptr<Histogram<T>> histogram(uint bars);

        // assignment
        std::unique_ptr<Vector<T>> operator=(const Vector& other);

        // conversion
        std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0) override;
        std::unique_ptr<Matrix<T>> asMatrix() override;

        // constructor & destructor declarations
        Vector(){};
        Vector(const int elements);
    private:
        const float _reserve = 0.5;
        int _capacity;
};

// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/array.cpp"