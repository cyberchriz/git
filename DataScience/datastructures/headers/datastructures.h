#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <typeinfo>
#include "../../distributions/headers/random_distributions.h"
//#define MEMLOG
#include "../../../utilities/headers/memlog.h"
#include "activation.h"
#include "correlation.h"
#include "fill.h"
#include "histogram.h"
#include "regression.h"
#include "outliers.h"
#include "pooling.h"
#include "scaling.h"

// forward declarations
template<typename T> class Array;
template<typename T> class Matrix;
template<typename T> class Vector;

// list of time series differencing methods for stationarity transformation
enum DIFFERENCING{
   integer=1,
   logreturn=2,
   fractional=3,
   deltamean=4,
   original=5
};

// class for multidimensional arrays
template<typename T>
class Array{
    public:
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value);
        void set(const std::vector<int>& index, const T value);
        void set(const Vector<int>& index, const T value);
        void set(const int index, const T value);
        T get(const std::initializer_list<int>& index) const;
        T get(const std::vector<int>& index) const;
        T get(const Vector<int>& index) const;
        int get_dimensions() const;
        int get_size(int dimension) const;
        std::vector<int> get_shape() const {return dim_size};
        int get_elements() const;
        int get_subspace(int dimension) const;    
        int get_element(const std::initializer_list<int>& index) const;
        int get_element(const std::vector<int>& index) const;
        std::vector<int> get_index(int flattened_index) const;    
        std::type_info const& get_type(){return typeid(T);}

        // basic distribution properties
        T min() const;
        T max() const;
        T mode() const;
        double mean() const;
        double median() const;
        double variance() const;
        double stddev() const;
        Array<T> nested_min() const;
        Array<T> nested_max() const;        
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
        Array<T>& operator++(); //=prefix increment
        void operator+=(const T value);
        void operator+=(const Array<T>& other);

        // substraction
        Array<T> operator-(const T value) const;
        Array<T> operator-(const Array<T>& other) const;
        Array<T> operator--(int) const; //=postfix decrement
        Array<T>& operator--(); //=prefix decrement
        void operator-=(const T value);
        void operator-=(const Array<T>& other);

        // multiplication
        T product() const; // product reduction (=multiplying all elements with each other)
        virtual Array<T> operator*(const T factor) const; // elementwise multiplication
        virtual Array<T> tensordot(const Array<T>& other, const std::vector<int>& axes) const; //=tensor reduction
        virtual T dotproduct(const Array<T>& other) const; //=scalar product
        virtual T operator*(const Array<T>& other) const; //=alias for the dotproduct (=scalar product)
        void operator*=(const T factor);
        Array<T> Hadamard_product(const Array<T>& other) const;
        
        // division
        Array<T> operator/(const T quotient) const;
        void operator/=(const T quotient);
        Array<T> Hadamard_division(const Array<T>& other) const;


        // modulo
        void operator%=(const double num);
        Array<double> operator%(const double num) const;

        // exponentiation & logarithm
        Array<T> pow(const T exponent);
        Array<T> pow(const Array<T>& other);
        Array<T> sqrt();
        Array<T> log();
        Array<T> log10();

        // rounding (elementwise)
        Array<T> round();
        Array<T> floor();
        Array<T> ceil();
        Array<T> abs();

        // min, max (elementwise comparison)
        Array<T> min(const T value);
        Array<T> max(const T value);
        Array<T> min(const Array<T>& other);
        Array<T> max(const Array<T>& other);

        // trigonometric functions (elementwise)
        Array<T> cos();
        Array<T> sin();
        Array<T> tan();
        Array<T> acos();
        Array<T> asin();
        Array<T> atan();  

        // hyperbolic functions (elementwise)
        Array<T> cosh();
        Array<T> sinh();
        Array<T> tanh();
        Array<T> acosh();
        Array<T> asinh();
        Array<T> atanh();                

        // find, replace
        Array<T> replace(const T old_value, const T new_value);
        int find(const T value) const;
        Array<char> sign();

        // custom functions
        Array<T> function(const T (*pointer_to_function)(T));

        // assignment
        virtual Array<T>& operator=(const Array<T>& other);
        virtual Array<T>& operator=(Array<T>&& other) noexcept; // =move assignment

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
        Array<T&> operator*(); // dereference operator
        Array<T*> operator&(); // 'address-of' operator
        
        // conversion
        Vector<T> flatten() const;
        virtual Matrix<T> asMatrix(const int rows, const int cols, T init_value=0) const;
        virtual Matrix<T> asMatrix() const;
        Array<T> asArray(const std::initializer_list<int>& initlist, T init_value=0) const;
        virtual void reshape(std::vector<int> shape);
        virtual void reshape(std::initializer_list<int> shape);
        virtual Array<T> concatenate(const Array<T>& other, const int axis=0);

        // output
        void print(std::string comment="", std::string delimiter=", ", std::string line_break="\n", bool with_indices=false) const;

    protected:

        // protected member variables
        bool equalsize(const Array<T>& other) const;
        int data_elements=0; // total number of data_elements in all _dimensions
        int dimensions=0;
        std::vector<int> dim_size; // holds the size (number of data_elements) per individual dimension 
        std::vector<int> subspace_size;
        
        // protected methods
        void resizeArray(std::unique_ptr<T[]>& arr, const int newSize);         

    public:
        // main data buffer
        std::unique_ptr<T[]> data = nullptr; // 1dimensional array of source data

        // outsourced classes
        Fill<T> fill;
        Activation<T> activation;
        Scaling<T> scale;
        Outliers<T> outliers;
        Pooling<T> pool;

        // constructor & destructor declarations
        Array(){};
        Array(const std::initializer_list<int>& shape);
        Array(const std::vector<int>& shape);
        
        //=move constructor
        Array(Array&& other) noexcept;

        // copy constructor
        Array(const Array& other);
        virtual ~Array();   
};

// derived class from Array<T>, for 2d matrix
template<typename T>
class Matrix : public Array<T>{
    public:
        // getters & setters
        void set(const int row, const int col, T value);
        T get(const int row, const int col) const;

        // special matrix operations
        T dotproduct(const Matrix<T>& other) const; //=scalar product
        T operator*(const Matrix<T>& other) const; //=Alias for the dotproduct (=scalar product)
        Matrix<T> tensordot(const Matrix<T>& other) const; //=tensor reduction, alias for operator*
        Matrix<T> transpose() const;

        // assignment
        Matrix<T>& operator=(const Matrix<T>& other);
        Matrix<T>& operator=(Matrix<T>&& other) noexcept; // =move assignment

        // conversion
        Matrix<T> asMatrix(const int rows, const int cols, T init_value=0) const override;
        void reshape(const int rows, const int cols);
        Matrix<T> concatenate(const Matrix<T>& other, const int axis=0);


        // constructor declarations
        Matrix(){};
        Matrix(const int rows, const int cols);
        Matrix(Matrix&& other) noexcept; //=move constructor
        ~Matrix() override;
};

// derived class from Array<T>, for 1d vectors
template<typename T>
class Vector : public Array<T>{
    public:
        // dynamic handling
        int push_back(T value);
        T pop_last();
        T pop_first();
        T erase(const int index);
        int grow(const int additional_elements);
        int shrink(const int remove_amount);       
        void resize(const int newsize);        
        int get_capacity() const;
        int size() const;
        Vector<T> flatten()=delete;

        // Multiplication
        T dotproduct(const Vector<T>& other) const; //=scalar product, alias for operator*
        T operator*(const Vector<T>& other) const; //=scalar product, alias for dotproduct()

        // sample analysis
        Vector<int> ranking() const;
        Vector<T> exponential_smoothing(bool as_series=false) const;
        double weighted_average(bool as_series=true) const;
        double Dickey_Fuller(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        double Engle_Granger(const Vector<T>& other) const;
        Vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        Vector<T> sort(bool ascending=true) const;
        Vector<T> shuffle() const;
        double covariance(const Vector<T>& other) const;
        Vector<T> binning(const int bins);

        // assignment
        Vector<T>& operator=(const Vector<T>& other);
        Vector<T>& operator=(Vector<T>&& other) noexcept; // =move assignment

        // indexing
        T& operator[](const int index) const;
        T& operator[](const int index);

        // conversion
        static Vector<T> asVector(const std::vector<T>& other);
        Matrix<T> asMatrix(const int rows, const int cols, T init_value=0)  const override;
        Matrix<T> asMatrix() const override;
        Matrix<T> transpose() const;
        Vector<T> reverse() const;
        Vector<T> concatenate(const Vector<T>& other);

        // outsourced classes
        Regression<T> regression;
        Histogram<T> histogram;

        // constructor & destructor declarations
        Vector();
        Vector(const int elements);
        ~Vector() override;
        Vector(Vector&& other) noexcept; //=move constructor
    private:
        const float _reserve = 0.5;
        int capacity;
};



// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/datastructures.cpp"
