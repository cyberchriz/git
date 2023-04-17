#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include "sample.h"
#include "../../distributions/headers/random_distributions.h"

// class for multidimensional arrays
template<typename T>
class Array{
    public:     
        // getters & setters
        void set(std::initializer_list<int> index, const T value);
        T get(std::initializer_list<int> index);
        int get_dimensions();
        int get_size(int dimension);
        int get_elements();

        // fill, initialize
        void fill_values(T value);
        void fill_zeros();
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);

        // distribution properties
        double mean();
        double median();
        double variance();
        double stddev();

        // addition
        T sum();
        Array<T> operator+(const T value);
        Array<T> operator+(const Array& other);
        void operator++();
        void operator+=(const T value);
        void operator+=(const Array& other);

        // substraction
        void substract(const T value);
        Array<T> operator-(const T value);
        Array<T> operator-(const Array& other);
        void operator--();
        void operator-=(const T value);
        void operator-=(const Array& other);

        // multiplication
        T product();
        Array<T> operator*(const T factor);
        void operator*=(const T factor);
        Array<T> Hadamard(const Array& other);
        
        // division
        Array<T> operator/(const T quotient);
        void operator/=(const T quotient);

        // modulo
        void operator%=(const double num);
        Array<double> operator%(const double num);

        // exponentiation
        void pow(const T exponent);
        void pow(const Array& other);
        void sqrt();

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
        Array<T> copy();

        // elementwise comparison by single value
        Array<bool> operator>(const T value);
        Array<bool> operator>=(const T value);
        Array<bool> operator==(const T value);
        Array<bool> operator!=(const T value);
        Array<bool> operator<(const T value);
        Array<bool> operator<=(const T value);

        // elementwise comparison with second array
        Array<bool> operator>(const Array& other);
        Array<bool> operator>=(const Array& other);
        Array<bool> operator==(const Array& other);
        Array<bool> operator!=(const Array& other);
        Array<bool> operator<(const Array& other);
        Array<bool> operator<=(const Array& other);

        // elementwise logical operations
        Array<bool> operator&&(const bool value);
        Array<bool> operator||(const bool value);
        Array<bool> operator!();
        Array<bool> operator&&(const Array& other);
        Array<bool> operator||(const Array& other);

        // type casting
        template<typename C> operator Array<C>();

        // constructor & destructor declarations
        Array() = delete;
        Array(std::initializer_list<int> dim_size);
        ~Array();

        // public member variables
        T* _data; // 1dimensional array of source _data
    
    private:

        // private member variables
        bool equal_size(const Array& other);
        std::initializer_list<int> _init_list;
        int _elements=0; // total number of _elements in all _dimensions
        int _dimensions=0;
        int* _size; // holds the size (number of _elements) per individual dimension 
        
        // private methods
        int get_element(std::initializer_list<int> index);
        void Array<T>::resizeArray(T*& arr, int newSize);  
};

// derived class from Array<T>,
// for 2d matrix
template<typename T>
class Matrix : public Array<T>{
    public:
        void set(const int row, const int col, T value);
        T get(const int row, const int col);
        Matrix<T> dotproduct(const Matrix& other);
        Matrix<T> operator*(const Matrix& otehr);
        Matrix<T> transpose();
        // constructor declarations
        Matrix() = delete;
        Matrix(const int rows, const int cols);
        // destructor declarations
        ~Matrix(){};
};

// derived class from Array<T>,
// for 1d vectors
template<typename T>
class Vector : public Array<T>{
    public:
        // getters & setters
        void set(std::initializer_list<int> index, const T value) = delete;
        T get(std::initializer_list<int> index) = delete;
        void set(const int index, const T value);
        T get(const int index);

        // dynamic handling
        int push_back(T value);
        T pop();
        int get_capacity();
        int size();

        // Vector as Matrix
        Matrix<T> transpose();
        Matrix<T> asMatrix();

        // Multiplication
        T dotproduct(const Vector& other);
        T operator*(const Vector& other);

        // sample analysis
        Vector<int> ranking(bool ascending=true);
        Vector<T> exponential_smoothing(bool as_series=false);
        double Dickey_Fuller();
        Vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        Vector<T> sort(bool ascending=true);
        Vector<T> shuffle();
        Vector<T> log_transform();
        T polynomial_predict(T x,int power=5);
        double polynomial_MSE(int power=5);
        bool isGoodFit_linear(double threshold=0.95);
        bool isGoodFit_polynomial(int power=5,double threshold=0.95);
        T linear_predict(T x);
        double get_slope();
        double get_y_intercept();
        double get_r_squared_linear();
        double get_r_squared_polynomial(int power=5);

        // constructor & destructor declarations
        Vector() = delete;
        Vector(const int _elements);
        ~Vector(){};
    private:
        const double _reserve = 0.5;
        int _capacity;
};

// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/array.cpp"