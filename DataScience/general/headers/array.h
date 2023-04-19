#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include <iostream>
#include <vector>
#include "sample.h"
#include "../../distributions/headers/random_distributions.h"

// forward declarations
template<typename T> class Matrix;
template<typename T> class Vector;

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
        int get_elements();

        // fill, initialize
        void fill_values(T value);
        void fill_zeros();
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);
        void fill_range(const T start=0, const T step=1);

        // distribution properties
        double mean();
        double median();
        double variance();
        double stddev();

        // addition
        T sum();
        std::unique_ptr<Array<T>> operator+(const T value);
        std::unique_ptr<Array<T>> operator+(const Array& other);
        void operator++();
        void operator+=(const T value);
        void operator+=(const Array& other);

        // substraction
        void substract(const T value);
        std::unique_ptr<Array<T>> operator-(const T value);
        std::unique_ptr<Array<T>> operator-(const Array& other);
        void operator--();
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
        std::unique_ptr<Array<T>> copy();

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
        std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols);
        std::unique_ptr<Matrix<T>> asMatrix();
        std::unique_ptr<Array<T>> asArray(const std::initializer_list<int>& initlist);

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
        int* _size; // holds the size (number of _elements) per individual dimension 
        
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
        void set(const int row, const int col, T value);
        T get(const int row, const int col);
        std::unique_ptr<Matrix<T>> dotproduct(const Matrix& other);
        std::unique_ptr<Matrix<T>> operator*(const Matrix& otehr);
        std::unique_ptr<Matrix<T>> transpose();
        void print(std::string delimiter=", ", std::string line_break="\n", bool with_indices=false);

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

        // dynamic handling
        int push_back(T value);
        T pop();
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
        double Dickey_Fuller();
        std::unique_ptr<Vector<T>> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2);
        std::unique_ptr<Vector<T>> sort(bool ascending=true);
        std::unique_ptr<Vector<T>> shuffle();
        std::unique_ptr<Vector<T>> log_transform();
        T polynomial_predict(T x,int power=5);
        double polynomial_MSE(int power=5);
        bool isGoodFit_linear(double threshold=0.95);
        bool isGoodFit_polynomial(int power=5,double threshold=0.95);
        T linear_predict(T x);
        double get_slope();
        double get_y_intercept();
        double get_r_squared_linear();
        double get_r_squared_polynomial(int power=5);

        // output
        void print(std::string delimiter=", ", std::string line_break="\n", bool with_indices=false);

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