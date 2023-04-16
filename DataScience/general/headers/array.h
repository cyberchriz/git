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
        void set(std::initializer_list<int> index, T value);
        T get(std::initializer_list<int> index);
        int get_dimensions(){return dimensions;};
        int get_size(int dimension){return size[dimension];};
        int get_elements(){return elements;};
        void fill_values(T value);
        void fill_zeros(){fill_values(0);}
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);
        double mean(){return Sample<T>(data).mean();};
        double median(){return Sample<T>(data).median();};
        double variance(){return Sample<T>(data).variance();};
        double stddev(){return Sample<T>(data).stddev();};
        T sum();
        void add(const T value);
        void add(const Array& other);
        T product();
        void multiply(const T factor);
        void multiply(const Array& other);
        void substract(const T value);
        void substract(const Array& other);
        void divide(const T quotient);
        void divide(const Array& other);
        void pow(const T exponent);
        void sqrt();
        void replace(const T old_value, const T new_value);
        int find(T value);
        void function(const T (*pointer_to_function)(T));
        void operator=(const Array& other);
        // constructor declarations
        Array() = delete;
        Array(std::initializer_list<int> dim_size);
        // destructor declarations
        ~Array();
    protected:
        int get_element(std::initializer_list<int> index);
        T* data; // 1dimensional array of source data
        int elements=0; // total number of elements in all dimensions
        int dimensions=0;
        int* size; // holds the size (number of elements) per individual dimension  
    private:      
};

// derived class from Array<T>,
// for 1d vectors
template<typename T>
class Vector : public Array<T>{
    public:
        void set(std::initializer_list<int> index, const T value) = delete;
        T get(std::initializer_list<int> index) = delete;
        void set(const int index, const T value);
        T get(const int index);
        // constructor declarations
        Vector() = delete;
        Vector(const int elements);
        // destructor declarations
        ~Vector();
};

// derived class from Array<T>,
// for 2d matrix
template<typename T>
class Matrix : public Array<T>{
    public:
        void set(std::initializer_list<int> index, const T value) = delete;
        T get(std::initializer_list<int> index) = delete;
        void set(const int index_x, const int index_y, T value);
        T get(const int index_x, const int index_y);
        Matrix<T> dotproduct(const Matrix& other);
        Matrix<T> transpose();
        // constructor declarations
        Matrix() = delete;
        Matrix(const int elements_x, const int elements_y);
        // destructor declarations
        ~Matrix();
};

// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/array.cpp"