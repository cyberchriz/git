#pragma once
#include <cmath> 
#include <memory>
#include "sample.h"
#include "../../distributions/headers/random_distributions.h"

// class for multidimensional arrays
template<typename T>
class Array{
    public:
        void set(int* index, T value);
        T get(int* index);
        int get_dimensions(){return dimensions;};
        int get_size(int dimension){return size[dimension];};
        int get_elements(){return elements;};
        void fill_values(T value);
        void fill_zeros(){fill_values(0);}
        void fill_identity();
        void fill_random_gaussian(T mu=0, T sigma=1);
        void fill_random_uniform(T min=0,T max=1.0);
        double mean(){return Sample<T>(data).mean();};
        double median(){return Sample<T>(data).median();};
        double variance(){return Sample<T>(data).variance();};
        double stddev(){return Sample<T>(data).stddev();};
        T sum();
        void add(T value);
        void add(const Array& array);
        T product();
        void multiply(T factor);
        void multiply(const Array& other);
        void substract(T value);
        void substract(const Array& other);
        void divide(T quotient);
        void divide(const Array& other);
        void pow(T exponent);
        void sqrt();
        void replace(T old_value, T new_value);
        int find(T value);
        void function(T (*pointer_to_function)(T));
        void operator=(const Array& other);
        // constructor declarations
        Array() = delete;
        Array(int* dim_size);
        // destructor declarations
        ~Array();
    protected:
        int get_element(int* index);
        T *data; // 1dimensional array of source data
        int elements=0; // total number of elements in all dimensions
        int dimensions=0;
        int *size; // holds the size (number of elements) per individual dimension  
    private:      
};

// derived class from Array<T>,
// for 1d vectors
template<typename T>
class Vector : Array<T>{
    public:
        void set(int* index, T value) = delete;
        T get(int* index) = delete;
        void set(int index, T value);
        T get(int index);
        // constructor declarations
        Vector() = delete;
        Vector(int elements);
        // destructor declarations
        ~Vector();
};

// derived class from Array<T>,
// for 2d matrix
template<typename T>
class Matrix : Array<T>{
    public:
        void set(int* index, T value) = delete;
        T get(int* index) = delete;
        void set(int index_x, int index_y, T value);
        T get(int index_x, int index_y);
        Matrix<T> dotproduct(const Matrix& other);
        Matrix<T> transpose();
        // constructor declarations
        Matrix() = delete;
        Matrix(int elements_x, int elements_y);
        // destructor declarations
        ~Matrix();
};