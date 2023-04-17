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
        void set(std::initializer_list<int> index, T value);
        T get(std::initializer_list<int> index);
        int get_dimensions(){return dimensions;};
        int get_size(int dimension){return size[dimension];};
        int get_elements(){return elements;};

        // fill, initialize
        void fill_values(T value);
        void fill_zeros(){fill_values(0);}
        void fill_identity();
        void fill_random_gaussian(const T mu=0, const T sigma=1);
        void fill_random_uniform(const T min=0, const T max=1.0);

        // distribution properties
        double mean(){return Sample<T>(data).mean();};
        double median(){return Sample<T>(data).median();};
        double variance(){return Sample<T>(data).variance();};
        double stddev(){return Sample<T>(data).stddev();};

        // addition
        T sum();
        void add(const T value);
        void add(const Array& other);
        Array<T> operator+(const T value){Array<T> result(init_list); result=this; return result.add(value);}
        void operator++(){this->add(1);}
        void operator+=(const T value){this->add(value);};
        void operator+=(const Array& other){this->add(other);}

        // substraction
        void substract(const T value);
        void substract(const Array& other);
        Array<T> operator-(const T value){Array<T> result(init_list); result=this; return result.substract(value);}
        void operator--(){this->substract(1);}
        void operator-=(const T value){this->substract(value);}
        void operator-=(const Array& other){this->substract(other);}

        // multiplication
        T product();
        void multiply(const T factor);
        void multiply(const Array& other);
        Array<T> operator*(const T factor){Array<T> result(init_list); result=this; return result.multiply(factor);}
        void operator*=(const T factor){this->multiply(factor);}
        void operator*=(const Array& other){this->multiply(other);}
        
        // division
        void divide(const T quotient);
        void divide(const Array& other);
        Array<T> operator/(const T quotient){Array<T> result(init_list); result=this; return result.divide(quotient);}
        void operator/=(const T quotient){this->divide(quotient);}
        void operator/=(const Array& other){this->divide(other);}

        // modulo
        void operator%=(const double num){for (int i=0;i<elements;i++){data[i]%=num;}}
        Array<double> modulo(const double num){Array<T> result(init_list); result=this; return result%=num;}
        Array<double> operator%(const double num){Array<T> result(init_list); result=this; return result%num;}

        // exponentiation
        void pow(const T exponent);
        Array<T> pow(const Array& other){Array<T> result(init_list);for (int i=0;i<elements;i++){result.data[i]=pow(this->data[i],other->data[i]);};return result;}
        void sqrt();

        // find, replace
        void replace(const T old_value, const T new_value);
        int find(T value);

        // custom functions
        void function(const T (*pointer_to_function)(T));

        // assignment
        void operator=(const Array& other);
        Array<T> copy(){Array<T> result(init_list);result.data=this.data;return result;}

        // elementwise comparison by single value
        Array<bool> operator>(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]>value;};return result}
        Array<bool> operator>=(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]>=value;};return result}
        Array<bool> operator==(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]==value;};return result}
        Array<bool> operator!=(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]!=value;};return result}
        Array<bool> operator<(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]<value;};return result}
        Array<bool> operator<=(const T value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]<=value;};return result}

        // elementwise comparison with second array
        Array<bool> operator>(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]>other->data[i];};return result}
        Array<bool> operator>=(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]>=other->data[i];};return result}
        Array<bool> operator==(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]==other->data[i];};return result}
        Array<bool> operator!=(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]!=other->data[i];};return result}
        Array<bool> operator<(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]<other->data[i];};return result}
        Array<bool> operator<=(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]<=other->data[i];};return result}

        // elementwise logical operations
        Array<bool> operator&&(const bool value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]&&value;};return result}
        Array<bool> operator||(const bool value){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]||value;};return result}
        Array<bool> operator!(){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=!this->data[i];};return result}
        Array<bool> operator&&(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]&&other->data[i];};return result}
        Array<bool> operator||(const Array& other){Array<bool> result(init_list);for (int i=0;i<elements;i++){result.data[i]=this->data[i]||other->data[i];};return result}

        // type casting
        template<typename C> operator Array<C>(){Array<C> result(init_list);for (int i=0;i<elements;i++){result.data[i]=C(this->data[i]);}; return result;}
        template<typename C> explicit operator Array<C>(){Array<C> result(init_list);for (int i=0;i<elements;i++){result.data[i]=C(this->data[i]);}; return result;}

        // constructor & destructor declarations
        Array() = delete;
        Array(std::initializer_list<int> dim_size);
        ~Array();

    protected:
        std::initializer_list<int> init_list;
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
        // getters & setters
        void set(std::initializer_list<int> index, const T value) = delete;
        T get(std::initializer_list<int> index) = delete;
        void set(const int index, const T value);
        T get(const int index);

        // sample analysis
        Vector<int> ranking(bool ascending=true){Vector<int> result(elements); result.data=Sample(this->data)::ranking(ascending);return result;}
        Vector<T> exponential_smoothing(bool as_series=false){Vector<T> result(elements);result.data=Sample(this->data)::exponential_smoothing(as_series);return result;} 
        double Dickey_Fuller(){return Sample(this->data)::Dickey_Fuller();}
        Vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2){Vector<T> result(elements);result.data=Sample(this->data)::stationary(method,degree,fract_exponent);return result;}
        Vector<T> sort(bool ascending=true){Vector<T> result(elements);result.data=Sample(this->data)::sort(ascending);return result;}
        Vector<T> shuffle(){Vector<T> result(elements);result.data=Sample(this->data)::shuffle();return result;}
        Vector<T> log_transform(){Vector<T> result(elements);result.data=Sample(this->data)::log_transform();return result;}
        T polynomial_predict(T x,int power=5){Sample temp(this->data);temp.polynomial_regression();return temp.polynomial_predict(x);}
        double polynomial_MSE(int power=5){Sample temp(this->data);temp.polynomial_regression(power);return temp.polynomial_MSE();}
        bool isGoodFit_linear(double threshold=0.95){Sample temp(this->data);temp.linear_regression();return temp.isGoodFit(threshold);}
        bool isGoodFit_polynomial(int power=5,double threshold=0.95){Sample temp(this->data);temp.polynomial_regression(power);return temp.isGoodFit(threshold);}
        T linear_predict(T x){return Sample(this->data)::linear_predict(x);}
        double get_slope(){return Sample(this->data)::get_slope();} 
        double get_y_intercept(){return Sample(this->data)::get_y_intercept();}
        double get_r_squared_linear(){Sample temp(this->data);temp.linear_regression();return temp.get_r_squared()};
        double get_r_squared_polynomial(int power=5){Sample temp(this->data);temp.polynomial_regression(power);return temp.get_r_squared();}

        // constructor & destructor declarations
        Vector() = delete;
        Vector(const int elements);
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