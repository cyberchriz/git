#pragma once
#include "array.h"

// forward declaration
template<typename T> class Array;


// Struct for Filling / Intializing
template<typename T>
class Fill {
    public:
        void values(const T value);
        void zeros();
        void identity();
        void random_gaussian(const T mu=0, const T sigma=1);
        void random_uniform(const T min=0, const T max=1.0);
        void range(const T start=0, const T step=1);
        void dropout(double ratio=0.2);
        void binary(double ratio=0.5);
        // constructor
        Fill() : _arr(nullptr), _elements(0), _dimensions(0) {}
        Fill(Array<T>* arr) : _arr(arr){
            this->_elements = arr->get_elements();
            this->_dimensions = arr->get_dimensions();
        }; 
    private:            
        Array<T>* _arr;
        int _elements;
        int _dimensions;
};


// +=================================+   
// | fill, initialize                |
// +=================================+

// fill entire array with given value
template<typename T>
void Fill<T>::values(const T value){
    for (int i=0;i<_elements;i++){
        _arr->_data[i]=value;
    }
}

// initialize all values of the array with zeros
template<typename T>
void Fill<T>::zeros(){
    values(0);
}

// fill with identity matrix
template<typename T>
void Fill<T>::identity(){
    // initialize with zeros
    values(0);
    // get size of smallest dimension
    int max_index=_arr->get_size(0);
    for (int i=1; i<_dimensions; i++){
        max_index=std::min(max_index,_arr->get_size(i));
    }
    std::vector<int> index(_dimensions);
    // add 'ones' of identity matrix
    for (int i=0;i<max_index;i++){
        for (int d=0;d<_dimensions;d++){
            index[d]=i;
        }
        _arr->set(index,1);
    }
}

// fill with values from a random normal distribution
template<typename T>
void Fill<T>::random_gaussian(const T mu, const T sigma){
    for (int i=0; i<_elements; i++){
        _arr->_data[i] = Random<T>::gaussian(mu,sigma);
    }           
}

// fill with values from a random uniform distribution
template<typename T>
void Fill<T>::random_uniform(const T min, const T max){
    for (int i=0; i<_elements;i++){
        _arr->_data[i] = Random<T>::uniform(min,max);
    }
}
// fills the _array with a continuous
// range of numbers (with specified start parameter
// referring to the zero position and a step parameter)
// in all dimensions
template<typename T>
void Fill<T>::range(const T start, const T step){
    if (_dimensions==1){
        for (int i=0;i<_elements;i++){
            _arr->_data[i]=start+i*step;
        }
    }
    else {
        std::vector<int> index(_dimensions);
        std::fill(index.begin(),index.end(),0);
        for (int d=0;d<_dimensions;d++){
            for (int i=0;i<_arr->get_size(d);i++){
                index[d]=i;
                _arr->set(index,start+i*step);
            }
        }
    }
}

// randomly sets a specified fraction of the values to zero
// and retains the rest
template<typename T>
void Fill<T>::dropout(double ratio){
    for (int i=0;i<_elements;i++){
        _arr->_data[i] *= Random<double>::uniform() > ratio;
    }
}

// randomly sets the specified fraction of the values to zero
// and the rest to 1 (default: 0.5, i.e. 50%)
template<typename T>
void Fill<T>::binary(double ratio){
    for (int i=0;i<_elements;i++){
        _arr->_data[i] = Random<double>::uniform() > ratio;
    }
}