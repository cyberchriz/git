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
        Fill() : arr(nullptr), elements(0), dimensions(0) {}
        Fill(Array<T>* arr) : arr(arr){
            this->elements = arr->get_elements();
            this->dimensions = arr->get_dimensions();
        }; 
    private:            
        Array<T>* arr;
        int elements;
        int dimensions;
};


// +=================================+   
// | fill, initialize                |
// +=================================+

// fill entire array with given value
template<typename T>
void Fill<T>::values(const T value){
    for (int i=0;i<elements;i++){
        arr->data[i]=value;
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
    int max_index=arr->get_size(0);
    for (int i=1; i<dimensions; i++){
        max_index=std::min(max_index,arr->get_size(i));
    }
    std::vector<int> index(dimensions);
    // add 'ones' of identity matrix
    for (int i=0;i<max_index;i++){
        for (int d=0;d<dimensions;d++){
            index[d]=i;
        }
        arr->set(index,1);
    }
}

// fill with values from a random normal distribution
template<typename T>
void Fill<T>::random_gaussian(const T mu, const T sigma){
    for (int i=0; i<elements; i++){
        arr->data[i] = Random<T>::gaussian(mu,sigma);
    }           
}

// fill with values from a random uniform distribution
template<typename T>
void Fill<T>::random_uniform(const T min, const T max){
    for (int i=0; i<elements;i++){
        arr->data[i] = Random<T>::uniform(min,max);
    }
}
// fills the array with a continuous
// range of numbers (with specified start parameter
// referring to the zero position and a step parameter)
// in all dimensions
template<typename T>
void Fill<T>::range(const T start, const T step){
    if (dimensions==1){
        for (int i=0;i<elements;i++){
            arr->data[i]=start+i*step;
        }
    }
    else {
        std::vector<int> index(dimensions);
        std::fill(index.begin(),index.end(),0);
        for (int d=0;d<dimensions;d++){
            for (int i=0;i<arr->get_size(d);i++){
                index[d]=i;
                arr->set(index,start+i*step);
            }
        }
    }
}

// randomly sets a specified fraction of the values to zero
// and retains the rest
template<typename T>
void Fill<T>::dropout(double ratio){
    for (int i=0;i<elements;i++){
        arr->data[i] *= Random<double>::uniform() > ratio;
    }
}

// randomly sets the specified fraction of the values to zero
// and the rest to 1 (default: 0.5, i.e. 50%)
template<typename T>
void Fill<T>::binary(double ratio){
    for (int i=0;i<elements;i++){
        arr->data[i] = Random<double>::uniform() > ratio;
    }
}