#pragma once
#include "array.h"

// forward declaration
template<typename T> class Array;

// +=================================+   
// | Feature Scaling                 |
// +=================================+

template<typename T>
class Scaling {
    public:
        void minmax(T min=0,T max=1){
            T data_min = _arr->_data.min();
            T data_max = _arr->_data.max();
            double factor = (max-min) / (data_max-data_min);
            for (int i=0; i<_arr->get_elements(); i++){
                _arr->_data[i] = (_arr->_data[i] - data_min) * factor + min;
            }
        }
        void mean(){
            T data_min = _arr->_data.min();
            T data_max = _arr->_data.max();
            T range = data_max - data_min;
            double mean = _arr->mean();
            for (int i=0; i<_arr->get_elements(); i++){
                _arr->_data[i] = (_arr->_data[i] - mean) / range;
            }
        }
        void standardized(){
            double mean = _arr->mean();
            double stddev = _arr->stddev();
            for (int i=0; i<_arr->get_elements(); i++){
                _arr->_data[i] = (_arr->_data[i] - mean) / stddev;
            }  
        }
        void unit_length(){
            // calculate the Euclidean norm of the data array
            T norm = 0;
            int elements = _arr->get_elements();
            for (int i = 0; i < elements; i++) {
                norm += std::pow(_arr->_data[i], 2);
            }
            if (norm==0){return;}
            norm = std::sqrt(norm);
            // scale the data array to unit length
            for (int i = 0; i < elements; i++) {
                _arr->_data[i] /= norm;
            }  
        }
        // constructor
        Scaling(): _arr(nullptr){}
        Scaling(Array<T>* arr): _arr(arr){}
    private:
        Array<T>* _arr;        
};