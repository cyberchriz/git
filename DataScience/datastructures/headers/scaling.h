#pragma once
#include "datastructures.h"

// forward declaration
template<typename T> class Array;

// +=================================+   
// | Feature Scaling                 |
// +=================================+

template<typename T>
class Scaling {
    public:
        void minmax(T min=0,T max=1){
            T data_min = arr->data.min();
            T data_max = arr->data.max();
            double factor = (max-min) / (data_max-data_min);
            for (int i=0; i<arr->get_elements(); i++){
                arr->data[i] = (arr->data[i] - data_min) * factor + min;
            }
        }
        void mean(){
            T data_min = arr->data.min();
            T data_max = arr->data.max();
            T range = data_max - data_min;
            double mean = arr->mean();
            for (int i=0; i<arr->get_elements(); i++){
                arr->data[i] = (arr->data[i] - mean) / range;
            }
        }
        void standardized(){
            double mean = arr->mean();
            double stddev = arr->stddev();
            for (int i=0; i<arr->get_elements(); i++){
                arr->data[i] = (arr->data[i] - mean) / stddev;
            }  
        }
        void unit_length(){
            // calculate the Euclidean norm of the data array
            T norm = 0;
            int elements = arr->get_elements();
            for (int i = 0; i < elements; i++) {
                norm += std::pow(arr->data[i], 2);
            }
            if (norm==0){return;}
            norm = std::sqrt(norm);
            // scale the data array to unit length
            for (int i = 0; i < elements; i++) {
                arr->data[i] /= norm;
            }  
        }
        // constructor
        Scaling(): arr(nullptr){}
        Scaling(Array<T>* arr): arr(arr){}
    private:
        Array<T>* arr;        
};