#pragma once
#include "datastructures.h"

// forward declaration
template<typename T> class Array;

// class for outlier detection and treatment
template<typename T>
class Outliers{
    public:
        void truncate(double z_score=3);
        void winsoring(double z_score=3);
        void mean_imputation(double z_score=3);
        void median_imputation(double z_score=3);
        void value_imputation(T value=0, double z_score=3);
        // constructor
        Outliers() : arr(nullptr){}
        Outliers(Array<T>* arr) : arr(arr){}         
    private:
        Array<T>* arr;
};


template<typename T>
void Outliers<T>::truncate(double z_score){
    double mean = arr->mean();
    double stddev = arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] > upper_margin){
            arr->_data[i] = upper_margin;
        }
        if (arr->_data[i] < lower_margin){
            arr->data[i] = lower_margin;
        }
    }
}

template<typename T>
void Outliers<T>::winsoring(double z_score){
    double mean = arr->mean();
    double stddev = arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    T highest_valid = mean;
    T lowest_valid = mean;
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] < upper_margin && arr->_data[i] > lower_margin){
            highest_valid = std::fmax(highest_valid, arr->_data[i]);
            lowest_valid = std::fmin(lowest_valid, arr->_data[i]);
        }
    }    
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] > upper_margin){
            arr->data[i] = highest_valid;
        }
        else if (arr->_data[i] < lower_margin){
            arr->data[i] = lowest_valid;
        }
    }
}

template<typename T>
void Outliers<T>::mean_imputation(double z_score){
    double mean = arr->mean();
    double stddev = arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] > upper_margin || arr->_data[i] < lower_margin){
            arr->_data[i] = mean;
        }
    }
}

template<typename T>
void Outliers<T>::median_imputation(double z_score){
    double median = arr->median();
    double mean = arr->mean();
    double stddev = arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] > upper_margin || arr->_data[i] < lower_margin){
            arr->_data[i] = median;
        }
    }
}

template<typename T>
void Outliers<T>::value_imputation(T value, double z_score){
    double mean = arr->mean();
    double stddev = arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<arr->_elements;i++){
        if (arr->_data[i] > upper_margin || arr->_data[i] < lower_margin){
            arr->_data[i] = value;
        }
    }
}

