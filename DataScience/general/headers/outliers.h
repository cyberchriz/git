#pragma once
#include "array.h"

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
        Outliers() : _arr(nullptr){}
        Outliers(Array<T>* arr) : _arr(arr){}         
    private:
        Array<T>* _arr;
};


template<typename T>
void Outliers<T>::truncate(double z_score){
    double mean = _arr->mean();
    double stddev = _arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] > upper_margin){
            _arr->_data[i] = upper_margin;
        }
        if (_arr->_data[i] < lower_margin){
            _arr->data[i] = lower_margin;
        }
    }
}

template<typename T>
void Outliers<T>::winsoring(double z_score){
    double mean = _arr->mean();
    double stddev = _arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    T highest_valid = mean;
    T lowest_valid = mean;
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] < upper_margin && _arr->_data[i] > lower_margin){
            highest_valid = std::fmax(highest_valid, _arr->_data[i]);
            lowest_valid = std::fmin(lowest_valid, _arr->_data[i]);
        }
    }    
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] > upper_margin){
            _arr->data[i] = highest_valid;
        }
        else if (_arr->_data[i] < lower_margin){
            _arr->data[i] = lowest_valid;
        }
    }
}

template<typename T>
void Outliers<T>::mean_imputation(double z_score){
    double mean = _arr->mean();
    double stddev = _arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] > upper_margin || _arr->_data[i] < lower_margin){
            _arr->_data[i] = mean;
        }
    }
}

template<typename T>
void Outliers<T>::median_imputation(double z_score){
    double median = _arr->median();
    double mean = _arr->mean();
    double stddev = _arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] > upper_margin || _arr->_data[i] < lower_margin){
            _arr->_data[i] = median;
        }
    }
}

template<typename T>
void Outliers<T>::value_imputation(T value, double z_score){
    double mean = _arr->mean();
    double stddev = _arr->stddev();
    double lower_margin = mean - z_score*stddev;
    double upper_margin = mean + z_score*stddev;
    for (int i=0;i<_arr->_elements;i++){
        if (_arr->_data[i] > upper_margin || _arr->_data[i] < lower_margin){
            _arr->_data[i] = value;
        }
    }
}

