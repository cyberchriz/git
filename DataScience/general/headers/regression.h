#pragma once
#include "array.h"

// forward declaration
template<typename T> class Vector;
template<typename T> class Matrix;


// result struct for linear regression
template<typename T>
struct LinRegResult{
    double x_mean, y_mean=0;
    double y_intercept, _slope;
    double r_squared;
    std::unique_ptr<double[]> _y_regression;
    std::unique_ptr<double[]> _residuals;
    double SST=0;
    double SSR=0;
    T predict(const T x){return y_intercept + _slope * x;}
    bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
    // parametric constructor
    LinRegResult(const int elements){
        _y_regression = std::make_unique<double[]>(elements);
        _residuals = std::make_unique<double[]>(elements);
    }
    // delete default constructor (because only a constructor that passes data makes sense)
    LinRegResult() = delete;
    ~LinRegResult(){};
};

// nested struct for polynomial regression
template<typename T>
struct PolyRegResult{
    public:
        double SS_res=0;
        double SS_tot=0;
        double RSS=0;
        double MSE;      
        double RSE;
        double y_mean=0;
        double x_mean=0;
        double r_squared;
        std::unique_ptr<double[]> coefficient;  
        bool _is_good_fit(double threshold=0.95){return r_squared>threshold;}
        T predict(const T x){
            double y_pred = 0;
            for (int p = 0; p<=power;p++) {
                y_pred += coefficient[p] * std::pow(x, p);
            }
            return y_pred;
        };  
        // constructor & destructor
        PolyRegResult() : power(0), coefficient(nullptr) {}; 
        PolyRegResult(const int elements, const int power) : power(power) {
            coefficient = std::make_unique<double[]>(power+1);
        };
        ~PolyRegResult(){}
    private:
        int power;
};

template<typename T>
class Regression{
    private:
        Vector<T>* x_data;
    public:
        LinRegResult<T> linear(const Vector<T>& y_data) const;
        PolyRegResult<T> polynomial(const Vector<T>& y_data, const uint power=5) const;
        // constructor
        Regression() : x_data(nullptr){}
        Regression(const Vector<T>* x_data) : x_data(x_data){}
};

// performs linear regression with the source vector as
// x_data and a second vector as corresponding the y_data;
// the results will be stored in a struct;
// make sure that both vectors have the same number of
// elements (otherwise the surplus elements of the
// larger vector will be discarded)
template<typename T>
LinRegResult<T> Regression<T>::linear(const Vector<T>& y_data) const {
    // create result struct
    int elements = std::min(x_data->_elements, y_data.get_elements());
    std::unique_ptr<LinRegResult<T>> result = std::make_unique<LinRegResult<T>>(elements);
    // get mean for x and y values
    for (int i = 0; i < elements; i++){
        result->x_mean += x_data->_data[i];
        result->y_mean += y_data._data[i];
    }
    result->x_mean /= elements;
    result->y_mean /= elements;
    // get sum of squared mean deviations
    double x_mdev2_sum = 0, y_mdev2_sum = 0, slope_num = 0;
    for (int n = 0; n < elements; n++){
        double x_mdev = x_data->_data[n] - result->x_mean;
        double y_mdev = y_data._data[n] - result->y_mean;
        x_mdev2_sum += x_mdev * x_mdev;
        y_mdev2_sum += y_mdev * y_mdev;
        slope_num += x_mdev * y_mdev;
        result->_y_regression[n] = result->y_intercept + result->_slope * x_data->_data[n];
        result->_residuals[n] = y_data._data[n] - result->_y_regression[n];
        result->SSR += result->_residuals[n] * result->_residuals[n];
    }
    // get slope
    result->_slope = slope_num / (x_mdev2_sum + std::numeric_limits<T>::min());
    // get y intercept
    result->y_intercept = result->y_mean - result->_slope * result->x_mean;
    // get r_squared
    result->SST = y_mdev2_sum;
    result->r_squared = 1 - result->SSR / (result->SST + std::numeric_limits<T>::min());

    return std::move(*result);
}


// performs polynomial regression (to the specified power)
// with the source vector as the x data and a second vector
// as the corresponding y data;
// make sure that both vectors have the same number of
// elements (y_datawise the surplus elements of the
// larger vector will be discarded)
template<typename T>
PolyRegResult<T> Regression<T>::polynomial(const Vector<T>& y_data, const uint power) const {
    // create result struct
    int elements=std::min(x_data->get_elements(), y_data.get_elements());
    std::unique_ptr<PolyRegResult<T>> result = std::make_unique<PolyRegResult<T>>(elements, power);

    // Create matrix of x values raised to different powers
    auto X = std::make_unique<Matrix<T>>(elements, power + 1);
    for (int i = 0; i < elements; i++) {
        for (int p = 1; p <= power; p++) {
            X->set(i,p,std::pow(x_data->_data[i],p));
        }
    }

    // Perform normal equation
    for (int i = 0; i <= power; i++) {
        for (int j = 0; j <= power; j++) {
            T sum = 0;
            for (int k = 0; k < elements; k++) {
                sum += X->get(k,i) * X->get(k,j);
            }
            X->set(i,j,sum);
        }
        result->coefficient[i] = 0;
        for (int k = 0; k < elements; k++) {
            result->coefficient[i] += y_data._data[k] * X->get(k,i);
        }
    }
    // Get R-squared value and other statistics
    result->y_mean = std::accumulate(y_data._data.begin(), y_data._data.end(), 0.0) / elements;
    result->x_mean = std::accumulate(x_data->_data.begin(), x_data->_data.end(), 0.0) / elements;
    for (int i = 0; i < elements; i++) {
        double y_pred = 0;
        for (int j = 0; j <= power; j++) {
            y_pred += result->coefficient[j] * pow(x_data->_data[i], j);
        }
        result->SS_res += std::pow(y_data._data[i] - y_pred, 2);
        result->SS_tot += std::pow(y_data._data[i] - result->y_mean, 2);
    }
    result->r_squared = 1 - result->SS_res / result->SS_tot;
    result->RSS = std::sqrt(result->SS_res / (elements - power - 1));
    result->MSE = result->RSS/elements;

    return std::move(*result);
}


