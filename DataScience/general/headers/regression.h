#pragma once
#include "array.h"

// forward declaration
template<typename T> class Vector;
template<typename T> class Matrix;


// result struct for linear regression
template<typename T>
struct LinRegResult{
    double _x_mean, _y_mean=0;
    double _y_intercept, _slope;
    double r_squared;
    std::unique_ptr<double[]> _y_regression;
    std::unique_ptr<double[]> _residuals;
    double _SST=0;
    double _SSR=0;
    T predict(const T x){return _y_intercept + _slope * x;}
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
        double _SS_res=0;
        double _SS_tot=0;
        double _RSS=0;
        double _MSE;      
        double _RSE;
        double _y_mean=0;
        double _x_mean=0;
        double _r_squared;
        std::unique_ptr<double[]> _coefficient;  
        bool _is_good_fit(double threshold=0.95){return _r_squared>threshold;}
        T predict(const T x){
            double y_pred = 0;
            for (int p = 0; p<=_power;p++) {
                y_pred += _coefficient[p] * std::pow(x, p);
            }
            return y_pred;
        };  
        // constructor & destructor
        PolyRegResult() : _power(0), _coefficient(nullptr) {}; 
        PolyRegResult(const int elements, const int power) : _power(power) {
            _coefficient = std::make_unique<double[]>(_power+1);
        };
        ~PolyRegResult(){}
    private:
        int _power;
};

template<typename T>
class Regression{
    private:
        Vector<T>* _x_data;
    public:
        LinRegResult<T> linear(const Vector<T>& y_data) const;
        PolyRegResult<T> polynomial(const Vector<T>& y_data, const uint power=5) const;
        // constructor
        Regression() : _x_data(nullptr){}
        Regression(const Vector<T>* x_data) : _x_data(x_data){}
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
    int elements = std::min(_x_data->_elements, y_data.get_elements());
    std::unique_ptr<LinRegResult<T>> result = std::make_unique<LinRegResult<T>>(elements);
    // get mean for x and y values
    for (int i = 0; i < elements; i++){
        result->_x_mean += _x_data->_data[i];
        result->_y_mean += y_data._data[i];
    }
    result->_x_mean /= elements;
    result->_y_mean /= elements;
    // get sum of squared mean deviations
    double x_mdev2_sum = 0, y_mdev2_sum = 0, slope_num = 0;
    for (int n = 0; n < elements; n++){
        double x_mdev = _x_data->_data[n] - result->_x_mean;
        double y_mdev = y_data._data[n] - result->_y_mean;
        x_mdev2_sum += x_mdev * x_mdev;
        y_mdev2_sum += y_mdev * y_mdev;
        slope_num += x_mdev * y_mdev;
        result->_y_regression[n] = result->_y_intercept + result->_slope * _x_data->_data[n];
        result->_residuals[n] = y_data._data[n] - result->_y_regression[n];
        result->_SSR += result->_residuals[n] * result->_residuals[n];
    }
    // get slope
    result->_slope = slope_num / (x_mdev2_sum + std::numeric_limits<T>::min());
    // get y intercept
    result->_y_intercept = result->_y_mean - result->_slope * result->_x_mean;
    // get r_squared
    result->_SST = y_mdev2_sum;
    result->_r_squared = 1 - result->_SSR / (result->_SST + std::numeric_limits<T>::min());

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
    int elements=std::min(_x_data->get_elements(), y_data.get_elements());
    std::unique_ptr<PolyRegResult<T>> result = std::make_unique<PolyRegResult<T>>(elements, power);

    // Create matrix of x values raised to different powers
    auto X = std::make_unique<Matrix<T>>(elements, power + 1);
    for (int i = 0; i < elements; i++) {
        for (int p = 1; p <= power; p++) {
            X->set(i,p,std::pow(_x_data->_data[i],p));
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
        result->_coefficient[i] = 0;
        for (int k = 0; k < elements; k++) {
            result->_coefficient[i] += y_data._data[k] * X->get(k,i);
        }
    }
    // Get R-squared value and other statistics
    result->_y_mean = std::accumulate(y_data._data.begin(), y_data._data.end(), 0.0) / elements;
    result->_x_mean = std::accumulate(_x_data->_data.begin(), _x_data->_data.end(), 0.0) / elements;
    for (int i = 0; i < elements; i++) {
        double y_pred = 0;
        for (int j = 0; j <= power; j++) {
            y_pred += result->_coefficient[j] * pow(_x_data->_data[i], j);
        }
        result->_SS_res += std::pow(y_data._data[i] - y_pred, 2);
        result->_SS_tot += std::pow(y_data._data[i] - result->_y_mean, 2);
    }
    result->_r_squared = 1 - result->_SS_res / result->_SS_tot;
    result->_RSS = std::sqrt(result->SS_res / (elements - power - 1));
    result->_MSE = result->_RSS/elements;

    return std::move(*result);
}


