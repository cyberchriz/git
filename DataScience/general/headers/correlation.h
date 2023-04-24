#pragma once
#include "array.h"

// forward declaration
template<typename T> class Array;
template<typename T> class Vector;

// return struct for correlation results
template<typename T>
struct CorrelationResults{ 
    double x_mean, y_mean;  
    double x_stddev, y_stddev;     
    double y_intercept, slope;
    double covariance;
    double Pearson_R, Spearman_Rho;   
    double r_squared;    
    double RSS, SST, SSE, SSR = 0; 
    double z_score, t_score;          
    std::vector<T> y_predict;
    void print(){
        std::cout
        << "Correlation Results (this vs. other):"
        << "\n   - x_mean = " << x_mean
        << "\n   - y_mean = " << y_mean
        << "\n   - x_stddev = " << x_stddev
        << "\n   - y_stddev = " << y_stddev
        << "\n   - y_intercept = " << y_intercept
        << "\n   - slope = " << slope
        << "\n   - covariance = " << covariance
        << "\n   - Pearson_R = " << Pearson_R
        << "\n   - Spearman_Rho = " << Spearman_Rho
        << "\n   - r_squared = " << r_squared
        << "\n   - RSS = " << RSS
        << "\n   - SST = " << SST
        << "\n   - SSE = " << SSE
        << "\n   - SSR = " << SSR
        << "\n   - z_score = " << z_score
        << "\n   - t_score = " << t_score << std::endl;
    }
    // constructor
    CorrelationResults(int elements){
        y_predict.resize(elements);
    }  
};

// class declaration
template<typename T>
class Correlation{
    public:
        CorrelationResults<T> get(const Vector<T>& y_data) const;
        // constructor
        Correlation() : x_data(nullptr){};
        Correlation(Vector<T>& x_data) : x_data(x_data){};
    private:
        Vector<T> x_data;
};


template<typename T>
CorrelationResults<T> Correlation<T>::get(const Vector<T>& y_data) const {
    if (x_data._elements != y_data._elements) {
        std::cout << "WARNING: Invalid use of method Vector<T>::correlation(); both vectors should have the same number of elements" << std::endl;
    }
    int elements=std::min(x_data._elements, y_data._elements);    
    std::unique_ptr<typename Array<T>::Correlation> result = std::make_unique<typename Array<T>::Correlation>(elements);

    // get empirical vector autocorrelation (Pearson coefficient R), assumimg linear dependence
    result->x_mean=x_data.mean();
    result->y_mean=y_data.mean();
    result->covariance=0;
    for (int i=0;i<elements;i++){
        result->covariance+=(x_data._data[i] - result->x_mean) * (y_data._data[i] - result->y_mean);
    }
    result->x_stddev=x_data.stddev();
    result->y_stddev=y_data.stddev();
    result->Pearson_R = result->covariance / (result->x_stddev * result->y_stddev);   

    // get r_squared (coefficient of determination) assuming linear dependence
    double x_mdev2_sum=0,y_mdev2_sum=0,slope_numerator=0;
    for (int i=0;i<elements;i++){
        x_mdev2_sum += std::pow(x_data._data[i] - result->x_mean, 2); //=slope denominator
        y_mdev2_sum += std::pow(y_data._data[i] - result->y_mean, 2); //=SST
        slope_numerator += (x_data._data[i] - result->x_mean) * (y_data._data[i] - result->y_mean);
    }
    result->SST = y_mdev2_sum;
    result->slope = slope_numerator / x_mdev2_sum;
    result->y_intercept = result->y_mean - result->slope * result->x_mean;

    // get regression line values
    for (int i=0; i<elements; i++){
        result->y_predict[i] = result->y_intercept + result->slope * x_data._data[i];
        // get sum of squared (y-^y) //=SSE   
        result->SSE += std::pow(result->y_predict[i] - result->y_mean, 2);
        result->SSR += std::pow(y_data._data[i] - result->y_predict[i], 2);
    };
    result->r_squared = result->SSE/(std::fmax(y_mdev2_sum,__DBL_MIN__)); //=SSE/SST, equal to 1-SSR/SST

    // Spearman correlation, assuming non-linear monotonic dependence
    auto rank_x = x_data.ranking();
    auto rank_y = y_data.ranking();
    double numerator=0;
    for (int i=0;i<elements;i++){
        numerator+=6*std::pow(rank_x._data[i] - rank_y._data[i],2);
    }
    result->Spearman_Rho=1-numerator/(elements*(std::pow(elements,2)-1));
    // test significance against null hypothesis
    double fisher_transform=0.5*std::log( (1 + result->Spearman_Rho) / (1 - result->Spearman_Rho) );
    result->z_score = sqrt((elements-3)/1.06)*fisher_transform;
    result->t_score = result->Spearman_Rho * std::sqrt((elements-2)/(1-std::pow(result->Spearman_Rho,2)));
    
    return std::move(*result);
}
