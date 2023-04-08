//+------------------------------------------------------------------+
//|      cumulative distribution functions (cdf)                     |
//+------------------------------------------------------------------+

#pragma once
#include <cmath>
#include "../headers/cumulative_distribution_functions.h"

// returns the probability of a random value
// as part of a gaussian normal distribution  (with a given µ and sigma)
// being less or equal than the given 'x_val'
template<typename T>
T CdfObject<T>::gaussian(T x_val, T mu,T sigma) {
    return 0.5*(1+erf((x_val-mu)/(sigma*sqrt(2))));
}
  
// returns the probability of a random value
// as part of a Cauchy distribution  (with given parameter x_peak and sigma)
// being less or equal than the given 'x_val'
template<typename T>
T CdfObject<T>::cauchy(T x_val, T x_peak, T gamma) {
    return 0.5 + atan((x_val-x_peak)/gamma)/M_PI;
}
  
// returns the probability of a random value
// as part of a Laplace distribution  (with given µ)
// being less or equal than the given 'x_val';
// the scale_factor is sigma/sqrt(2)=0.707106781 by default
template<typename T>
T CdfObject<T>::laplace(T x_val, T mu, T sigma){
    static double scale_factor;
    scale_factor=sigma/sqrt(2);
    if (x_val<mu)
        {return 0.5*exp((x_val-mu)/scale_factor);}
    else
        {return 1-0.5*exp(-(x_val-mu)/scale_factor);}
}
  
// returns the probability of a random value
// as part of a Pareto distribution
// being less or equal than the given 'x_val';
// alpha is 1 by default;
// the tail_index is 1 by default;
// note: not defined for x<=0!!
template<typename T>
T CdfObject<T>::pareto(T x_val, T alpha, T tail_index) {
    if (x_val>=tail_index)
        {return 1-pow(tail_index/x_val,alpha);}
    else
        {return 0;}
}
  
// returns the probability of a random value
// as part of a Lomax distribution
// being less or equal than the given 'x_val';
// alpha is 1 by default;
// the tail_index is 1 by default;
// note: not defined for x<=0!!
template<typename T>
T CdfObject<T>::lomax(T x_val, T alpha, T tail_index){
    return 1-pow(1+std::fmax(0,x_val)/tail_index,-alpha);
}