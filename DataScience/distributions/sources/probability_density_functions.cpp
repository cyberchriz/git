//+------------------------------------------------------------------+
//|      probability density functions (pdf)                         |
//+------------------------------------------------------------------+

/* A probability density function (PDF) is a statistical function that gives the probability
of a random variable taking on any particular value.
It is the derivative of the cumulative distribution function */

#pragma once
#include <cmath>
#include "../headers/probability_density_functions.h"

// gaussian normal probability density function
template<typename T>
T PdfObject<T>::gaussian(const T& x_val,const T& mu, const T& sigma){
    return (1/sqrt(2*M_PI*pow(sigma,2))) * exp(-0.5*pow((x_val-mu)/sigma,2));
}

// Cauchy probability density function
template<typename T>
T PdfObject<T>::cauchy(const T& x_val,const T& x_peak, const T& gamma){
    return 1 / (M_PI*gamma*(1+pow((x_val-x_peak)/gamma,2)));
}

// Laplace probability density function;
// scale factor default: sigma/sqrt(2)=0.707106781
template<typename T>
T PdfObject<T>::laplace(const T& x_val,const T& mu,const T& sigma) {
    static double scale_factor;
    scale_factor = sigma/sqrt(2);
    return exp(-fabs(x_val-mu)/scale_factor)/(2*scale_factor);
}

// Pareto probability density function;
// note: not defined for x<=0 !!
// note: the tail index can be understood as the scale parameter (default=1)
//       the alpha can be understood as the shape parameter (default=1)
template<typename T>
T PdfObject<T>::pareto(const T& x_val,const T& alpha, const T& tail_index) {
    if (x_val>=tail_index) {
        return (alpha*pow(tail_index,alpha))/pow(x_val,alpha+1);
    }
    else {return 0;}
}
  
// Lomax (=Pareto type II) probability density function;
// note: not defined for x<=0 !!
// note: the tail index can be understood as the scale parameter (default=1)
//       the alpha can be understood as the shape parameter (default=1)
template<typename T>
T PdfObject<T>::lomax(const T& x_val,const T& alpha,const T& tail_index) {
    return (alpha/tail_index)*pow(1+std::fmax(x_val,0)/tail_index,-(alpha+1));
}