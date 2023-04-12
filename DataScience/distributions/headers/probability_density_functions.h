//+------------------------------------------------------------------+
//|      probability density functions (pdf)                         |
//+------------------------------------------------------------------+

/* A probability density function (PDF) is a statistical function that gives the probability
of a random variable taking on any particular value.
It is the derivative of the cumulative distribution function and is used to analyze
the probability of a random variable taking on a particular value. */

// author: Christian Suer

#pragma once
#include <cmath>
#define SCALE_FACTOR 0.707106781

template<typename T>
class PdfObject{
    public: 
        static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value,
                      "T must be either type <float> or <double>");       
        // methods
        static T gaussian(T x_val,T mu=0, T sigma=1);
        static T cauchy(T x_val,T x_peak, T gamma);
        static T laplace(T x_val, T mu=0, T sigma=1);
        static T pareto(T x_val, T alpha=1, T tail_index=1);
        static T lomax(T x_val, T alpha=1, T tail_index=1);
        // constructor
        PdfObject(){};
        // destructor
        ~PdfObject(){};
};

// ------------------------------------------------------------------
// ALIAS CLASS
template<typename T>
class pdf:public PdfObject<T>{};
// ------------------------------------------------------------------


// include .cpp resource (required due to this being a template class)
// -> thus mitigating 'undefined reference' compiler errors
#include "../sources/probability_density_functions.cpp"