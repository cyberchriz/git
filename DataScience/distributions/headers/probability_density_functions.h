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
                      std::is_same<T, double>::value ||
                      std::is_same<T, double double>::value,
                      "T must be either float, double or double double");       
        // methods
        static T gaussian(const T&x_val,const T&mu=0, const T&sigma=1);
        static T cauchy(const T&x_val,const T&x_peak, const T&gamma);
        static T laplace(const T& x_val, const T& mu=0, const T& sigma=1);
        static T pareto(const T& x_val, const T& alpha=1, const T& tail_index=1);
        static T lomax(const T& x_val, const T& alpha=1, const T& tail_index=1);
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
