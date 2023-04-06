//+------------------------------------------------------------------+
//|      random values from a given distribution                     |
//+------------------------------------------------------------------+

// author: Christian Suer

#pragma once
#include <cmath>

template<typename T>
class Random {
    public:   
        static T gaussian(const T& mu=0, const T& sigma=1);
        static T cauchy(const T& x_peak=0, const T& gamma=1);
        static T uniform(const T& min=0, const T& max=1);
        static T laplace(const T& mu=0, const T& sigma=1);
        static T pareto(const T& alpha=1, const T& tail_index=1);
        static T lomax(const T& alpha=1, const T& tail_index=1);
        static T binary();     
        static T sign();
};

// ALIAS CLASS
template<typename T> class rnd: public Random<T>{};