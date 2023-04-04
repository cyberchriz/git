//+------------------------------------------------------------------+
//|      cumulative distribution functions (cdf)                     |
//+------------------------------------------------------------------+

/* A cumulative distribution function gives the probability of a random variable
taking on values less than or equal to a given value. */

// author: Christian Suer

#define SCALE_FACTOR 0.707106781

template<typename T>
class CdfObject{
    public:
        static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, double double>::value,
                      "T must be either float, double or double double");    
        // methods
        static T gaussian(T x_val,T mu=0,T sigma=1);
        static T cauchy(T x_val,T x_peak=0,T gamma=1);
        static T laplace(T x_val,T mu=0,T sigma=1);
        static T pareto(T x_val,T alpha=1,T tail_index=1);
        static T lomax(T x_val,T alpha=1,T tail_index=1);
        // constructor
        CdfObject<T>(){};
        // destructor
        ~CdfObject<T>(){};
};

// ------------------------------------------------------------------
// ALIAS CLASS
template<typename T>
class cdf:public CdfObject<T>{};
// ------------------------------------------------------------------