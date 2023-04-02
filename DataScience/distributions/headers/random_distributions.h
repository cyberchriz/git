//+------------------------------------------------------------------+
//|      random values from a given distribution                     |
//+------------------------------------------------------------------+

// author: Christian Suer


template<typename T>
class Random {
    private:
        static bool seeded=false;
    public:    
        static T normal(T mu=0,T sigma=1);
        static T cauchy(T x_peak=0,T gamma=1);
        static T uniform(T min=0, T max=1);
        static T laplace(T mu=0,T sigma=1);
        static T pareto(T alpha=1,T tail_index=1);
        static T lomax(T alpha=1,T tail_index=1);
        static T binary();     
        static T sign();
};

// ALIAS CLASS
template<typename T> class rnd:Random<T>{};