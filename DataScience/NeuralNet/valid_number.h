#pragma once
#include <cmath>

double vnum(double expression, bool add_randomness=true){
    // NaN
    if (expression!=expression){
        if (add_randomness){
            return __DBL_MIN__ + 10000 * __DBL_MIN__ * ((double)rand())/RAND_MAX;
        }
        else{
            return __DBL_MIN__;
        }
    }
    else if (std::isinf(expression)){
        if (expression<0){
            if (add_randomness){
                return -__DBL_MAX__ * (((double)rand())/RAND_MAX);
            }
            else{
                return -__DBL_MAX__;
            }
        }
        else {
            if (add_randomness){
                return __DBL_MAX__ * (((double)rand())/RAND_MAX);
            }
            else{
                return __DBL_MAX__;
            }
        }
    }
    return expression;
}

int vnum(int expression, bool add_randomness=true){
    // NaN
    if (expression!=expression){
        return 0;
        }
    else if (std::isinf(expression)){
        if (expression<0){
            if (add_randomness){
                return -__INT_MAX__ * (((double)rand())/RAND_MAX);
            }
            else{
                return -__INT_MAX__;
            }
        }
        else {
            if (add_randomness){
                return __INT_MAX__ * (((double)rand())/RAND_MAX);
            }
            else{
                return __INT_MAX__;
            }
        }
    }
    return expression;
}