// functions for NaN/Inf error mitigation

// author: 'cyberchriz' (Christian Suer)


// preprocessor directives
#pragma once
#include <cmath>


// =============================================
// replace expression with given alternative
// =============================================

double validate(double expression, double alternative){
    if (std::isnan(expression) || std::isinf(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

long double validate(long double expression, long double alternative){
    if (std::isnan(expression) || std::isinf(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

float validate(float expression, float alternative){
    if (isnanf(expression) || isinff(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

int validate(int expression, int alternative){
    if (std::isnan(expression) || std::isinf(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

long validate(long expression, long alternative){
    if (std::isnan(expression) || std::isinf(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

// =============================================
// replace expression with automatic alternative
// =============================================

double validate(double expression){
    if (expression!=expression){return __DBL_MIN__;}
    if (std::isinf(expression)){
        if (expression>__DBL_MAX__){
            return __DBL_MAX__;
        }
        else {
            return -__DBL_MAX__;
        }
    }
}

long double validate(long double expression){
    if (expression!=expression){return __LDBL_MIN__;}
    if (std::isinf(expression)){
        if (expression>__LDBL_MAX__){
            return __LDBL_MAX__;
        }
        else {
            return -__LDBL_MAX__;
        }
    }
}

float validate(float expression){
    if (expression!=expression){return __FLT_MIN__;}
    if (std::isinf(expression)){
        if (expression>__FLT_MAX__){
            return __FLT_MAX__;
        }
        else {
            return -__FLT_MAX__;
        }
    }
}

int validate(int expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){
        if (expression>__INT_MAX__){
            return __INT_MAX__;
        }
        else {
            return -__INT_MAX__;
        }
    }
}

uint validate(uint expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){return __INT_MAX__;}
}

long validate(long expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){
        if (expression>__LONG_MAX__){
            return __LONG_MAX__;
        }
        else {
            return -__LONG_MAX__;
        }
    }
}

u_char validate(u_char expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){return 255;}
}

char validate(char expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){
        if (expression>127){
            return 127;
        }
        else {
            return -127;
        }
    }
}

short validate(short expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){
        if (expression>__SHRT_MAX__){
            return __SHRT_MAX__;
        }
        else {
            return -__SHRT_MAX__;
        }
    }
}

ushort validate(ushort expression){
    if (expression!=expression){return 0;}
    if (std::isinf(expression)){return 2*__SHRT_MAX__;}
}

// =============================================
// pass by reference, with automatic alternative
// =============================================

void validate_r(double& expression){
    if (expression!=expression){expression = __DBL_MIN__;}
    if (std::isinf(expression)){
        if (expression>__DBL_MAX__){
            expression = __DBL_MAX__;
        }
        else {
            expression = -__DBL_MAX__;
        }
    }
}

void validate_r(long double& expression){
    if (expression!=expression){expression= __LDBL_MIN__;}
    if (std::isinf(expression)){
        if (expression>__LDBL_MAX__){
            expression = __LDBL_MAX__;
        }
        else {
            expression = -__LDBL_MAX__;
        }
    }
}

void validate_r(float& expression){
    if (expression!=expression){expression = __FLT_MIN__;}
    if (std::isinf(expression)){
        if (expression>__FLT_MAX__){
            expression = __FLT_MAX__;
        }
        else {
            expression = -__FLT_MAX__;
        }
    }
}

void validate_r(int& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){
        if (expression>__INT_MAX__){
            expression = __INT_MAX__;
        }
        else {
            expression = -__INT_MAX__;
        }
    }
}

void validate_r(uint& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){expression = 2* __INT_MAX__;}
}

long validate_r(long& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){
        if (expression>__LONG_MAX__){
            expression = __LONG_MAX__;
        }
        else {
            expression = -__LONG_MAX__;
        }
    }
}

void validate_r(u_char& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){expression = 255;}
}

void validate_r(char& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){
        if (expression>127){
            expression = 127;
        }
        else {
            expression = -127;
        }
    }
}

void validate_r(short& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){
        if (expression>__SHRT_MAX__){
            expression = __SHRT_MAX__;
        }
        else {
            expression = -__SHRT_MAX__;
        }
    }
}

void validate_r(ushort& expression){
    if (expression!=expression){expression = 0;}
    if (std::isinf(expression)){expression = 2*__SHRT_MAX__;}
}