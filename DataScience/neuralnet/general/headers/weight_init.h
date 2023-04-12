#pragma once
#include "../../../distributions/headers/random_distributions.h"
#include <cmath>

// weight initialization methods for neural networks
// author: 'cyberchriz' (Christian Suer)



// normal "Xavier" weight initialization (by Xavier Glorot & Bengio) for tanh activation
double f_Xavier_normal(int fan_in, int fan_out);

// uniform "Xavier" weight initializiation (by Xavier Glorot & Bengio) for tanh activation
double f_Xavier_uniform(int fan_in, int fan_out);

// uniform "Xavier" weight initialization for sigmoid activation
double f_Xavier_sigmoid(int fan_in, int fan_out);

// "Kaiming He" normal weight initialization, used for ReLU activation
double f_He_ReLU(int fan_in);

// modified "Kaiming He" nornal weight initialization, used for ELU activation
double f_He_ELU(int fan_in);



#include"../sources/weight_init.cpp"