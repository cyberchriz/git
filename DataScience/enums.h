#pragma once
#include <string>
   
// available types of neural network layers
enum LAYER_TYPE {
    standard,
    recurrent,
    mod_recurrent,
    lstm,
    gru,
    convolutional,
    pooling
};

// neural network weight initialization methods
enum WEIGHT_INIT
  {
   Xavier_normal=1,     // Xavier Glorot normal
   Xavier_uniform=2,    // Xavier Glorot uniform
   Sigmoid_uniform=3,   // sigmoid uniform
   Kaiming_ReLU=4,      // Kaiming He (ReLU)
   Kaiming_ELU=5,       // Kaiming He (ELU)
   Kaiming_uniform=6,   // Kaiming He uniform
   custom_uniform=7     // custom uniform
  };
  
// neural network feature and label scaling methods
enum SCALING
  {
   none,              // no scaling
   standardized,      // standard deviation method (µ=0, sigma=1)
   normalized,        // minmax, range 0 to 1
   maxabs             // minmax, range -1 to +1
  };

// neural network optimization method
enum OPTIMIZATION_METHOD
  {
   Vanilla=1,           // Vanilla Stochastic Gradient Descent
   Nesterov=2,          // Nesterov Accelerated Gradient (NAG)
   RMSprop=3,           // RMSprop
   ADADELTA=4,          // ADADELTA
   ADAM=5,              // ADAM
   AdaGrad=6            // AdaGrad
  };
  
// neural network loss functions
enum LOSS_FUNCTION
  {
   MSE=1,                                                         // mean squared error (MSE)
   CategoricalCrossEntropy=2,                                     // categorical cross entropy
   BinaryCrossEntropy=3,                                          // binary cross entropy
   MAE=4                                                          // mean absolute error (MAE)
  };
  