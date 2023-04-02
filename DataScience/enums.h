#pragma once
#include <string>
using namespace std;

// enumeration of available activation functions for neural networks
enum ACTIVATION_FUNC
  {
   f_ident,        // identity function
   f_sigmoid,      // sigmoid (logistic)
   f_ELU,          // exponential linear unit (ELU)
   f_ReLU,         // rectified linear unit (ReLU)
   f_LReLU,        // leaky ReLU
   f_tanh,         // hyperbolic tangent (tanh)
   f_oblique_tanh, // oblique tanh (custom)
   f_tanh_rectifier,// tanh rectifier
   f_arctan,       // arcus tangent (arctan)
   f_arsinh,       // area sin. hyperbolicus (inv. hyperbol. sine)
   f_softsign,     // softsign (Elliot)
   f_ISRU,         // inverse square root unit (ISRU)
   f_ISRLU,        // inv.squ.root linear unit (ISRLU)
   f_softplus,     // softplus
   f_bentident,    // bent identity
   f_sinusoid,     // sinusoid
   f_sinc,         // cardinal sine (sinc)
   f_gaussian,     // gaussian
   f_differentiable_hardstep, // differentiable hardstep
   f_leaky_diff_hardstep, // leaky differentiable hardstep
   f_softmax,      // normalized exponential (softmax)
   f_oblique_sigmoid, // oblique sigmoid
   f_log_rectifier, // log rectifier
   f_leaky_log_rectifier, // leaky log rectifier
   f_ramp          // ramp
                   // note: the softmax function can't be part of this library because it has no single input but needs the other neurons of the layer, too, so it
  };               //       needs to be defined from within the neural network code
   

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
  