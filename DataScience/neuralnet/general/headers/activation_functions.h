#pragma once
#include <string>
#include <cmath>

// enumeration of available activation functions for neural networks
enum ACTIVATION_FUNC {
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

//+------------------------------------------------------------------+
//|      return act.function as string variable equivalent           |
//+------------------------------------------------------------------+
std::string actfunct_string(ACTIVATION_FUNC f);

// +------------------------------------------------------------------+
// |       function Activate / DeActivate                             |
// +------------------------------------------------------------------+
double activate(double x, ACTIVATION_FUNC f);
double deactivate(double x, ACTIVATION_FUNC f);



#include "../sources/activation_functions.cpp"