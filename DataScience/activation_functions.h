#pragma once
#include <string>
#include <cmath>
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

//+------------------------------------------------------------------+
//|      return act.function as string variable equivalent           |
//+------------------------------------------------------------------+
string actfunct_string(ACTIVATION_FUNC f)
  {
   switch(f)
     {
      case f_ident:     return "identity";
      case f_sigmoid:   return "sigmoid (logistic)";
      case f_oblique_sigmoid: return "oblique sigmoid (custom)";
      case f_ELU:       return "exponential linear unit (ELU)";
      case f_ReLU:      return "rectified linear unit (ReLU)";
      case f_tanh:      return "hyperbolic tangent (tanh)";
      case f_oblique_tanh: return "oblique hyperbolic tangent (f(x)=tanh(x)+0.01x)";
      case f_tanh_rectifier: return "tanh rectifier (custom, x<0: 0.01*tanh(x); x>=0: tanh(x)+0.01x)";
      case f_arctan:    return "arcus tangent (arctan)";
      case f_arsinh:    return "area sinus hyperbolicus (inv. hyperbol.sine)";
      case f_softsign:  return "softsign";
      case f_ISRU:      return "inverse square root unit (ISRU)";
      case f_ISRLU:     return "inverse square root linear unit (ISRLU)";
      case f_softplus:  return "softplus";
      case f_bentident: return "bent identity";
      case f_sinusoid:  return "sinusoid";
      case f_sinc:      return "cardinal sine (sinc)";
      case f_gaussian:  return "gaussian";
      case f_differentiable_hardstep: return "differentiable hardstep";
      case f_leaky_diff_hardstep: return "leaky differentiable hardstep";
      case f_softmax:   return "normalized exponential (softmax)";
      case f_log_rectifier: return "log rectifier (x<=0: 0, x>0: ln(x+1))";
      case f_leaky_log_rectifier: return "leaky log rectifier (x<=0: 0.01x, x>0: ln(x+1))";
      case f_ramp:      return "ramp";
      default:          return "";
     }
  }

// mitigate NaN/Inf errors
double ValidNumber(double expression, double alternative){
    if (isnan(expression) || isinf(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

float ValidNumber(float expression, float alternative){
    if (isnanf(expression) || isinff(expression)){
        return alternative;
    }
    else{
        return expression;
    }
}

// +------------------------------------------------------------------+
// |       function ELU / ELU_drv                                     |
// +------------------------------------------------------------------+
double ELU(double z)
  {
   if (z>0)
     {return z;}
   else
     {
      double alpha=0.01;
      return ValidNumber(alpha*(exp(z)-1),__DBL_EPSILON__-alpha);
     }
  }   

double ELU_drv(double z)
  {
   if (z>0)
     {return 1;} 
   else
     {
      double alpha=0.01;
      return ValidNumber(alpha*exp(z),__DBL_MIN__);
     }
  }   

// +------------------------------------------------------------------+
// |       function sigmoid / sigmoid_drv                             |
// +------------------------------------------------------------------+   
 double sigmoid(double z)
  {
   if (z>0)
     {return ValidNumber(1/(1+exp(-z)),1-__DBL_EPSILON__);}
   else
     {return ValidNumber(1/(1+exp(-z)),__DBL_MIN__);}
  }

double sigmoid_drv(double z)
  {
   return ValidNumber(exp(z)/(pow(exp(z)+1,2)),__DBL_MIN__);
  }   

// +------------------------------------------------------------------+
// |       function oblique sigmoid / obl. sigmoid_drv (custom)       |
// +------------------------------------------------------------------+   
 double oblique_sigmoid(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return ValidNumber(1/(1+exp(-z)),1-__DBL_EPSILON__)+alpha*z;}
   else
     {return ValidNumber(1/(1+exp(-z)),__DBL_EPSILON__)+alpha*z;}
  }

double oblique_sigmoid_drv(double z)
  {
   double alpha=0.01;
   return ValidNumber(exp(z)/(pow(exp(z)+1,2)),__DBL_EPSILON__)+alpha;
  } 

// +------------------------------------------------------------------+
// |       function ReLU / ReLU_drv                                   |
// +------------------------------------------------------------------+      
double ReLU(double z)
  {
   return fmax(z,0);  
  }
      
double ReLU_drv(double z)
  {
   return z>0;
  }

// +------------------------------------------------------------------+
// |       function LReLU / LReLU_drv                                 |
// +------------------------------------------------------------------+
double LReLU(double z)
  {
   if (z>0)
     {return z;}
   else
     {return 0.001*z;}
  }
      
double LReLU_drv(double z)
  {
   if (z>0)
     {return 1;}
   else
     {return 0.001;}
  }
    
// +------------------------------------------------------------------+
// |       function modtanh / modtanh_drv                             |
// +------------------------------------------------------------------+
double modtanh(double z)
  {
   if (z>0)
     {return ValidNumber(tanh(z),1-__DBL_EPSILON__);}
   else
     {return ValidNumber(tanh(z),__DBL_EPSILON__-1);}
  }

double modtanh_drv(double z)
  {
   return ValidNumber(1-pow(tanh(z),2),__DBL_MIN__);
  }

// +------------------------------------------------------------------+
// |       function oblique_tanh / oblique_tanh_drv (custom)          |
// +------------------------------------------------------------------+
double oblique_tanh(double z)
  {
   double alpha=0.01;  
   if (z>0)
     {return ValidNumber(tanh(z),1-__DBL_EPSILON__)+alpha*z;}
   else
     {return ValidNumber(tanh(z),__DBL_EPSILON__-1)+alpha*z;}
  }
     
double oblique_tanh_drv(double z)
  {
   double alpha=0.01;
   return ValidNumber(1-pow(tanh(z),2),__DBL_EPSILON__)+alpha;
  }
  
// +------------------------------------------------------------------+
// |       function tanh rectifier / _drv (custom)                    |
// +------------------------------------------------------------------+
double tanh_rectifier(double z)
  {
   double alpha=0.01;  
   if (z>=0)
     {return ValidNumber(tanh(z),1-__DBL_EPSILON__)+alpha*z;}
   else
     {return ValidNumber(alpha*tanh(z),__DBL_EPSILON__-1)+alpha*z;}
  }
     
double tanh_rectifier_drv(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return ValidNumber(1-pow(tanh(z),2),__DBL_EPSILON__)+alpha;}
   else
     {return ValidNumber(alpha*(1-pow(tanh(z),2))+alpha,__DBL_EPSILON__+alpha);}
  }  
  
//+------------------------------------------------------------------+
//|      function arctan / arctan_drv                                |
//+------------------------------------------------------------------+
double arctan(double z)
  {
   if (z>0)
     {return ValidNumber(atan(z),0.5*M_PI);}
   else
     {return ValidNumber(atan(z),-0.5*M_PI);}
  }
  
double arctan_drv(double z)
  {
   return ValidNumber(1/(pow(z,2)+1),__DBL_MIN__);
  }

//+------------------------------------------------------------------+
//|      function arsinh / arsinh_drv                                |
//+------------------------------------------------------------------+
double arsinh(double z)
  {
   if (z>0)
     {return ValidNumber(asinh(z),__DBL_MAX__);}
   else
     {return ValidNumber(asinh(z),-__DBL_MAX__);}
  }
  
double arsinh_drv(double z)
  {
   return ValidNumber(1/sqrt(pow(z,2)+1),__DBL_MIN__);
  }

//+------------------------------------------------------------------+
//|      function softsign / softsign_drv                            |
//+------------------------------------------------------------------+
double softsign(double z)
  {
   if (z>0)
     {return ValidNumber(z/(1+fabs(z)),1-__DBL_EPSILON__);}
   else
     {return ValidNumber(z/(1+fabs(z)),__DBL_EPSILON__-1);}
  }
  
double softsign_drv(double z)
  {
   return ValidNumber(1/pow(1+fabs(z),2),__DBL_MIN__);
  }

//+------------------------------------------------------------------+
//|      function ISRU / ISRU_drv                                    |
//+------------------------------------------------------------------+
double ISRU(double z)
  {
   double alpha=1;
   if (z>0)
     {return ValidNumber(z/sqrt(1+alpha*pow(z,2)),1/sqrt(alpha)-__DBL_EPSILON__);}
   else
     {return ValidNumber(z/sqrt(1+alpha*pow(z,2)),__DBL_EPSILON__-1/sqrt(alpha));}
  }
  
double ISRU_drv(double z)
  {
   double alpha=1;
   return ValidNumber(pow(1/sqrt(1+alpha*pow(z,2)),3),__DBL_MIN__);
  }

//+------------------------------------------------------------------+
//|      function ISRLU / ISRLU_drv                                  |
//+------------------------------------------------------------------+
// note: ISRLU="inverse square root linear unit"
double ISRLU(double z)
  {
   double alpha=1;
   if (z<0)
     {return ValidNumber(z/sqrt(1+alpha*pow(z,2)),__DBL_EPSILON__-1/sqrt(alpha));}
   else
     {
      return z;
     }
  }
  
double ISRLU_drv(double z)
  {
   double alpha=1;
   if (z<0)
     {return ValidNumber(pow(1/sqrt(1+alpha*pow(z,2)),3),__DBL_MIN__);}
   else
     {return 1;}
  }

//+------------------------------------------------------------------+
//|      function softplus / softplus_drv                            |
//+------------------------------------------------------------------+
double softplus(double z)
  {
   if (z>0)
     {return ValidNumber(log(1+exp(z)),__DBL_MAX__);}
   else
     {return ValidNumber(log(1+exp(z)),__DBL_MIN__);}
  }
  
double softplus_drv(double z)
  {
   if (z>0)
     {return 1/(1+exp(-z));} //all positive results are valid (gradient is close to 1)
   else
     {return ValidNumber(1/(1+exp(-z)),__DBL_MIN__);}
  }

//+------------------------------------------------------------------+
//|      function bentident / bentident_drv                          |
//+------------------------------------------------------------------+
double bentident(double z)
  {
   if (z>0)
     {return ValidNumber((sqrt(pow(z,2)+1)-1)/2+z,__DBL_MAX__);}
   else
     {return ValidNumber((sqrt(pow(z,2)+1)-1)/2+z,-__DBL_MAX__);}
  }
  
double bentident_drv(double z)
  {
   return z/(2*sqrt(pow(z,2)+1))+1;
  }

//+------------------------------------------------------------------+
//|      function sinusoid / sinusoid_drv                            |
//+------------------------------------------------------------------+
double sinusoid(double z)
  {
   return sin(z);
  }
  
double sinusoid_drv(double z)
  {
   return cos(z);
  }

//+------------------------------------------------------------------+
//|      function sinc / sinc_drv                                    |
//+------------------------------------------------------------------+
double sinc(double z)
  {
   if (z==0)
     {return 1;}
   else
     {return ValidNumber(sin(z)/z,__DBL_MIN__);}
  }
  
double sinc_drv(double z)
  {
   if (z==0)
     {return 0;}
   else
     {
      if (z>0)
        {return ValidNumber(cos(z)/z-sin(z)/pow(z,2),-__DBL_MIN__);}
      else
        {return ValidNumber(cos(z)/z-sin(z)/pow(z,2),__DBL_MIN__);} 
     }
  }     

//+------------------------------------------------------------------+
//|      function gaussian / gaussian_drv                            |
//+------------------------------------------------------------------+
double gaussian(double z)
  {
   return ValidNumber(exp(-pow(z,2)),__DBL_MIN__);
  }
  
double gaussian_drv(double z)
  {
   if (z>0)
     {return ValidNumber(-2*z*pow(M_E,-pow(z,2)),-__DBL_MIN__);}
   else
     {return ValidNumber(-2*z*pow(M_E,-pow(z,2)),__DBL_MIN__);}
  }

//+------------------------------------------------------------------+
//|      function differentiable hardstep (custom) / _drv            |
//+------------------------------------------------------------------+
double diff_hardstep(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return ValidNumber(1+alpha*z,__DBL_MAX__);}
   else
     {return 0;}
  }
double diff_hardstep_drv(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return alpha;}
   else
     {return 0;}
  }
  
//+------------------------------------------------------------------+
//|      function leaky differentiable hardstep (custom) / _drv      |
//+------------------------------------------------------------------+
double inv_diff_hardstep(double z)
  {
   double alpha=0.01;
   if (z>=0)
     {return ValidNumber(1+alpha*z,__DBL_MAX__);}
   else
     {return ValidNumber(alpha*z,-__DBL_MAX__);}
  }
double inv_diff_hardstep_drv()
  {
   double alpha=0.01;
   return alpha;
  }  
  
//+------------------------------------------------------------------+
//|      function log rectifier (custom) / _drv                      |
//+------------------------------------------------------------------+
double log_rectifier(double z)
  {
   if (z>0)
     {return log(z+1);}
   else
     {return 0;}
  }
  
double log_rectifier_drv(double z)
  {
   if (z>0)
     {return ValidNumber(1/(z+1),__DBL_MIN__);}
   else
     {return 0;}
  }
  
//+------------------------------------------------------------------+
//|      function leaky log rectifier (custom) / _drv                |
//+------------------------------------------------------------------+
double leaky_log_rectifier(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return log(z+1);}
   else
     {return alpha*z;}
  }
  
double leaky_log_rectifier_drv(double z)
  {
   double alpha=0.01;
   if (z>0)
     {return ValidNumber(1/(z+1),__DBL_MIN__);}
   else
     {return alpha;}
  }  
  
//+------------------------------------------------------------------+
//|      function ramp / ramp_drv                                    |
//+------------------------------------------------------------------+
double ramp(double z)
  {
   if (z>1){return 1;}
   if (z>-1){return z;}
   return -1;
  }
  
double ramp_drv(double z)
  {
   if (z>1){return 0;}
   if (z>-1){return 1;}
   return 0;
  }

// +------------------------------------------------------------------+
// |       function Activate / DeActivate                             |
// +------------------------------------------------------------------+
double activate(double x, ACTIVATION_FUNC f)
  {
   switch(f)
     {
      case f_ident:     return x;
      case f_sigmoid:   return sigmoid(x);
      case f_ELU:       return ELU(x);
      case f_ReLU:      return ReLU(x);
      case f_tanh:      return modtanh(x);
      case f_oblique_tanh: return oblique_tanh(x);
      case f_tanh_rectifier: return tanh_rectifier(x);
      case f_arctan:    return arctan(x);
      case f_arsinh:    return arsinh(x);
      case f_softsign:  return softsign(x);
      case f_ISRU:      return ISRU(x);
      case f_ISRLU:     return ISRLU(x);
      case f_softplus:  return softplus(x);
      case f_bentident: return bentident(x);
      case f_sinusoid:  return sinusoid(x);
      case f_sinc:      return sinc(x);
      case f_gaussian:  return gaussian(x);
      case f_differentiable_hardstep: return diff_hardstep(x);
      case f_leaky_diff_hardstep: return inv_diff_hardstep(x);
      case f_oblique_sigmoid: return oblique_sigmoid(x);
      case f_log_rectifier: return log_rectifier(x);
      case f_leaky_log_rectifier: return leaky_log_rectifier(x);
      case f_ramp:      return ramp(x);
      default:          return x; //="identity function"
     }
  }

double deactivate(double x, ACTIVATION_FUNC f)
  {
   switch(f)
     {
      case f_ident:     return 1;
      case f_sigmoid:   return sigmoid_drv(x);
      case f_ELU:       return ELU_drv(x);
      case f_ReLU:      return ReLU_drv(x);
      case f_tanh:      return modtanh_drv(x);
      case f_oblique_tanh: return oblique_tanh_drv(x);
      case f_tanh_rectifier: return tanh_rectifier_drv(x);
      case f_arctan:    return arctan_drv(x);
      case f_arsinh:    return arsinh_drv(x);
      case f_softsign:  return softsign_drv(x);
      case f_ISRU:      return ISRU_drv(x);
      case f_ISRLU:     return ISRLU_drv(x);
      case f_softplus:  return softplus_drv(x);
      case f_bentident: return bentident_drv(x);
      case f_sinusoid:  return sinusoid_drv(x);
      case f_sinc:      return sinc_drv(x);
      case f_gaussian:  return gaussian_drv(x);
      case f_differentiable_hardstep: return diff_hardstep_drv(x);
      case f_leaky_diff_hardstep: return inv_diff_hardstep_drv();
      case f_oblique_sigmoid: return oblique_sigmoid_drv(x);
      case f_log_rectifier: return log_rectifier_drv(x);
      case f_leaky_log_rectifier: return leaky_log_rectifier_drv(x);
      case f_ramp:      return ramp_drv(x);
      default:          return 1; //="identity function" derivative
     }
  }