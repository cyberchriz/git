// code by: Christian Suer
// to be used as helper class in conjunction with <neuralnet2.h>
#pragma once
#include "network.h"
#include "../../array.h"

class Neuron{
    private:
    protected:
    public:
        // core parameters
        bool dropout=false;             // dropout for overtraining mitigation
        double x=0;                     // receives sum of weighted inputs
        double h=0;                     // for result of activation function (=output of hidden neurons; also used for input value of initial layer)
        double scaled_label;            // used for output error calculation
        double output;                  // used for rescaled output (rescaling of h)
        int* input_shape;
        int* layer_shape;
        bool self_attention;
        bool recurrent;
        int input_dimensions;
        int layer_dimensions;
        int inputs;        
        OPTIMIZATION_METHOD opt_method;

        // recurrent inputs ("memory")    
        double m1=0;                    // stores h value from last iteration
        double m2=0;                    // stores rolling average with ratio 9:1 previous average versus new value for h
        double m3=0;                    // stores rolling average with ratio 99:1 previous average versus new value for h
        double m4=0;                    // stores rolling average with ratio 999:1 previous average versus new value for h

        // error calculations
        double gradient=0;              // used for output error (derivative of loss function) or hidden error, depending on the layer
        double loss=0;                  // for MSE loss
        double loss_sum=0;              // used for average loss (after dividing by number of backprop iterations) 

        // values for input and label scaling
        double input_min=__DBL_MAX__;   // used for minmax scaling
        double input_max=-__DBL_MAX__;   // used for minmax scaling
        double input_maxabs=__DBL_EPSILON__;// used for input scaling
        double input_rolling_average = 0; // used for input scaling
        double input_mdev2 = 0;         // used for input scaling
        double input_variance = 1;      // used for input scaling, calculated as approximation from rolling mdev2
        double input_stddev=1;          // used for standardized scaling
        double label;                   // true/target values (before scaling)
        double label_min=__DBL_MAX__;   // used for label minmax scaling
        double label_max=-__DBL_MAX__;  // used for label minmax scaling
        double label_maxabs=__DBL_EPSILON__;// used for label scaling
        double label_rolling_average = 0; // used for label scaling
        double label_mdev2 = 0;         // used for label scaling
        double label_variance = 1;      // used for label scaling, calculated as approximation from rolling mdev2
        double label_stddev =1;         // used for standardized label scaling

        // weight matrix
        Array* input_weight;
        Array* input_weight_delta;
        Array* self_weight;
        Array* self_weight_delta;
        Array* opt_v;                   // used for RMSprop, ADADELTA and ADAM
        Array* opt_w;                   // used for ADADELTA and ADAM
        
        // weights and weight-deltas for recurrent inputs
        double m1_weight;
        double m2_weight;
        double m3_weight;
        double m4_weight;
        double delta_m1=0;
        double delta_m2=0;
        double delta_m3=0;
        double delta_m4=0;    

        // bias weight
        double bias_weight;
        double delta_b=0;                 // calculated bias correction from backpropagation  

        // constructor(s) declaration
        Neuron(int* _input_shape, int* _layer_shape=nullptr, OPTIMIZATION_METHOD _opt_method=Vanilla, bool _self_attention=false, bool _recurrent=false);
        
        // destructor declaration
        ~Neuron();
};

// constructor definition
Neuron::Neuron(int* _input_shape, int* _layer_shape, OPTIMIZATION_METHOD _opt_method, bool _self_attention, bool _recurrent){
    // make a local copy of the input shape
    input_dimensions = sizeof(_input_shape) / __SIZEOF_POINTER__;
    input_shape = new int[input_dimensions];
    for (int i=0;i<input_dimensions;i++){
        input_shape[i]=_input_shape[i];
    }
    // count total inputs
    inputs=input_shape[0];
    for (int i=1;i<input_dimensions;i++){
        inputs*=input_shape[i];
    }
    // declare input weight vector
    input_weight = new Array(input_shape);
    input_weight_delta = new Array(input_shape);
    input_weight_delta->fill(0.0);
    opt_method=_opt_method;
    if (opt_method==RMSprop || opt_method==ADADELTA || opt_method==ADAM || opt_method==AdaGrad){
        opt_v = new Array(input_shape);
        opt_v->fill(0.0);
    }
    if (opt_method==ADADELTA || opt_method==ADAM){
        opt_w = new Array(input_shape);
        opt_w->fill(0.0);
    }
    // make a local copy of the layer shape
    layer_dimensions = sizeof(_layer_shape) / __SIZEOF_POINTER__;
    layer_shape = new int[layer_dimensions];
    for (int i=0;i<layer_dimensions;i++){
        layer_shape[i] = _layer_shape[i];
    }
    // declare self_weights (=weights from self-attention with regard to other neurons from local layer)
    self_attention = _self_attention;
    if (self_attention){
        self_weight = new Array(layer_shape);
        self_weight_delta = new Array(layer_shape);
        self_weight_delta->fill(0.0);
    }
    // declare recurrent weights
    recurrent = _recurrent;
}

// destructor definition
Neuron::~Neuron(){
    delete input_shape;
    delete layer_shape;
    delete input_weight;
    delete input_weight_delta;
    if (opt_method==RMSprop || opt_method==ADADELTA || opt_method==ADAM){
        delete opt_v;
    }
    if (opt_method==ADADELTA || opt_method==ADAM){
        delete opt_w;
    }
    if (self_attention){
        delete self_weight;
        delete self_weight_delta;
    }
}
