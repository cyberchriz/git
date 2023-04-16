// author: 'cyberchriz' (Christian Suer)
// this is a neural network library for multilayer perceptrons (MLP) with flexible topologies

// preprocessor directives
#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "../../../distributions/headers/random_distributions.h"
#include "../../general/headers/enums.h"
#include "../../general/headers/weight_init.h"
#include "../../general/headers/activation_functions.h"
#include "layer.h"

// MLP network class declaration
class MLP{
    public:
        int add_layer(int neurons, OPTIMIZATION_METHOD _opt_method=ADADELTA, ACTIVATION_FUNC _activation=f_LReLU);
        void reset_weights(uint start_layer=1, uint end_layer=__INT_MAX__, double factor=1.0);
        void feedforward(uint start_layer=1, uint end_layer=__INT_MAX__);
        void backpropagate();
        void set_input(uint index, double value);
        double get_input(uint index, uint l=0){return layer[l].neuron[index].x;}
        double get_output(uint index);
        double get_hidden(uint index,uint layer_index);
        void set_label(uint index, double value);
        void autoencode();
        void save(std::string filename="network_backup.dat");
        bool network_exists(){return layers!=0;}
        double get_loss_avg();
        void set_training_mode(bool active=true){training_mode=active;}
        void set_learning_rate(double value){lr=fmin(fmax(0,value),1.0);}
        void set_learning_rate_decay(uint value){lr_decay=value;}
        void set_learning_momentum(double value){lr_momentum=fmin(fmax(0,value),1.0);}
        void set_learning_rate_auto(bool active=true){lr_auto=active;}
        void set_scaling_method(SCALING method=normalized){scaling_method=method;}
        void set_dropout(double value){dropout=fmax(0,fmin(value,1));}
        void set_recurrent(bool confirm){recurrent=confirm;}
        void set_gradient_clipping(bool active,double threshold=0.499){gradient_clipping=active;gradient_clipping_threshold=threshold;}
        double get_avg_h();
        double get_avg_output();     
        double get_lr(){return lr;}
        // constructor
        MLP(std::string filename="");
        // destructor
        ~MLP();    
    protected:
        int layers=0;
        std::vector<Layer> layer;
        void load(std::string filename);
        int backprop_iterations=0;
        SCALING scaling_method=none;
        double lr=0.005;
        double lr_momentum=0.0;
        double lr_decay=1000000;
        bool lr_auto=false;
        double opt_beta1=0.9;
        double opt_beta2=0.99;
        bool training_mode=true;
        bool gradient_clipping=false;
        double gradient_clipping_threshold=0.999;
        double dropout=0;
        bool recurrent=false;
        std::string filename;     
};

#include "../sources/mlp.cpp"