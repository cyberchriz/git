// author: 'cyberchriz' (Christian Suer)

#pragma once
#include "network.h"
#include "neuron.h"
#include <vector>

class Layer{
    public:
        int neurons;
        OPTIMIZATION_METHOD opt_method;
        ACTIVATION_FUNC activation;
        int layer_dimensions;
        int input_dimensions;
        vector<Neuron> neuron;   
        // constructor
        Layer(int neurons, int inputs_per_neuron, OPTIMIZATION_METHOD _opt_method=Vanilla, ACTIVATION_FUNC _activation=f_tanh){
            this->neurons=neurons;
            this->opt_method=_opt_method;
            this->activation=_activation;
            // setup neurons
            neuron.reserve(neurons);
            for (int i=0;i<neurons;i++){
                neuron.push_back(sizeof(Neuron(inputs_per_neuron)));
            }
        }
        // destructor
        ~Layer(){}
};