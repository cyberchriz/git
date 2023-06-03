#pragma once
#include "../../MLP/headers/mlp.h"

class Autoencoder:public MLP {
    private:
        u_char bottleneck_layer;
    public:
        double get_encoded(uint index){return get_hidden(index,bottleneck_layer);}
        double get_decoded(uint index){return get_output(index);}
        void set_encoded(uint index,double value){layer[bottleneck_layer].neuron[index].h=value;}
        void decode(){feedforward(bottleneck_layer);}
        void encode(){feedforward(1,bottleneck_layer);}
        void sweep(){feedforward();autoencode();backpropagate();}
        // delete default constructor
        Autoencoder() = delete;
        // parametric constructor
        Autoencoder(uint inputs, uint bottleneck_neurons, u_char encoder_hidden_layers=1, u_char decoder_hidden_layers=1, ACTIVATION_FUNC act_func=f_oblique_sigmoid, bool recurrent=true, double dropout=0){
            // add input layer
            add_layer(inputs,VANILLA,act_func);
            // add hidden layers for encoder section
            for (u_char n=0;n<encoder_hidden_layers;n++){
                uint neurons=ceil(0.7*(layer[layers-1].neurons+bottleneck_neurons));
                add_layer(neurons,VANILLA,act_func);
            }
            // add bottleneck layer
            add_layer(bottleneck_neurons,VANILLA,act_func);
            bottleneck_layer=encoder_hidden_layers+1;
            // add hidden layers for decoder section
            for (u_char n=0;n<decoder_hidden_layers;n++){
                uint neurons=ceil(0.7*(layer[layers-1].neurons+inputs));
                add_layer(neurons,VANILLA,act_func);
            }
            // add output layer
            add_layer(inputs,VANILLA,act_func);
            // set hyperparameters
            set_learning_rate_auto();
            set_recurrent(recurrent);
            set_dropout(dropout);
            set_training_mode(true);
        }
};

#include "../sources/autoencoder.cpp"