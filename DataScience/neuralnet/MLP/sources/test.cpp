#include <iostream>
#include <cmath>
#include "../headers/mlp.h"
#include "../../autoencoder/headers/autoencoder.h"

int main(){
    // set up a test network
    MLP network = MLP("");
    
    // hyperparameters
    network.set_learning_rate(0.005);
    network.set_learning_rate_auto(true);
    network.set_recurrent(false);
    network.set_dropout(0);
    network.set_training_mode(true);
    ACTIVATION_FUNC act_func=f_LReLU;
    OPTIMIZATION_METHOD opt=Vanilla;
      
    // topology
    int input_neurons=5;
    int hidden_neurons=50;
    network.add_layer(input_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(input_neurons,opt,act_func);
    network.reset_weights(1,__INT_MAX__,1);

    // test iterations
    for (int i=0;i<=10000000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network.set_input(r,((double)rand())/RAND_MAX);
        }
        // feed inputs through network
        network.feedforward();
        // trying to recreate the inputs as a simple test to confirm the program is calculating correctly
        network.autoencode();
        // update weights according to their contribution to the output error
        network.backpropagate();
        if (i<30 || i%100000==0){
            std::cout << "==============================================================\n";
            std::cout << "iteration: " << i << ", loss: " << network.get_loss_avg() << ", effective learning rate: " << network.get_lr() << "\n";
            std::cout << "inputs:    " << network.get_input(0) << " | " << network.get_input(1) << " | " << network.get_input(2) << " | " << network.get_input(3) << " | " << network.get_input(4) << "\n";
            std::cout << "outputs:   " << network.get_output(0) << " | " << network.get_output(1) << " | " << network.get_output(2) << " | " << network.get_output(3) << " | " << network.get_output(4) << "\n";
        }
    }
    std::cout << "[...done]\n\n\n";

    return 0;
} 