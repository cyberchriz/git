#include <iostream>
#include <cmath>
#include "../headers/network.h"

int main(){

    // set up a test network
    Network network = Network("");
    
    // hyperparameters
    network.set_learning_rate(0.01);
    network.set_learning_momentum(0.0);
    network.set_learning_rate_decay(0.0001);
    network.set_recurrent(false);
    network.set_dropout(0);
    network.set_training_mode(true);
    ACTIVATION_FUNC act_func=f_oblique_sigmoid;
    OPTIMIZATION_METHOD opt=Nesterov;
      
    // topology
    int input_neurons=5;
    int hidden_neurons=10;
    network.add_layer(input_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(input_neurons,opt,act_func);

    // test iterations
    for (int i=0;i<=10000000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network.set_input(r,((double)rand())/RAND_MAX);
        }
        network.feedforward();
        network.autoencode();
        network.backpropagate();
        if (i<30 || i%100000==0){
            std::cout << "iteration: " << i << ", loss: " << network.get_loss_avg() << " ========================================================================================================\n";
            std::cout << "inputs:    " << network.get_input(0) << " | " << network.get_input(1) << " | " << network.get_input(2) << " | " << network.get_input(3) << " | " << network.get_input(4) << "\n";
            std::cout << "outputs:   " << network.get_output(0) << " | " << network.get_output(1) << " | " << network.get_output(2) << " | " << network.get_output(3) << " | " << network.get_output(4) << "\n";
        }
    }
    std::cout << "[...done]\n\n\n";
    return 0;
} 