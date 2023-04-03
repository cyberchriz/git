#include <iostream>
#include <cmath>
#include "../headers/network.h"

int main(){

    // set up a test network
    Network* network = new Network("");
    
    // hyperparameters
    ACTIVATION_FUNC act_func=f_LReLU;
    OPTIMIZATION_METHOD opt=ADADELTA;
    
    // topology
    int input_neurons=5;
    int hidden_neurons=200;
    network->add_layer(input_neurons,opt,act_func);
    network->add_layer(hidden_neurons,opt,act_func);
    network->add_layer(hidden_neurons,opt,act_func);
    network->add_layer(input_neurons,opt,act_func);

    // test iterations
    for (int i=0;i<=100000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network->set_input(r,((double)rand())/RAND_MAX);
        }
        network->feedforward();
        network->autoencode();
        network->backpropagate();
        if (i<30 || i%1000==0){
            std::cout << "iteration: " << i << ", loss: " << network->get_loss_avg() << " ========================================================================================================\n";
            std::cout << "inputs:    " << network->get_input(0) << " | " << network->get_input(1) << " | " << network->get_input(2) << " | " << network->get_input(3) << " | " << network->get_input(4) << "\n";
            std::cout << "outputs:   " << network->get_output(0) << " | " << network->get_output(1) << " | " << network->get_output(2) << " | " << network->get_output(3) << " | " << network->get_output(4) << "\n";
        }
    }
    cout << "[...done]\n\n\n";
    delete network;
    return 0;
} 