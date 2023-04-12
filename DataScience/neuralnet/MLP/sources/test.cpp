#include <iostream>
#include <cmath>
#include "../headers/mlp.h"
#include "../../autoencoder/headers/autoencoder.h"

int main(){
    /*
    // set up a test network
    Autoencoder ae = Autoencoder(5,5,1,1,f_LReLU,false,0);
 
    // test iterations 
    for (int i=0;i<=10000000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            ae.set_input(r,((double)rand())/RAND_MAX);
        }
        ae.sweep();
        if (i<30 || i%100000==0){
            std::cout << "iteration: " << i << ", loss: " << ae.get_loss_avg() <<  ========================================================================================================\n";
            std::cout << "inputs:    " << ae.get_input(0) << " | " << ae.get_input(1) << " | " << ae.get_input(2) << " | " << ae.get_input(3) << " | " << ae.get_input(4) << "\n";
            std::cout << "outputs:   " << ae.get_decoded(0) << " | " << ae.get_decoded(1) << " | " << ae.get_decoded(2) << " | " << ae.get_decoded(3) << " | " << ae.get_decoded(4) << "\n";
        }
    }
    std::cout << "[...done]\n\n\n"; 
    */   


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
        network.feedforward();
        network.autoencode();
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