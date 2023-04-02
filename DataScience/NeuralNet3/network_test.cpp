#include <iostream>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/network.h>
#include <cmath>


int main(){
    // set up a test network
    Network network;
    int inputs=5;
    int outputs=5;
    int hidden_layers=1;
    int hidden_width=10;
    int lower_connections=5;
    int level_connections=5;
    ACTIVATION_FUNC act_func=f_sigmoid;
    network.add_inputlayer(inputs,normalized);
    for (int n=0;n<hidden_layers;n++){
        network.add_hiddenlayer(hidden_width,lower_connections,level_connections,act_func);
    }
    network.add_outputlayer(outputs,lower_connections,level_connections,normalized);
    network.set_optimizer(Vanilla);
    network.set_learning_rate(0.0001);
    network.set_learning_rate_decay(0.001);
    network.set_learning_momentum(0);

    // test iterations
    for (int i=0;i<=100000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network.set_input(r,((double)rand())/RAND_MAX);
        }
        network.run();
        network.autoencode();
        network.backprop(false);
        // print results
        if (i<30 || i%1000==0){
            std::cout << "[iteration: " << i << "] ========================================================================================================\n";
            std::cout << "loss_avg:  " << network.get_loss_avg(0) << " | " << network.get_loss_avg(1) << " | " << network.get_loss_avg(2) << " | " << network.get_loss_avg(3) << " | " << network.get_loss_avg(4) << "\n";
            std::cout << "inputs:    " << network.get_input(0) << " | " << network.get_input(1) << " | " << network.get_input(2) << " | " << network.get_input(3) << " | " << network.get_input(4) << "\n";
            std::cout << "outputs:   " << network.get_output(0) << " | " << network.get_output(1) << " | " << network.get_output(2) << " | " << network.get_output(3) << " | " << network.get_output(4) << "\n";
        }
    }
    cout << "[...done]\n\n\n";
    return 0;
} 