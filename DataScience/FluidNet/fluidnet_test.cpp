#include <iostream>
#include </home/christian/Documents/own_code/c++/DataScience/FluidNet/fluidnet.h>
#include <cmath>


int main(){
    // set up a test network
    int inputs=5;
    int outputs=5;
    int connections=10;
    FluidNet *network = new FluidNet(inputs, outputs, connections);
    network->set_optimizer(Vanilla);
    network->set_learning_rate(0.01);
    network->set_learning_rate_decay(0.0001);
    network->set_learning_momentum(0.9);
    network->add_neurons(1000,connections,f_LReLU);

    // test iterations
    for (int i=0;i<=100000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network->set_input(r,((double)rand())/RAND_MAX);
        }
        network->run();
        network->autoencode();
        network->backprop();
        // print results
        if (i<30 || i%1000==0){
            std::cout << "[iteration: " << i << "] ========================================================================================================\n";
            std::cout << "loss_avg:  " << network->get_loss_avg(0) << " | " << network->get_loss_avg(1) << " | " << network->get_loss_avg(2) << " | " << network->get_loss_avg(3) << " | " << network->get_loss_avg(4) << "\n";
            std::cout << "inputs:    " << network->get_input(0) << " | " << network->get_input(1) << " | " << network->get_input(2) << " | " << network->get_input(3) << " | " << network->get_input(4) << "\n";
            std::cout << "outputs:   " << network->get_output(0) << " | " << network->get_output(1) << " | " << network->get_output(2) << " | " << network->get_output(3) << " | " << network->get_output(4) << "\n";
        }
    }
    cout << "[...done]\n\n\n";
    delete network;
    return 0;
} 