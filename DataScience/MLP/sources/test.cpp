#include <iostream>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet2/neuralnet2.h>
#include <cmath>
#include <jsoncpp/json/json.h>
#include <curl/curl.h>
#include <iex-cpp-client/IEX.h>
#include <iex-cpp-client/IEXCloud.h>
#include <iex-cpp-client/IEXRequest.h>
#include <iex-cpp-client/IEXResponse.h>

int main(int argc, char *argv[]){
    IEX::Stock stock;
    IEX::PriceData price = stock.getPrice("GOOG");
    double latest = price.latest_price;
    std::cout << "google latest price: " << latest <<"\n";

    // set up a test network
    Network* network = new Network("");
    
    // hyperparameters
    ACTIVATION_FUNC act_func=f_LReLU;
    OPTIMIZATION_METHOD opt=ADADELTA;
    bool self_attention=false;
    bool recurrent=false;
    
    // topology
    int input_shape[]={5};
    network->add_layer(input_shape,opt,act_func,self_attention,recurrent);
    int hidden_shape[]={100,100};
    network->add_layer(hidden_shape,opt,act_func,self_attention,recurrent);
    network->add_layer(hidden_shape,opt,act_func,self_attention,recurrent);
    network->add_layer(input_shape,opt,act_func,self_attention,recurrent);

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