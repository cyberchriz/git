#include <iostream>
#include <neuralnet.h>
#include <cmath>
using namespace std;

int main(int argc, char *argv[]){

    // set up a test network
    NeuralNet* network = new NeuralNet("");
    
    ACTIVATION_FUNC act_func=f_LReLU;
    OPTIMIZATION_METHOD opt=Vanilla;

    network->add_layer({5},standard,opt,act_func);
    network->add_layer({10,10},standard,opt,act_func);
    network->add_layer({10,10},standard,opt,act_func);
    network->add_layer({5},standard,opt,act_func);

    network->set_training_mode(true);
    network->set_learning_rate(0.01);
    network->set_learning_rate_decay(0.0001);
    network->set_learning_momentum(0.9);
    network->set_loss_dependent_attention(true);

    // test iterations
    vector<double> input_vector(5); 
    vector<double> output_vector(5);
    for (int i=0;i<=10000; i++){
        // chose random numbers
        for (int r=0;r<5;r++){
            input_vector[r]=((double)rand())/RAND_MAX;
        }
        network->set_inputs(input_vector);
        network->forward_pass();
        output_vector = network->get_outputs_1d();
        network->autoencode();
        network->backpropagate();
        if (i%1000==0){
            cout << "iteration" << i << ", loss " << network->get_loss_avg() << "========================================================================================================\n";
            cout << "input_vector " << i << ": " << input_vector[0] << " " << input_vector[1] << " " <<input_vector[2]<< " " << input_vector[3]<< " " << input_vector[4] << "\n";        
            cout << "output_vector " << i << ": " << output_vector[0] << " " << output_vector[1] << " " << output_vector[2] << " " << output_vector[3]<< " " << output_vector[4] << "\n\n";
        }
    }
    delete network;
    return 0;
} 