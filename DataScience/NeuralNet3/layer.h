#pragma once
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/input.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/output.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/neuron.h>
#include <vector>
using namespace std;

class Layer{
    private:
    protected:
    public:
        int index=0;
        int inputs=0;
        int neurons=0;
        int outputs=0;
        vector<Neuron> neuron;
        vector<Input> input;
        vector<Output> output;
        void add_inputs(int amount=1,SCALING scaling_method=normalized);
        void add_input(SCALING scaling_method=normalized){add_inputs(1,scaling_method);}
        void add_outputs(int amount=1,SCALING scaling_method=normalized);
        void add_output(SCALING scaling_method=normalized){add_outputs(1,scaling_method);}
        void add_hidden(int amount=1, ACTIVATION_FUNC act_func=f_LReLU);
        // constructor
        Layer(){}
        // destructor
        ~Layer(){}
};

// add inputs with associated neurons
void Layer::add_inputs(int amount, SCALING scaling_method){
    for (int i=0;i<amount;i++){
        input.push_back(Input(scaling_method));
        input[inputs].to_index=inputs;
        inputs++;
        if (neurons<inputs){
            neuron.push_back(Neuron());
            neurons++;
        }
    }
}

// add oututs with associated neurons
void Layer::add_outputs(int amount,SCALING scaling_method){
    for (int i=0;i<amount;i++){
        output.push_back(Output(scaling_method));
        output[outputs].from_index=outputs;
        outputs++;
        if (neurons<outputs){
            neuron.push_back(Neuron());
            neurons++;
        }
    }
}

// ad hidden neurons
void Layer::add_hidden(int amount, ACTIVATION_FUNC act_func){
    for (int i=0;i<amount;i++){
        neuron.push_back(Neuron(act_func));
        neurons++;
    }
}