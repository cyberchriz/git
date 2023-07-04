#include "../headers/neuralnet.h"

int main(){
    NeuralNet model;
    model.addlayer_input({5,5});
    model.addlayer_dense({5,5});
    model.addlayer_ReLU();
    model.addlayer_convolutional(1,true);
    model.addlayer_output(5,MSE);
    model.log_summary();
}