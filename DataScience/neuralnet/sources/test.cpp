#include "../headers/neuralnet.h"

int main(){
    NeuralNet model;
    model.addlayer_input({3,3});
    model.addlayer_dense(5);
    model.addlayer_output(5,MSE);
    model.log_summary();
}