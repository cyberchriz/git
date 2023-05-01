#include "../headers/neuralnet.h"

int main(){
    NeuralNet model;
    model.add_layer.input({3,3});
    model.add_layer.dense(5);
    model.add_layer.output(5,MSE);
}