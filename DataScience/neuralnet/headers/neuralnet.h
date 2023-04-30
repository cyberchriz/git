// author: 'cyberchriz' (Christian Suer)
// license: 'github.com/cyberchriz/git/LICENCE.md'

// objective of this file: flexible implementation of modular neural networks
// with a variety of different layer types, making use of matrix operations

#pragma once
#include "../../datastructures/headers/datastructures.h"
#include "layer.h"

// forward declarations
struct Layer;
struct AddLayer;
enum LossFunction;

class NeuralNet{
    public:
        // public methods
        void fit(const Vector<Array<double>>& features, const Vector<Array<double>>& labels, const int batch_size, const int epochs); // for batch training
        void fit(const Array<double>& features, const Array<double>& labels); // for online training
        void predict(const Array<double>& features); // predict output from new feature input
        void save(); // save model to file
        void load(); // load model from file
        void summary(); // prints a summary of the model architecture        
        AddLayer add_layer;
        // constructor(s)
        NeuralNet();
        // destructor
        ~NeuralNet();
        // public member objects
        Vector<Layer> layer;
        int layers;
        LossFunction loss_function;
        int backprop_iterations=0;
        int batch_counter=0;
        int loss_counter=0;
    private:
        // private methods
        void backpropagate();
        void calculate_loss();
};

// including the source file: required because this is a template class
#include "../sources/neuralnet.cpp"