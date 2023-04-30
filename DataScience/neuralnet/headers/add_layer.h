// author: 'cyberchriz' (Christian Suer)
// license: 'github.com/cyberchriz/git/LICENCE.md'

// objective of this file: helper struct as interface for different
// methods of adding new layers to a neural network

#pragma once
#include "neuralnet.h"
#include "../../datastructures/headers/datastructures.h"
#include "../../distributions/headers/distributions.h"

// forward declarations
class NeuralNet;


// struct for adding new layers to an instance of NeuralNet
struct AddLayer{
    private:

        struct ActivationLayer{
            public:
                void sigmoid();
                void ReLU();
                void lReLU();
                void ELU();
                void tanh();
                // constructor
                ActivationLayer(NeuralNet* network) : network(network){};
            private:
                NeuralNet* network;
        };

        struct Pooling{
            void max();
            void avg();
        };                

    public:
        // public methods
        void input(std::initializer_list<int> shape);
        void output(std::initializer_list<int> shape, LossFunction loss_function);
        void lstm(std::initializer_list<int> shape, const int timesteps);
        void lstm(const int timesteps);
        void recurrent(std::initializer_list<int> shape);
        void recurrent();
        void dense(std::initializer_list<int> shape);
        void dense();
        void convolutional();
        void GRU(std::initializer_list<int> shape);
        void GRU();
        void GRU(uint neurons);
        void dropout(const double ratio=0.2);
        void flatten();
        Pooling pool;
        ActivationLayer activation;

        // constructor
        AddLayer(NeuralNet* network) :
            network(network),
            activation(network){};
        // destructor
        ~AddLayer(){};
    protected:
    private:
        // private methods
        static void init(NeuralNet* network, LayerType type, std::initializer_list<int> shape);
        void make_dense_connections();
        // private member objects
        NeuralNet* network;
};

#include "../sources/add_layer.cpp"