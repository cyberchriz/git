// author: 'cyberchriz' (Christian Suer)
// license: 'github.com/cyberchriz/git/LICENCE.md'

// objective of this file: flexible implementation of modular neural networks
// with a variety of different layer types, making use of matrix operations

#pragma once
#include "../../datastructures/headers/datastructures.h"
#include "../../distributions/headers/distributions.h"
#include "../../../utilities/headers/log.h"
#include <vector>
#include <initializer_list>

// forward declarations
struct AddLayer;
struct Layer;
class NeuralNet;

enum LossFunction{
    MSE,                // Mean Squared Error
    MAE,                // Mean Absolute Error
    MAPE,               // Mean Absolute Percentage Error
    MSLE,               // Mean Squared Logarithmic Error
    CosProx,            // Cosine Proximity
    CatCrossEntr,       // Categorical Crossentropy
    SparceCatCrossEntr, // Sparse Categorical Crossentropy
    BinCrossEntr,       // Binary Crossentropy
    KLD,                // Kullback-Leibler Divergence
    Poisson,            // Poisson
    Hinge,              // Hinge
    SquaredHinge,       // Squared Hinge
    LogCosh,            // LogCosh
    LOSS_FUNCTION_COUNT
};

enum LayerType{
    max_pooling_layer,
    avg_pooling_layer,
    input_layer,
    output_layer,
    lstm_layer,
    recurrent_layer,
    dense_layer,
    convolutional_layer,
    GRU_layer,
    dropout_layer,
    ReLU_layer,
    lReLU_layer,
    ELU_layer,
    sigmoid_layer,
    tanh_layer,
    flatten_layer,
    LAYER_TYPE_COUNT // used to verify valid enum argument          
};

struct Layer{
    public:
        // public member objects
        LayerType type;
        int neurons;
        int dimensions;
        std::initializer_list<int> shape;
        Array<double> h; // internal states
        Array<double> label; // labels for output layer
        Array<double> gradient;
        Array<double> loss;
        Array<double> loss_sum;
        Array<Vector<double>> h_t; // for each neuron in an array of neurons: vector of timesteps holding the hidden states for LSTMs
        Array<Vector<double>> c_t; // for each neuron in an array of neurons: vector of timesteps holding the cell states for LSTMs
        Array<Array<double>> U_f; // stores the weights for h(t-1) to calculate the forget gate: f(t) = σ(W_f * x(t) + U_f * h(t-1) + b_f)
        Array<Array<double>> U_i; // stores the weights for h(t-1) to calculate the input gate: i(t) = σ(W_i * x(t) + U_i * h(t-1) + b_i)
        Array<Array<double>> U_o; // stores the weights for h(t-1) to calculate the outout gate: o(t) = σ(W_o * x(t) + U_o * h(t-1) + b_o)
        Array<Array<double>> U_c; // stores the weights for h(t-1) to calculate the candidate gate: c(t) = tanh(W_c * x(t) + U_c * h(t-1) + b_c)
        Array<Array<double>> W_f; // forget gate weights for x(t)
        Array<Array<double>> W_i; // input gate weights for x(t)
        Array<Array<double>> W_o; // output gate weights for x(t)
        Array<Array<double>> W_c; // candidate gate weights for x(t)
        Array<double> b_f; // forget gate bias weights;
        Array<double> b_i; // input gate bias weights;
        Array<double> b_o; // output gate bias weights;
        Array<double> b_c; // candidate gate bias weights;
        // weights for dense connections
        Array<Array<double *>> W_out; // pointers to W_i weights of next layer
        double dropout_ratio=0;
        Vector<Array<double>> kernel; // vector of kernels for CNN layers
        Vector<int> pooling_slider_shape; // vector of pointers to pooling slider shapes
        Vector<int> pooling_stride_shape; // vector of pointers to pooling stride shapes

        // public constructor
        Layer();

        // destructor
        ~Layer();
};

// struct for adding new layers to an instance of NeuralNet
struct AddLayer{
    private:

        struct ActivationLayer{
            public:
                // public methods
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
            public:
                // public methods
                void max(std::vector<int> slider_shape, std::vector<int> stride_shape);
                void avg(std::vector<int> slider_shape, std::vector<int> stride_shape);
                // constructor
                Pooling(NeuralNet* network) : network(network){};
            private:
                NeuralNet* network;
                Log logger;
        };                

    public:
        // public methods
        void input(std::initializer_list<int> shape);
        void input(const int neurons){input({neurons});}
        void output(std::initializer_list<int> shape, LossFunction loss_function=MSE);
        void output(const int neurons, LossFunction loss_function=MSE){output({neurons},loss_function);}
        void lstm(std::initializer_list<int> shape, const int timesteps);
        void lstm(const int neurons, const int timesteps=10){lstm({neurons},timesteps);}
        void lstm(const int timesteps=10);
        void recurrent(std::initializer_list<int> shape);
        void recurrent(const int neurons){recurrent({neurons});}
        void recurrent();
        void dense(std::initializer_list<int> shape);
        void dense(const int neurons){dense({neurons});}
        void dense();
        void convolutional();
        void GRU(std::initializer_list<int> shape);
        void GRU(const int neurons){GRU({neurons});}
        void GRU();
        void dropout(const double ratio=0.2);
        void flatten();
        Pooling pool;
        ActivationLayer activation;

        // constructor
        AddLayer(NeuralNet* network) :
            // member initialization list
            network(network),
            activation(network),
            pool(network) {
            // constructor definition
            logger.enable_to_console(true);
            logger.enable_to_file(false);
            logger.set_level(LOG_LEVEL_DEBUG);
        };

        // destructor
        ~AddLayer(){};
    protected:
    private:
        // private methods
        static void init(NeuralNet* network, LayerType type, std::initializer_list<int> shape);
        void make_dense_connections();
        std::initializer_list<int> vector_to_initlist(const std::vector<int>& vec);
        // private member objects
        NeuralNet* network;
        Log logger;
};

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
        NeuralNet():
            // member initialization list
            add_layer(this),
            layers(0) {
            // constructor definition
            logger.enable_to_console(true);
            logger.enable_to_file(false);
            logger.set_level(LOG_LEVEL_DEBUG);
        }
        // destructor
        ~NeuralNet();
        // public member objects
        std::vector<Layer> layer;
        int layers;
        double loss_avg;
        LossFunction loss_function;
        int backprop_iterations=0;
        int batch_counter=0;
        int loss_counter=0;
    private:
        // private methods
        void backpropagate();
        void calculate_loss();
        Log logger;
};