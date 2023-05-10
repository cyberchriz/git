// author: 'cyberchriz' (Christian Suer)
// license: 'github.com/cyberchriz/git/LICENCE.md'

// objective of this file: flexible implementation of modular neural networks
// with a variety of different layer types, making use of matrix operations

#pragma once
#include "../../datastructures/headers/datastructures.h"
#include "../../distributions/headers/distributions.h"
#include "../../../utilities/headers/log.h"
#include "../../../utilities/headers/initlists.h"
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

enum ScalingMethod{
    standardized,
    min_max_normalized,
    mean_normalized,
    none
};

struct Layer{
    public:
        // public member objects
        LayerType type;
        int neurons;
        int dimensions;
        int timesteps;
        bool stacked=false; // indicates whether this is a stacked layer such as CNN or stacked pooling
        int filter_radius; // radius for CNN filters (=kernels)
        Vector<Array<double>> feature_stack;
        Vector<Array<double>> filter_stack;
        std::vector<int> filter_shape;
        std::initializer_list<int> shape;
        Array<double> h; // hidden states
        Array<double> label; // labels for output layer
        Array<double> gradient;
        Array<double> loss;
        Array<double> loss_sum;
        Array<double> filter; // holds a stack of filters for CNNs
        Vector<Array<double>> x_t; // vector of timesteps holding the input states for RNN, LSTM
        Vector<Array<double>> h_t; // vector of timesteps holding the hidden states for LSTMs
        Vector<Array<double>> c_t; // vector of timesteps holding the cell states for LSTMs
        Array<Array<double>> U_f; // stores the weights for h(t-1) to calculate the forget gate: f(t) = σ(W_f * x(t) + U_f * h(t-1) + b_f)
        Array<Array<double>> U_i; // stores the weights for h(t-1) to calculate the input gate: i(t) = σ(W_i * x(t) + U_i * h(t-1) + b_i)
        Array<Array<double>> U_o; // stores the weights for h(t-1) to calculate the outout gate: o(t) = σ(W_o * x(t) + U_o * h(t-1) + b_o)
        Array<Array<double>> U_c; // stores the weights for h(t-1) to calculate the candidate gate: c(t) = tanh(W_c * x(t) + U_c * h(t-1) + b_c)
        Array<Array<double>> W_f; // forget gate weights for x(t)
        Array<Array<double>> W_i; // input gate weights for x(t)
        Array<Array<double>> W_o; // output gate weights for x(t)
        Array<Array<double>> W_c; // candidate gate weights for x(t)
        Array<Array<double>> W_x; // stores the weights for dense connections (dense layers, output layers)
        Array<double> f_gate;
        Array<double> i_gate;
        Array<double> o_gate;
        Array<double> c_gate;
        Array<double> W_h; // stores the weights for h(t-1) in recurrent neural networks
        Array<double> b; // stores the bias matrix for the given layer
        Array<double> b_f; // forget gate bias weights;
        Array<double> b_i; // input gate bias weights;
        Array<double> b_o; // output gate bias weights;
        Array<double> b_c; // candidate gate bias weights;
        // weights for dense connections
        Array<Array<double *>> W_out; // pointers to W_i weights of next layer
        double dropout_ratio=0;
        std::initializer_list<int> pooling_slider_shape; // vector of pointers to pooling slider shapes
        std::initializer_list<int> pooling_stride_shape; // vector of pointers to pooling stride shapes

        // public constructor
        Layer();

        // destructor
        ~Layer();
};

class NeuralNet{
    public:
        // public methods
        void fit(const Vector<Array<double>>& features, const Vector<Array<double>>& labels, const int batch_size, const int epochs); // for batch training
        void fit(const Array<double>& features, const Array<double>& labels); // for online training
        Array<double> predict(const Array<double>& features, bool rescale=true); // predict output from new feature input
        void save(); // save model to file
        void load(); // load model from file
        void summary(); // prints a summary of the model architecture     
        void set_scaling_method(ScalingMethod method){scaling_method = method;}  
        AddLayer add_layer;
        // constructor(s)
        NeuralNet():
            // member initialization list
            add_layer(this),
            layers(0) {
            // constructor definition
            logger = Log();
            logger.enable_to_console(true);
            logger.enable_to_file(false);
            logger.set_level(LogLevel::LOG_LEVEL_DEBUG);
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
        int feature_maps = 10;
    private:
        // private methods
        void backpropagate();
        void calculate_loss();
        // private members
        Log logger;
        Array<double> features_mean;
        Array<double> features_stddev;
        Array<double> features_min;
        Array<double> features_max;
        Array<double> labels_mean;
        Array<double> labels_stddev;
        Array<double> labels_min;
        Array<double> labels_max;        
        ScalingMethod scaling_method = standardized;
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
                void max(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape);
                void avg(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape);
                // constructor
                Pooling(NeuralNet* network) : network(network){};
            private:
                NeuralNet* network;
        };                

    public:
        // public methods
        void input(std::initializer_list<int> shape);
        void input(const int neurons){input({neurons});}
        void output(std::initializer_list<int> shape, LossFunction loss_function=MSE);
        void output(const int neurons, LossFunction loss_function=MSE){output({neurons},loss_function);}
        void lstm(std::initializer_list<int> shape, const int timesteps);
        void lstm(const int neurons, const int timesteps=10){lstm({neurons},timesteps);}
        void lstm(const int timesteps=10){lstm(network->layer[network->layers-1].shape,timesteps);}
        void recurrent(std::initializer_list<int> shape, const int timesteps=10);
        void recurrent(const int neurons, const int timesteps=10){recurrent({neurons}, timesteps);}
        void recurrent(const int timesteps=10){recurrent(network->layer[network->layers-1].shape,timesteps);}
        void dense(std::initializer_list<int> shape);
        void dense(const int neurons){dense({neurons});}
        void dense(){dense(network->layer[network->layers-1].shape);}
        void convolutional(const int filter_radius=1, bool padding=false);
        void GRU(std::initializer_list<int> shape);
        void GRU(const int neurons){GRU({neurons});}
        void GRU(){GRU(network->layer[network->layers-1].shape);}
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
            logger = Log();
            logger.enable_to_console(true);
            logger.enable_to_file(false);
            logger.set_level(LogLevel::LOG_LEVEL_DEBUG);
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