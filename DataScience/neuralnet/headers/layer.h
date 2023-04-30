#pragma once
#include "../../datastructures/headers/datastructures.h"

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
    reshape_layer,
    LAYER_TYPE_COUNT // used to verify valid enum argument          
};

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
        Vector<Array<double>> h_t; // vector of timesteps holding the hidden states for LSTMs
        Vector<Array<double>> c_t; // vector of timesteps holding the cell states for LSTMs
        Array<Array<double>> weight_in;
        Array<Array<double*>> weight_out;
        Array<double> recurrent_weight;
        double dropout_ratio=0;
        double bias_weight; // one bias weight per parallel array
        Vector<Array<double>> kernel; // vector of kernels for CNN layers
        Array<int> pooling_slider_shape; // vector of pointers to pooling slider shapes
        Array<int> pooling_stride_shape; // vector of pointers to pooling stride shapes

        // public constructor
        Layer();

        // destructor
        ~Layer();
};