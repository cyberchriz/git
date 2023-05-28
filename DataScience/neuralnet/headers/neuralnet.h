// author: 'cyberchriz' (Christian Suer)
// license: 'github.com/cyberchriz/git/LICENCE.md'

// objective of this file: flexible implementation of sequential modular neural networks
// with a variety of different layer types, making use of matrix operations

#pragma once
#include "../../datastructures/headers/datastructures.h"
#include "../../distributions/headers/distributions.h"
#include "../../../utilities/headers/log.h"
#include "../../../utilities/headers/initlists.h"
#include <vector>
#include <initializer_list>

// forward declarations
class NeuralNet;

enum LossFunction{
    MSE,                // Mean Squared Error
    MAE,                // Mean Absolute Error
    MAPE,               // Mean Absolute Percentage Error
    MSLE,               // Mean Squared Logarithmic Error
    CosProx,            // Cosine Proximity
    CatCrossEntr,       // Categorical Crossentropy
    SparseCatCrossEntr, // Sparse Categorical Crossentropy
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

enum OPTIMIZATION_METHOD {
   VANILLA,           // Vanilla Stochastic Gradient Descent
   MOMENTUM,          // Stochastic Gradient Descent with Momentum
   NESTEROV,          // Nesterov Accelerated Gradient (NAG)
   RMSPROP,           // RMSprop
   ADADELTA,          // ADADELTA
   ADAM,              // ADAM
   ADAGRAD            // AdaGrad
};

struct Layer{
    public:
        // public member objects
        LayerType type;
        int neurons;
        int dimensions;
        int timesteps;
        bool stacked=false; // indicates whether this is a stacked layer such as CNN or stacked pooling
        Array<Array<double>> feature_stack_x;
        Array<Array<double>> feature_stack_h;
        Array<Array<double>> gradient_stack;
        Array<Array<double>> filter_stack;
        std::vector<int> filter_shape;
        std::initializer_list<int> shape;
        Array<double> h; // hidden states
        Array<double> label; // labels for output layer
        Array<double> gradient;
        Array<double> loss;
        Array<double> loss_sum;
        Array<Array<double>> gradient_t; // vector of timesteps holding the gradients for h_t
        Array<Array<double>> x_t; // vector of timesteps holding the input states for RNN, LSTM
        Array<Array<double>> h_t; // vector of timesteps holding the hidden states for LSTMs
        Array<Array<double>> c_t; // vector of timesteps holding the cell states for LSTMs
        Array<Array<double>> U;   // stores the weights for h(t-1) for RNN
        Array<Array<double>> U_f; // stores the weights for h(t-1) to calculate the forget gate: f(t) = σ(W_f * x(t) + U_f * h(t-1) + b_f)
        Array<Array<double>> U_i; // stores the weights for h(t-1) to calculate the input gate: i(t) = σ(W_i * x(t) + U_i * h(t-1) + b_i)
        Array<Array<double>> U_o; // stores the weights for h(t-1) to calculate the output gate: o(t) = σ(W_o * x(t) + U_o * h(t-1) + b_o)
        Array<Array<double>> U_c; // stores the weights for h(t-1) to calculate the candidate gate: c(t) = tanh(W_c * x(t) + U_c * h(t-1) + b_c)
        Array<Array<double>> U_z; // stores the weights for h(t-1) to calculate the update gate: z(t) = σ(W_z * x(t) + U_z + h(t-1) + b_z)
        Array<Array<double>> U_r; // stores the weights for h(t-1) to calculate the reset gate: r(t) = σ(W_r * x(t) + U_r + h(t-1) + b_r)
        Array<Array<double>> W_f; // forget gate weights for x(t)
        Array<Array<double>> W_i; // input gate weights for x(t)
        Array<Array<double>> W_o; // output gate weights for x(t)
        Array<Array<double>> W_c; // candidate gate weights for x(t)
        Array<Array<double>> W_z; // update gate weights for x(t)
        Array<Array<double>> W_r; // reset gate weights for x(t)
        Array<Array<double>> W_x; // stores the weights for dense connections (dense layers, output layers)
        Array<Array<double>> delta_W_x; // stores the weight deltas for dense connections in order to support momentum optimization
        Array<Array<double>> opt_v;
        Array<Array<double>> opt_w;
        Array<Array<double>> f_gate_t;
        Array<Array<double>> i_gate_t;
        Array<Array<double>> o_gate_t;
        Array<Array<double>> c_gate_t;
        Array<Array<double>> z_gate_t; // update gate for GRU
        Array<Array<double>> r_gate_t; // reset gate for GRU
        Array<double> b; // stores the bias matrix for the given layer
        Array<double> b_f; // forget gate bias weights;
        Array<double> b_i; // input gate bias weights;
        Array<double> b_o; // output gate bias weights;
        Array<double> b_c; // candidate gate bias weights;
        Array<double> b_z; // update gate bias weights;
        Array<double> b_r; // reset gate bias weights;
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
        void fit(const Array<Array<double>>& features, const Array<Array<double>>& labels, const int batch_size, const int epochs); // for batch training
        void fit(const Array<double>& features, const Array<double>& labels); // for online training
        Array<double> predict(const Array<double>& features, bool rescale=true); // predict output from new feature input
        void save(); // save model to file
        void load(); // load model from file
        void log_summary(LogLevel level=LOG_LEVEL_INFO); // logs a summary of the model architecture     
        void set_scaling_method(ScalingMethod method){scaling_method = method;}
        void set_optimizer(OPTIMIZATION_METHOD method){opt_method = method;}
        void set_lr(const double value){lr = std::fmin(1,std::fmax(0,value));}
        void set_momentum(const double value){momentum = std::fmin(1,std::fmax(0,value));}
        void set_loss_function(const LossFunction func){loss_function = func;}
        void set_gradient_clipping(const bool active=true, const double limit=0.49999){gradient_clipping=active; max_gradient = std::fmax(0,limit);}

        void addlayer_input(std::initializer_list<int> shape);
        void addlayer_input(const int neurons){addlayer_input({neurons});}
        void addlayer_output(std::initializer_list<int> shape, LossFunction loss_function=MSE);
        void addlayer_output(const int neurons, LossFunction loss_function=MSE){addlayer_output({neurons},loss_function);}
        void addlayer_sigmoid();
        void addlayer_ReLU();
        void addlayer_lReLU();
        void addlayer_ELU();
        void addlayer_tanh();         
        void addlayer_pool_max(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape);
        void addlayer_pool_avg(const std::initializer_list<int> slider_shape, const std::initializer_list<int> stride_shape);               
        void addlayer_lstm(std::initializer_list<int> shape, const int timesteps);
        void addlayer_lstm(const int neurons, const int timesteps=10){addlayer_lstm({neurons},timesteps);}
        void addlayer_lstm(const int timesteps=10){addlayer_lstm(layer[layers-1].shape,timesteps);}
        void addlayer_recurrent(std::initializer_list<int> shape, const int timesteps=10);
        void addlayer_recurrent(const int neurons, const int timesteps=10){addlayer_recurrent({neurons}, timesteps);}
        void addlayer_recurrent(const int timesteps=10){addlayer_recurrent(layer[layers-1].shape,timesteps);}
        void addlayer_dense(std::initializer_list<int> shape);
        void addlayer_dense(const int neurons){addlayer_dense({neurons});}
        void addlayer_dense(){addlayer_dense(layer[layers-1].shape);}
        void addlayer_convolutional(const int filter_radius=1, bool padding=false);
        void addlayer_GRU(std::initializer_list<int> shape, const int timesteps=10);
        void addlayer_GRU(const int neurons, const int timesteps=10){addlayer_GRU({neurons}, timesteps);}
        void addlayer_GRU(const int timesteps=10){addlayer_GRU(layer[layers-1].shape, timesteps);}
        void addlayer_dropout(const double ratio=0.2);
        void addlayer_flatten();

        // constructor(s)
        NeuralNet(){
            Log::enable_to_console(true);
            Log::enable_to_file(false);
            Log::set_level(LogLevel::LOG_LEVEL_DEBUG);            
        }
        // destructor
        ~NeuralNet(){};
    private:
        // private methods
        void backpropagate();
        void calculate_loss();
        void layer_init(LayerType type, std::initializer_list<int> shape);
        void layer_make_dense_connections();
        std::vector<int> initlist_to_vector(const std::initializer_list<int>& list);
        std::initializer_list<int> vector_to_initlist(const std::vector<int>& vec);
        void addlayer_pool(LayerType type, std::initializer_list<int> slider_shape, std::initializer_list<int> stride_shape);         
            
        // private members
        std::vector<Layer> layer;
        int layers=0;
        double loss_avg=0;
        LossFunction loss_function=LossFunction::MSE;
        int backprop_iterations=0;
        int forward_iterations=0;
        int batch_counter=0;
        int loss_counter=0;
        int feature_maps = 10;        
        Array<double> features_mean;
        Array<double> features_stddev;
        Array<double> features_min;
        Array<double> features_max;
        Array<double> labels_mean;
        Array<double> labels_stddev;
        Array<double> labels_min;
        Array<double> labels_max;        
        ScalingMethod scaling_method = standardized;
        double lr = 0.001;
        double momentum = 0.9;
        OPTIMIZATION_METHOD opt_method = VANILLA;
        bool gradient_clipping = false;
        double max_gradient;
};