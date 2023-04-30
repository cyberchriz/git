#include "../headers/add_layer.h"

// creates a new input layer or adds a new parallel input shape
// to a preexisting input layer
void AddLayer::input(std::initializer_list<int> shape){
    init(network, input_layer, shape);  
}

// creates a new output layer or adds a new parallel output shape
// to a preexisting output layer
void AddLayer::output(std::initializer_list<int> shape, LossFunction loss_function){
    init(network, output_layer, shape);
    int l = network->layers-1;
    network->layer[l].label = Array<double>(shape);
    network->layer[l].loss = Array<double>(shape);
    network->layer[l].loss_sum = Array<double>(shape);
    network->loss_function = loss_function;
    make_dense_connections();
}

// creates an LSTM layer of the specified shape
void AddLayer::lstm(std::initializer_list<int> shape, const int timesteps){
    init(network, lstm_layer, shape);
    int l = network->layers-1;
    // create vectors of timesteps for the cell state and hidden state
    network->layer[l].c_t = Vector<Array<double>>(timesteps);
    network->layer[l].h_t = Vector<Array<double>>(timesteps);
    for (int t=0;t<timesteps;t++){
        network->layer[l].c_t[t] = Array<double>(shape);
        network->layer[l].h_t[t] = Array<double>(shape);
    }
    make_dense_connections();
    // TODO
}

// creates an LSTM layer of the same shape as the preceding layer
void AddLayer::lstm(const int timesteps){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    lstm(shape,timesteps);
}

// creates a recurrent layer of the specified shape
void AddLayer::recurrent(std::initializer_list<int> shape){
    init(network, recurrent_layer, shape);
    int l = network->layers-1;
    // created and initialize recurrent weights
    network->layer[l].recurrent_weight = Array<double>(shape);
    network->layer[l].recurrent_weight.fill.He_ReLU(network->layer[l-1].neurons);
    make_dense_connections();  
    // TODO  
}

// creates a recurrent layer of the same shape as the preceding layer
void AddLayer::recurrent(){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    recurrent(shape);
}

// creates a fully connected layer
void AddLayer::dense(std::initializer_list<int> shape){
    init(network, dense_layer, shape);
    make_dense_connections();
}

// creates a fully connected layer of the same shape as the preceding layer
void AddLayer::dense(std::initializer_list<int> shape){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    dense(shape);
}

// creates a convolutional layer
void AddLayer::convolutional(){
    
}

// creates a GRU layer
void AddLayer::GRU(std::initializer_list<int> shape){
    init(network, GRU_layer, shape);
    make_dense_connections();    
    // TODO
}

// creates a GRU layer of the same shape as the preceding layer
void AddLayer::GRU(){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    GRU(shape);    
}

// creates a dropout layer
void AddLayer::dropout(const double ratio){
    int l = network->layers - 1;
    init(network, dropout_layer, network->layer[l-1].shape);
    network->layer[l].dropout_ratio = ratio;
   
}

void AddLayer::ActivationLayer::sigmoid(){
    int l = network->layers - 1;
    init(network, sigmoid_layer, network->layer[l-1].shape);
}

void AddLayer::ActivationLayer::ReLU(){
    int l = network->layers - 1;
    init(network, ReLU_layer, network->layer[l-1].shape);;   
}

void AddLayer::ActivationLayer::lReLU(){
    int l = network->layers - 1;
    init(network, lReLU_layer, network->layer[l-1].shape);  
}

void AddLayer::ActivationLayer::ELU(){
    int l = network->layers - 1;
    init(network, ELU_layer, network->layer[l-1].shape);   
}

void AddLayer::ActivationLayer::tanh(){
    int l = network->layers - 1;
    init(network, tanh_layer, network->layer[l-1].shape);   
}

void AddLayer::flatten(){
    std::initializer_list shape = {network->layer[network->layers-1].neurons};
    init(network, flatten_layer, shape);
}

// helper method to make dense connections from
// preceding layer (l-1) to current layer (l)
void AddLayer::make_dense_connections(){
    int l = network->layers-1;
    // create incoming weights
    network->layer[l].weight_in = Array<Array<double>>(network->layer[l].shape);
    int neurons_i = network->layer[l-1].neurons;
    int neurons_j = network->layer[l].neurons;
    for (int j=0;j<neurons_j;j++){
        network->layer[l].weight_in.data[j] = Array<double>(network->layer[l-1].shape);
        network->layer[l].weight_in.data[j].fill.He_ReLU(neurons_i);
    }
    // attach outgoing weights of preceding layer
    network->layer[l-1].weight_out = Array<Array<double*>>(network->layer[l-1].shape);
    for (int i=0;i<neurons_i;i++){
        network->layer[l-1].weight_out.data[i] = Array<double*>(network->layer[l].shape);
        for (int j=0;j<neurons_j;j++){
            network->layer[l-1].weight_out.data[i].data[j] = &(network->layer[l].weight_in.data[j].data[i]);
        }
    }
    // initialize bias weight
    network->layer[l-1].bias_weight = Random<double>::uniform();
}

// basic layer setup
void AddLayer::init(NeuralNet* network, LayerType type, std::initializer_list<int> shape){
    // check valid layer type
    if (network->layers==0 && type != input_layer){
        throw std::invalid_argument("the first layer always has to be of type 'input_layer'");
    }
    if (network->layers>0 && type == input_layer){
        throw std::invalid_argument("input layer already exists");
    }    
    if (network->layer[network->layers-1].type == output_layer){
        throw std::invalid_argument("an output layer has already been defined; can't add any new layers on top");
    }
    if (static_cast<int>(type) < 0 || static_cast<int>(type) >= static_cast<int>(LayerType::LAYER_TYPE_COUNT)){
        throw std::invalid_argument("layer type enum value must be a cardinal that's less than LAYER_TYPE_COUNT");
    }
    // setup layer parameters
    network->layers++;
    int l = network->layers - 1;
    network->layer[l].type = type;
    network->layer[l].shape = shape;
    network->layer[l].h = Array<double>(shape);
    network->layer[l].gradient = Array<double>(shape);
    network->layer[l].gradient.fill.zeros();
    network->layer[l].dimensions = shape.size();
    network->layer[l].neurons = network->layer[l].h.get_elements();    
}