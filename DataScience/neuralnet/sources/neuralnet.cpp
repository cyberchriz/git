#include "../headers/neuralnet.h"

// batch training
void NeuralNet::fit(const Vector<Array<double>>& features, const Vector<Array<double>>& labels, const int batch_size, const int epochs){
    int total_samples = features.get_elements();
    for (int epoch = 0; epoch<epochs; epoch++){
        int batch = 0;
        for (int sample = 0; sample < total_samples; sample++){
            if (batch_counter < batch_size){
                predict(features[sample]);
                layer[layers-1].label = labels[sample];
                calculate_loss();
                logger.log(LOG_LEVEL_DEBUG, "epoch ", epoch, ", batch ", batch, ", sample ", batch_counter, "/", batch_size, ", average loss across all outputs ", loss_avg);
            }
            else {
                logger.log(LOG_LEVEL_INFO, "epoch ", epoch, ", batch ", batch, "FINISHED, average loss (across all outputs) = ", loss_avg, "starting backprop (iteration ", backprop_iterations, ")");
                backpropagate();
                // reset loss_counter and loss_sum at end of mini-batch
                loss_counter = 0;
                layer[layers-1].loss_sum.fill.zeros();
                batch++;
            }
        }
    }
}

// single iteration online training
void NeuralNet::fit(const Array<double>& features, const Array<double>& labels){
    predict(features);
    layer[layers-1].label = labels;
    calculate_loss();
    logger.log(LOG_LEVEL_DEBUG, "average loss (across all output neurons): ", loss_avg, "; starting backprop (iteration ", backprop_iterations, ")");
    backpropagate();
}

// feedforward, i.e. predict output from new feature input
void NeuralNet::predict(const Array<double>& features){
    // iterate over layers
    for (int l=1;l<layers;l++){
        switch (layer[l].type){

            case max_pooling_layer: {
                // create a sliding box for pooling
                Array<double> slider = Array<double>(layer[l].pooling_slider_shape);
                // iterate over target
                for (int j=0;j<layer[l].h.get_elements();j++){
                    // get associated index
                    std::vector<int> index_j = layer[l].h.get_index(j);
                    // get corresponding source index
                    Vector<int> index_i;
                    for (int d=0;d<layer[l].dimensions;d++){
                        index_i.push_back(index_j[d]*layer[l].pooling_stride_shape[d]);
                    }
                    // iterate over elements of the slider
                    Vector<int> index_slider;
                    for (int n=0;n<slider.get_elements();n++){
                        // get multidimensional index of the slider element
                        index_slider = Vector<int>::asVector(slider.get_index(n));
                        // assing slider value from the element with the index of the sum of index_i+index_slider
                        slider.set(n, layer[l-1].h.get((index_i+index_slider).flatten()));
                    }
                    layer[l].h.set(j,slider.max()); 
                }
            } break;

            case avg_pooling_layer: {
                // create a sliding box for pooling
                Array<double> slider = Array<double>(layer[l].pooling_slider_shape);
                // iterate over target
                for (int j=0;j<layer[l].h.get_elements();j++){
                    // get associated index
                    std::vector<int> index_j = layer[l].h.get_index(j);
                    // get corresponding source index
                    Vector<int> index_i;
                    for (int d=0;d<layer[l].dimensions;d++){
                        index_i.push_back(index_j[d]*layer[l].pooling_stride_shape[d]);
                    }
                    // iterate over elements of the slider
                    Vector<int> index_slider;
                    for (int n=0;n<slider.get_elements();n++){
                        // get multidimensional index of the slider element
                        index_slider = Vector<int>::asVector(slider.get_index(n));
                        // assing slider value from the element with the index of the sum of index_i+index_slider
                        slider.set(n, layer[l-1].h.get((index_i+index_slider).flatten()));
                    }
                    layer[l].h.set(j,slider.mean()); 
                }                
            } break;

            case output_layer: {
                // iterate over elements
                for (int j=0;j<layer[l].h.get_elements();j++){
                    // assign dotproduct of weight matrix * inputs
                    std::vector<int> index = layer[l].h.get_index(j);
                    layer[l].h.set(j,layer[l-1].h.dotproduct(layer[l].W_i.get(index)));
                }
            } break;

            case lstm_layer: {
                // TODO
                } break;

            case recurrent_layer: {
                // TODO
            } break;
                
            case dense_layer: {
                // iterate over elements
                for (int j=0;j<layer[l].h.get_elements();j++){
                    // assign dotproduct of weight matrix * inputs
                    std::vector<int> index = layer[l].h.get_index(j);
                    layer[l].h.set(j,layer[l-1].h.dotproduct(layer[l].W_i.get(index)));
                }
            } break;
                
            case convolutional_layer: {
                // TODO
            } break;
                
            case GRU_layer: {
                // TODO
            } break;
                
            case dropout_layer: {
                layer[l].h = layer[l-1].h;
                layer[l].h.fill.dropout(layer[l].dropout_ratio);
            } break;
                
            case ReLU_layer: {
                layer[l].h = layer[l-1].h.activation.function(ActFunc::ReLU);
            } break;
                
            case lReLU_layer: {
                layer[l].h = layer[l-1].h.activation.function(ActFunc::lReLU);
            } break;
                
            case ELU_layer: {
                layer[l].h = layer[l-1].h.activation.function(ActFunc::ELU);
            } break;
                
            case sigmoid_layer: {
                layer[l].h = layer[l-1].h.activation.function(ActFunc::sigmoid);
            } break;
                
            case tanh_layer: {
                layer[l].h = layer[l-1].h.activation.function(ActFunc::tanh);
            } break;
                
            case flatten_layer: {
                layer[l].h = layer[l-1].h.flatten();
            } break;
                
            default: {
                // do nothing
            } break;
        }        
    }
}

// save the model to a file
void NeuralNet::save(){

}

// load the model from a file
void NeuralNet::load(){

}

// prints a summary of the model architecture
void NeuralNet::summary(){

}   

// performs a single iteration of backpropagation
void NeuralNet::backpropagate(){
    backprop_iterations++;
    // get average gradients for this batch
    layer[layers-1].gradient/=batch_counter;
    // reset batch counter
    batch_counter = 0;
    // calculate hidden gradients
    for (int l=layers-2; l>0; l--){
        switch (layer[l].type){
            case max_pooling_layer:
                // TODO
                break;
            case avg_pooling_layer:
                // TODO
                break;
            case input_layer:
                // TODO
                break;
            case output_layer:
                // TODO
                break;
            case lstm_layer:
                // TODO
                /* STEPS:
                Iterate over the timesteps from last (current) to oldest (first); for each timestep:

                Compute the derivative of the loss function with respect to the output at the current time step:

                dL/dy(t) = the derivative of the loss function with respect to the output y(t) at time step t. This will depend on the specific loss function you are using.
                Compute the derivative of the loss function with respect to the LSTM output gate activation at the current time step:

                dL/do(t) = dL/dy(t) * tanh(c(t)) * sigmoid_prime(o(t))
                where sigmoid_prime(o(t)) is the derivative of the sigmoid activation function applied to the output gate activation o(t)
                Compute the derivative of the loss function with respect to the LSTM cell state at the current time step:

                dL/dc(t) = dL/dy(t) * o(t) * tanh_prime(c(t)) + dL/dc(t+1) * f(t+1)
                where tanh_prime(c(t)) is the derivative of the hyperbolic tangent activation function applied to the cell state c(t), and f(t+1) is the forget gate activation at the next time step
                Compute the derivative of the loss function with respect to the LSTM forget gate activation at the current time step:

                dL/df(t) = dL/dc(t) * c(t-1) * sigmoid_prime(f(t))
                Compute the derivative of the loss function with respect to the LSTM input gate activation at the current time step:

                dL/di(t) = dL/dc(t) * g(t) * sigmoid_prime(i(t))
                Compute the derivative of the loss function with respect to the LSTM candidate activation at the current time step:

                dL/dg(t) = dL/dc(t) * i(t) * tanh_prime(g(t))
                Compute the gradients of the weight matrices and bias vectors for the current time step:

                dL/dW_i = dL/di(t) * x(t)^T
                dL/dW_f = dL/df(t) * x(t)^T
                dL/dW_o = dL/do(t) * x(t)^T
                dL/dW_c = dL/dg(t) * x(t)^T
                dL/dU_i = dL/di(t) * h(t-1)^T
                dL/dU_f = dL/df(t) * h(t-1)^T
                dL/dU_o = dL/do(t) * h(t-1)^T
                dL/dU_c = dL/dg(t) * h(t-1)^T
                dL/db_i = dL/di(t)
                dL/db_f = dL/df(t)
                dL/db_o = dL/do(t)
                dL/db_c = dL/dg(t)
                Update the weights and biases using the gradients and a suitable optimization algorithm, such as stochastic gradient descent (SGD) or Adam.

                Propagate the derivative of the loss function with respect to the hidden state h(t-1) and cell state c(t-1) to the previous time step, and repeat steps 1-8 for the previous time step, until you reach the first time step in the sequence.
                */
                break;
            case recurrent_layer:
                // TODO
                break;
            case dense_layer:
                // TODO
                break;
            case convolutional_layer:
                // TODO
                break;
            case GRU_layer:
                // TODO
                break;
            case dropout_layer:
                // TODO
                break;
            case ReLU_layer:
                // TODO
                break;
            case lReLU_layer:
                // TODO
                break;
            case ELU_layer:
                // TODO
                break;
            case sigmoid_layer:
                // TODO
                break;
            case tanh_layer:
                // TODO
                break;
            case flatten_layer:
                // TODO
                break;
            default:
                // do nothing
                break;
        }   
    }
    // update weights
    for (int l=layers-2; l>0; l--){
        switch (layer[l].type){
            case max_pooling_layer:
                // TODO
                break;
            case avg_pooling_layer:
                // TODO
                break;
            case input_layer:
                // TODO
                break;
            case output_layer:
                // TODO
                break;
            case lstm_layer:
                // TODO
                break;
            case recurrent_layer:
                // TODO
                break;
            case dense_layer:
                // TODO
                break;
            case convolutional_layer:
                // TODO
                break;
            case GRU_layer:
                // TODO
                break;
            case dropout_layer:
                // TODO
                break;
            case ReLU_layer:
                // TODO
                break;
            case lReLU_layer:
                // TODO
                break;
            case ELU_layer:
                // TODO
                break;
            case sigmoid_layer:
                // TODO
                break;
            case tanh_layer:
                // TODO
                break;
            case flatten_layer:
                // TODO
                break;
            default:
                // do nothing
                break;
        }   
    }
    // reset gradients
    layer[layers-1].gradient.fill.zeros();
}

// calculates the loss according to the specified loss function;
// note: forward pass and label-assignments must be done first!
void NeuralNet::calculate_loss(){
    // get index of output layer
    int l=layers-1;
    batch_counter++;
    loss_counter++;
    switch (loss_function){
        // Mean Squared Error
        case MSE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).pow(2);
            layer[l].gradient += (layer[l].h - layer[l].label) * 2;
        } break;

        // Mean Absolute Error
        case MAE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).abs();
            layer[l].gradient += (layer[l].h - layer[l].label).sign();
        } break;

        // Mean Absolute Percentage Error
        case MAPE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).Hadamard_division(layer[l].label).abs();
            layer[l].gradient += (layer[l].h - layer[l].label).sign().Hadamard_product(layer[l].label).Hadamard_division((layer[l].h - layer[l].label)).abs();
        } break;

        // Mean Squared Logarithmic Error
        case MSLE: {              
            layer[l].loss_sum += ((layer[l].h + 1).log() - (layer[l].label + 1).log()).pow(2);
            layer[l].gradient += (((layer[l].h + 1).log() - (layer[l].label + 1).log()).Hadamard_division(layer[l].h + 1)) * 2;
        } break;

        // Categorical Crossentropy
        case CatCrossEntr: {
            // because the label is one-hot encoded, only the output that is labeled as true gets the update
            int true_output = 0;
            for (int j=0;j<layer[l].neurons;j++){
                if (layer[l].h.data[j]) {
                    true_output = j;
                }
            }
            layer[l].loss_sum.data[true_output] -= layer[l].label.Hadamard_product(layer[l].h.log()).sum();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h);
        } break;

        // Sparse Categorical Crossentropy
        case SparceCatCrossEntr: {
            layer[l].loss_sum -= layer[l].h.log();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h);
        } break;

        // Binary Crossentropy
        case BinCrossEntr: {
            layer[l].loss_sum -= layer[l].label.Hadamard_product(layer[l].h.log()) + ((layer[l].label*-1)+1) * ((layer[l].h*-1)+1).log();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h) - ((layer[l].label*-1)+1).Hadamard_division((layer[l].h*-1)+1);
        } break;

        // Kullback-Leibler Divergence
        case KLD: {
            layer[l].loss_sum += layer[l].label.Hadamard_product(layer[l].label.Hadamard_division(layer[l].h).log());
            layer[l].gradient += layer[l].label.Hadamard_product(layer[l].label.log() - layer[l].h.log() + 1);
        } break;

        // Poisson
        case Poisson: {
            layer[l].loss_sum += layer[l].h - layer[l].label.Hadamard_product(layer[l].h.log());
            layer[l].gradient += layer[l].h - layer[l].label;
        } break;

        // Hinge
        case Hinge: {
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1) + 1).max(0);
            layer[l].gradient -= layer[l].label.Hadamard_product((layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).sign());
        } break;

        // Squared Hinge
        case SquaredHinge: {
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1 + 1).max(0)).pow(2);
            layer[l].gradient += (layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).max(0) * 2;
        } break;

        default: {
            // do nothing
        } break;
    }
    layer[l].loss = layer[l].loss_sum / loss_counter;
    loss_avg = layer[l].loss.sum() / layer[l].neurons;
}

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
    network->layer[l].c_t = Array<Vector<double>>(shape);
    network->layer[l].h_t = Array<Vector<double>>(shape);
    network->layer[l].U_f = Array<Array<double>>(shape);
    network->layer[l].U_i = Array<Array<double>>(shape);
    network->layer[l].U_o = Array<Array<double>>(shape);
    network->layer[l].U_c = Array<Array<double>>(shape);
    network->layer[l].W_f = Array<Array<double>>(network->layer[l-1].h.get_shape());
    network->layer[l].W_i = Array<Array<double>>(network->layer[l-1].h.get_shape());
    network->layer[l].W_o = Array<Array<double>>(network->layer[l-1].h.get_shape());
    network->layer[l].W_c = Array<Array<double>>(network->layer[l-1].h.get_shape());     
    for (int j=0;j<network->layer[l].c_t.get_elements();j++){
        // initialize vectors of timesteps for the cell state and hidden state
        network->layer[l].c_t.data[j] = Vector<double>(timesteps); network->layer[l].c_t.fill.zeros();
        network->layer[l].h_t.data[j] = Vector<double>(timesteps); network->layer[l].h_t.fill.zeros();
        // initialize weight matrices
        network->layer[l].U_f.data[j] = Array<double>(shape); network->layer[l].U_f.fill.He_ReLU(layer[l].h.get_elements());
        network->layer[l].U_i.data[j] = Array<double>(shape); network->layer[l].U_i.fill.He_ReLU(layer[l].h.get_elements());
        network->layer[l].U_o.data[j] = Array<double>(shape); network->layer[l].U_o.fill.He_ReLU(layer[l].h.get_elements());
        network->layer[l].U_c.data[j] = Array<double>(shape); network->layer[l].U_c.fill.He_ReLU(layer[l].h.get_elements());
        network->layer[l].W_f.data[j] = Array<double>(network->layer[l-1].h.get_shape()); network->layer[l].U_f.fill.He_ReLU(layer[l-1].h.get_elements());
        network->layer[l].W_i.data[j] = Array<double>(network->layer[l-1].h.get_shape()); network->layer[l].U_i.fill.He_ReLU(layer[l-1].h.get_elements());
        network->layer[l].W_o.data[j] = Array<double>(network->layer[l-1].h.get_shape()); network->layer[l].U_o.fill.He_ReLU(layer[l-1].h.get_elements());
        network->layer[l].W_c.data[j] = Array<double>(network->layer[l-1].h.get_shape()); network->layer[l].U_c.fill.He_ReLU(layer[l-1].h.get_elements());    
    }
    // initialize biases
    network->layer[l].b_f = Array<double>(shape); network->layer[l].b_f.fill.He_ReLU(layer[l].h.get_elements());
    network->layer[l].b_i = Array<double>(shape); network->layer[l].b_f.fill.He_ReLU(layer[l].h.get_elements());
    network->layer[l].b_o = Array<double>(shape); network->layer[l].b_f.fill.He_ReLU(layer[l].h.get_elements());
    network->layer[l].b_c = Array<double>(shape); network->layer[l].b_f.fill.He_ReLU(layer[l].h.get_elements());   
    make_dense_connections();
    // TODO
}

// creates an LSTM layer of the same shape as the preceding layer
void AddLayer::lstm(const int timesteps){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    lstm(shape,timesteps);
    // TODO
}

// creates a recurrent layer of the specified shape
void AddLayer::recurrent(std::initializer_list<int> shape){
    init(network, recurrent_layer, shape);
    int l = network->layers-1;
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
void AddLayer::dense(){
    std::initializer_list<int> shape = network->layer[network->layers-1].shape;
    dense(shape);
}

// creates a convolutional layer
void AddLayer::convolutional(){
    // TODO
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
    std::initializer_list<int> shape = {network->layer[network->layers-1].neurons};
    init(network, flatten_layer, shape);
}

void AddLayer::Pooling::avg(std::vector<int> slider_shape, std::vector<int> stride_shape){
    int l = network->layers - 1;
    // confirm valid slider shape
    if (slider_shape.size() != network->layer[l-1].dimensions){
        throw std::invalid_argument("slider shape for avg pooling must have same number of dimensions as the layer it is acting upon");
    }
    // confirm valid stride shape
    if (stride_shape.size() != network->layer[l-1].dimensions){
        throw std::invalid_argument("stride shape for avg pooling must have same number of dimensions as the layer it is acting upon");
    }    
    // get layer shape: temporarily store as std::vector
    std::vector<int> shape;
    auto iterator = network->layer[l-1].shape.begin();
    int d=0;
    for (;d<network->layer[l-1].dimensions;d++, iterator++){
        shape.emplace_back((*iterator - slider_shape[d])/stride_shape[d]);
    }
    // convert to std::initializer_list
    std::initializer_list<int> init_list;
    for (auto& elem : shape) {
        init_list = {std::initializer_list<int>{elem}};
    }
    init(network, avg_pooling_layer, init_list);
}

// helper method to convert a std::vector<int> to std::initializer_list<int>
std::initializer_list<int> AddLayer::vector_to_initlist(const std::vector<int>& vec) {
    std::initializer_list<int> init_list;
    for (auto& elem : vec) {
        init_list = {std::initializer_list<int>{elem}};
    }
}

void AddLayer::Pooling::max(std::vector<int> slider_shape, std::vector<int> stride_shape){

}

// helper method to make dense connections from
// preceding layer (l-1) to current layer (l)
void AddLayer::make_dense_connections(){
    int l = network->layers-1;
    // create incoming weights
    network->layer[l].W_i = Array<Array<double>>(network->layer[l].shape);
    int neurons_i = network->layer[l-1].neurons;
    int neurons_j = network->layer[l].neurons;
    for (int j=0;j<neurons_j;j++){
        network->layer[l].W_i.data[j] = Array<double>(network->layer[l-1].shape);
        network->layer[l].W_i.data[j].fill.He_ReLU(neurons_i);
    }
    // attach outgoing weights of preceding layer
    network->layer[l-1].W_out = Array<Array<int>>(network->layer[l-1].shape);
    for (int i=0;i<neurons_i;i++){
        network->layer[l-1].W_out.data[i] = Array<int>(network->layer[l].shape);
        for (int j=0;j<neurons_j;j++){
            // store references to associated weights into <double *>
            network->layer[l-1].W_out.data[i].data[j] = &network->layer[l].W_i.data[j].data[i];
            // example of accessing the 'fan out' of weight values for any given neuron 'i' by dereferencing:
            // Array<double> dereferenced = *network->layer[l-1].W_out.data[i];
        }
    }
    // initialize bias weights
    network->layer[l-1].b_i.fill.He_ReLU(neurons_i);
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
    // create new layer
    network->layer.emplace_back(Layer());
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