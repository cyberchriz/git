#include "../headers/neuralnet.h"

enum LossFunction{
    MSE,                // Mean Squared Error
    MAE,                // Mean Absolute Error
    MAPE,               // Mean Absolute Percentage Error
    MSLE,               // Mean Squared Logarithmic Error
    CatCrossEntr,       // Categorical Crossentropy
    SparceCatCrossEntr, // Sparse Categorical Crossentropy
    BinCrossEntr,       // Binary Crossentropy
    KLD,                // Kullback-Leibler Divergence
    Poisson,            // Poisson
    Hinge,              // Hinge
    SquaredHinge,       // Squared Hinge
    LOSS_FUNCTION_COUNT
};

// public constructor
NeuralNet::NeuralNet() :
    // member initialization list
    add_layer(this),
    layers(0) {
    // empty
}

// batch training
void NeuralNet::fit(const Vector<Array<double>>& features, const Vector<Array<double>>& labels, const int batch_size, const int epochs){
    int total_samples = features.get_elements();
    for (int epoch = 0; epoch<epochs; epoch++){
        for (int sample = 0; sample < total_samples; sample++){
            while (batch_counter < batch_size){
                predict(features[sample]);
                layer[layers-1].label = labels[sample];
                calculate_loss();
            }
            backpropagate();
            // reset loss_counter and loss_sum at end of mini-batch
            loss_counter = 0;
            layer[layers-1].loss_sum.fill.zeros();
        }
    }
}

// single iteration online training
void NeuralNet::fit(const Array<double>& features, const Array<double>& labels){
    predict(features);
    layer[layers-1].label = labels;
    calculate_loss();
    backpropagate();
}

// feedforward, i.e. predict output from new feature input
void NeuralNet::predict(const Array<double>& features){
    // iterate over layers
    for (int l=1;l<layers;l++){
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
            case reshape_layer:
                // TODO
                break;
            default:
                // do nothing
                break;
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
            case reshape_layer:
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
            case reshape_layer:
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
        case MSE:
            layer[l].loss_sum += (layer[l].label - layer[l].h).pow(2);
            layer[l].gradient += (layer[l].h - layer[l].label) * 2;
            break;

        // Mean Absolute Error
        case MAE:
            layer[l].loss_sum += (layer[l].label - layer[l].h).abs();
            layer[l].gradient += (layer[l].h - layer[l].label).sign();
            break;
        
        // Mean Absolute Percentage Error
        case MAPE:
            layer[l].loss_sum += (layer[l].label - layer[l].h).Hadamard_division(layer[l].label).abs();
            layer[l].gradient += (layer[l].h - layer[l].label).sign().Hadamard_product(layer[l].label).Hadamard_division((layer[l].h - layer[l].label)).abs();
            break;

        // Mean Squared Logarithmic Error
        case MSLE:               
            layer[l].loss_sum += ((layer[l].h + 1).log() - (layer[l].label + 1).log()).pow(2);
            layer[l].gradient += (((layer[l].h + 1).log() - (layer[l].label + 1).log()).Hadamard_division(layer[l].h + 1)) * 2;
            break;

        // Categorical Crossentropy
        case CatCrossEntr:
            // because the label is one-hot encoded, only the output that is labeled as true gets the update
            int true_output = 0;
            for (int j=0;j<layer[l].neurons;j++){
                if (layer[l].h.data[j]) {true_output = j;}
            }
            layer[l].loss_sum.data[true_output] -= layer[l].label.Hadamard_product(layer[l].h.log()).sum();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h);
            break;

        // Sparse Categorical Crossentropy
        case SparceCatCrossEntr:
            layer[l].loss_sum -= layer[l].h.log();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h);
            break;

        // Binary Crossentropy
        case BinCrossEntr:
            layer[l].loss_sum -= layer[l].label.Hadamard_product(layer[l].h.log()) + ((layer[l].label*-1)+1) * ((layer[l].h*-1)+1).log();
            layer[l].gradient -= layer[l].label.Hadamard_division(layer[l].h) - ((layer[l].label*-1)+1).Hadamard_division((layer[l].h*-1)+1);
            break;

        // Kullback-Leibler Divergence
        case KLD:
            layer[l].loss_sum += layer[l].label.Hadamard_product(layer[l].label.Hadamard_division(layer[l].h).log());
            layer[l].gradient += layer[l].label.Hadamard_product(layer[l].label.log() - layer[l].h.log() + 1);
            break;

        // Poisson
        case Poisson:
            layer[l].loss_sum += layer[l].h - layer[l].label.Hadamard_product(layer[l].h.log());
            layer[l].gradient += layer[l].h - layer[l].label;
            break;

        // Hinge
        case Hinge:
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1) + 1).max(0);
            layer[l].gradient -= layer[l].label.Hadamard_product((layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).sign());
            break;

        // Squared Hinge
        case SquaredHinge:
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1 + 1).max(0)).pow(2);
            layer[l].gradient += (layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).max(0) * 2;
            break;
            
        default:
            // do nothing
    }
    layer[l].loss = layer[l].loss_sum / loss_counter;
}
