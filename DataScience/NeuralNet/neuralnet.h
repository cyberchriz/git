#pragma once
#include <cmath>
#include <vector>
#include <enums.h>
#include <activation_functions.h>
#include <neuron.h>
#include <layer.h>
#include <valid_number.h>
#include <random_distributions.h>
using namespace std;

// class declaration
class NeuralNet {
    private:
        void load(string filename);
        int backprop_iterations=0;
        OPTIMIZATION_METHOD opt_method=Vanilla;
        SCALING scaling_method=none;
        ACTIVATION_FUNC activation=f_tanh;
        double learning_rate=0.05;
        double lr=0.001;
        double lr_momentum=0.9;
        double lr_decay=0.001;
        double opt_beta1=0.9;
        double opt_beta2=0.99;
        bool training_mode=true;
        bool gradient_dependent_attention=true;
        bool gradient_clipping=false;
        double gradient_clipping_threshold=1.0;
        string filename;
        vector<double> outputs1d;
        vector<vector<double>> outputs2d;
        vector<vector<vector<double>>> outputs3d;
    protected:
    public:
        vector<LayerObject> layer;
        int layers=0;
        NeuralNet(string filename="");
        ~NeuralNet();
        void add_layer(vector<int> neurons, LAYER_TYPE type=standard, OPTIMIZATION_METHOD opt_method=Vanilla, ACTIVATION_FUNC activation=f_tanh);
        void set_input(vector<int> input_index, double value); // set single input for 1-3d input layer
        void set_inputs(vector<double> input_vector); // for 1d input layer
        void set_inputs(vector<vector<double>> input_vector); // for 2d input layer
        void set_inputs(vector<vector<vector<double>>> input_vector); // for 3d input layer
        double get_output(vector<int> output_index); // for 1-3d output layer
        vector<double> get_outputs_1d(); // for 1d output layer
        vector<vector<double>> get_outputs_2d(); // for 2d output layer
        vector<vector<vector<double>>> get_outputs_3d(); // for 3d output layer
        vector<double> get_hidden_1d(int layer_index); // get 'h' values for 1d hidden layer (e.g. for autoencoder bottleneck)
        vector<vector<double>> get_hidden_2d(int layer_index); // get 'h' values for 2d hidden layer (e.g. for autoencoder bottleneck)
        vector<vector<vector<double>>> get_hidden_3d(int layer_index); // get 'h' values for 3d hidden layer (e.g. for autoencoder bottleneck)       
        void set_label(vector<int> output_index, double label_value);
        void set_labels_1d(vector<double> label_vector);
        void set_labels_2d(vector<vector<double>> label_vector);
        void set_labels_3d(vector<vector<vector<double>>> label_vector);
        void autoencode();
        void forward_pass();
        void backpropagate();
        void save(string filename="neuralnet_backup.dat");
        bool network_exists(){return layers!=0;}
        double get_loss_avg();
        void set_training_mode(bool active=true){training_mode=active;}
        void set_gradient_dependent_attention(bool active=true){gradient_dependent_attention=active;}
        void set_learning_rate(double value){learning_rate=fmin(fmax(0,value),1);}
        void set_learning_rate_decay(double value){lr_decay=fmin(fmax(0,value),1);}
        void set_learning_momentum(double value){lr_momentum=fmin(fmax(0,value),1);}
        void set_scaling_method(SCALING method=normalized){scaling_method=method;}
        double get_relative_error(){return layer[layers-1].rSSG/(layer[layers-1].rSSG_average+__DBL_EPSILON__);}
        void set_gradient_clipping(bool active,double threshold=1){gradient_clipping=active;gradient_clipping_threshold=threshold;}
        double get_avg_h();
        double get_avg_output();
        void balance_weights();
        void reset_weights();
    };


// constructor
NeuralNet::NeuralNet(string filename){
    this->filename=filename;
    // TO DO:
    // if file exists: load file
}

// destructor
NeuralNet::~NeuralNet(){
}

// add a new layer (input, hidden or output)
void NeuralNet::add_layer(vector<int> neurons, LAYER_TYPE type, OPTIMIZATION_METHOD opt_method, ACTIVATION_FUNC activation) {
    // initial layer (=input layer)
    if (layers==0){
        layer.push_back(LayerObject(neurons,{0},opt_method,type, activation));
        layers++;
        return;
    }
    else{
    // higher layers: set number of inputs_per_neuron according to preceding layer
    vector<int> input_dimensions;
    input_dimensions.push_back(layer[layers-1].neurons_x);
    if (layer[layers-1].neurons_y>0){input_dimensions.push_back(layer[layers-1].neurons_y);}
    if (layer[layers-1].neurons_z>0){input_dimensions.push_back(layer[layers-1].neurons_z);}
    // add new layer
    layer.push_back(LayerObject(neurons, input_dimensions, opt_method, type, activation));
    layers++;
    }
}

// set a single input value via index (with auto-scaling)
void NeuralNet::set_input(vector<int> input_index,double value) {
    int x,y,z;
    int dimensions=input_index.size();
    x=input_index[0];
    if (dimensions>1){y=input_index[1];}
    if (dimensions>2){z=input_index[2];}
    if (training_mode){
        if (dimensions==1){
            layer[0].neuron1d[x].input_min = fmin(value,layer[0].neuron1d[x].input_min);
            layer[0].neuron1d[x].input_max = fmax(value,layer[0].neuron1d[x].input_max);
            layer[0].neuron1d[x].input_maxabs = fmax(layer[0].neuron1d[x].input_maxabs,abs(value));
            layer[0].neuron1d[x].input_rolling_average = layer[0].neuron1d[x].input_rolling_average*0.99+value*0.01;
            layer[0].neuron1d[x].input_mdev2 = vnum(pow((value-layer[0].neuron1d[x].input_rolling_average),2));
            layer[0].neuron1d[x].input_variance = vnum(layer[0].neuron1d[x].input_variance*0.99+layer[0].neuron1d[x].input_mdev2*0.01);
            layer[0].neuron1d[x].input_stddev = vnum(sqrt(layer[0].neuron1d[x].input_variance));    
        }
        else if (dimensions==2){
            layer[0].neuron2d[x][y].input_min = fmin(value,layer[0].neuron2d[x][y].input_min);
            layer[0].neuron2d[x][y].input_max = fmax(value,layer[0].neuron2d[x][y].input_max);
            layer[0].neuron2d[x][y].input_maxabs = fmax(layer[0].neuron2d[x][y].input_maxabs,abs(value));
            layer[0].neuron2d[x][y].input_rolling_average = layer[0].neuron2d[x][y].input_rolling_average*0.99+value*0.01;
            layer[0].neuron2d[x][y].input_mdev2 = vnum(pow((value-layer[0].neuron2d[x][y].input_rolling_average),2));
            layer[0].neuron2d[x][y].input_variance = vnum(layer[0].neuron2d[x][y].input_variance*0.99+layer[0].neuron2d[x][y].input_mdev2*0.01);
            layer[0].neuron2d[x][y].input_stddev = (sqrt(layer[0].neuron2d[x][y].input_variance));
        }
        else if (dimensions==3){
            layer[0].neuron3d[x][y][z].input_min = fmin(value,layer[0].neuron3d[x][y][z].input_min);
            layer[0].neuron3d[x][y][z].input_max = fmax(value,layer[0].neuron3d[x][y][z].input_max);
            layer[0].neuron3d[x][y][z].input_maxabs = fmax(layer[0].neuron3d[x][y][z].input_maxabs,abs(value));
            layer[0].neuron3d[x][y][z].input_rolling_average = vnum(layer[0].neuron3d[x][y][z].input_rolling_average*0.99+value*0.01);
            layer[0].neuron3d[x][y][z].input_mdev2 = vnum(pow((value-layer[0].neuron3d[x][y][z].input_rolling_average),2));
            layer[0].neuron3d[x][y][z].input_variance = vnum(layer[0].neuron3d[x][y][z].input_variance*0.99+layer[0].neuron3d[x][y][z].input_mdev2*0.01);
            layer[0].neuron3d[x][y][z].input_stddev = vnum(sqrt(layer[0].neuron3d[x][y][z].input_variance));
        }
    }
    if (dimensions==1){
        switch (scaling_method){
            case none:
                layer[0].neuron1d[x].h = value;
                break;
            case maxabs:
                // -1 to 1
                layer[0].neuron1d[x].h = vnum(value / (layer[0].neuron1d[x].input_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[0].neuron1d[x].h = vnum(value - layer[0].neuron1d[x].input_min);
                layer[0].neuron1d[x].h = vnum(layer[0].neuron1d[x].h / (layer[0].neuron1d[x].input_max - layer[0].neuron1d[x].input_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[0].neuron1d[x].h = vnum(layer[0].neuron1d[x].h - layer[0].neuron1d[x].input_rolling_average); 
                layer[0].neuron1d[x].h = vnum(layer[0].neuron1d[x].h / (layer[0].neuron1d[x].input_stddev + __DBL_EPSILON__));
                break;
        }
    }
    else if (dimensions==2){
            switch (scaling_method){
                case none:
                    layer[0].neuron2d[x][y].h = value;
                    break;
                case maxabs:
                    // -1 to 1
                    layer[0].neuron2d[x][y].h = vnum(value / (layer[0].neuron2d[x][y].input_maxabs + __DBL_EPSILON__));
                    break;
                case normalized:
                    // 0 to 1
                    layer[0].neuron2d[x][y].h = vnum(value - layer[0].neuron2d[x][y].input_min);
                    layer[0].neuron2d[x][y].h = vnum(layer[0].neuron2d[x][y].h/(layer[0].neuron2d[x][y].input_max - layer[0].neuron2d[x][y].input_min + __DBL_EPSILON__));
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[0].neuron2d[x][y].h = vnum(layer[0].neuron2d[x][y].h - layer[0].neuron2d[x][y].input_rolling_average); 
                    layer[0].neuron2d[x][y].h = vnum(layer[0].neuron2d[x][y].h / (layer[0].neuron2d[x][y].input_stddev + __DBL_EPSILON__));
                    break;
            }
        }
    else if (dimensions==3){
            switch (scaling_method){
                case none:
                    layer[0].neuron3d[x][y][z].h = value;
                    break;
                case maxabs:
                    // -1 to 1
                    layer[0].neuron3d[x][y][z].h = vnum (value / (layer[0].neuron3d[x][y][z].input_maxabs + __DBL_EPSILON__));
                    break;
                case normalized:
                    // 0 to 1
                    layer[0].neuron3d[x][y][z].h = vnum(value - layer[0].neuron3d[x][y][z].input_min);
                    layer[0].neuron3d[x][y][z].h = vnum(layer[0].neuron3d[x][y][z].h / (layer[0].neuron3d[x][y][z].input_max - layer[0].neuron3d[x][y][z].input_min + __DBL_EPSILON__));
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[0].neuron3d[x][y][z].h = vnum(layer[0].neuron3d[x][y][z].h - layer[0].neuron3d[x][y][z].input_rolling_average); 
                    layer[0].neuron3d[x][y][z].h = vnum(layer[0].neuron3d[x][y][z].h / (layer[0].neuron3d[x][y][z].input_stddev + __DBL_EPSILON__));
                    break;
            }
        }        
}

// set all inputs at once via vector, for 1d input layer, with auto-scaling
void NeuralNet::set_inputs(vector<double> input_vector) {
    int size_x=input_vector.size();
    for (int x=0;x<fmin(size_x,layer[0].neurons_x);x++){
        if (training_mode){
            layer[0].neuron1d[x].input_min = fmin(input_vector[x],layer[0].neuron1d[x].input_min);
            layer[0].neuron1d[x].input_max = fmax(input_vector[x],layer[0].neuron1d[x].input_max);
            layer[0].neuron1d[x].input_maxabs = fmax(layer[0].neuron1d[x].input_maxabs,abs(input_vector[x]));
            layer[0].neuron1d[x].input_rolling_average = vnum(layer[0].neuron1d[x].input_rolling_average*0.99+input_vector[x]*0.01);
            layer[0].neuron1d[x].input_mdev2 = vnum(pow((input_vector[x]-layer[0].neuron1d[x].input_rolling_average),2));
            layer[0].neuron1d[x].input_variance = vnum(layer[0].neuron1d[x].input_variance*0.99+layer[0].neuron1d[x].input_mdev2*0.01);
            layer[0].neuron1d[x].input_stddev = vnum(sqrt(layer[0].neuron1d[x].input_variance));
        }
        switch (scaling_method){
            case none:
                layer[0].neuron1d[x].h = input_vector[x];
                break;
            case maxabs:
                // -1 to 1
                layer[0].neuron1d[x].h = vnum(input_vector[x] / (layer[0].neuron1d[x].input_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[0].neuron1d[x].h = vnum(input_vector[x] - layer[0].neuron1d[x].input_min);
                layer[0].neuron1d[x].h = vnum(layer[0].neuron1d[x].h / (layer[0].neuron1d[x].input_max - layer[0].neuron1d[x].input_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[0].neuron1d[x].h = vnum(input_vector[x] - layer[0].neuron1d[x].input_rolling_average); 
                layer[0].neuron1d[x].h = vnum(layer[0].neuron1d[x].h / (layer[0].neuron1d[x].input_stddev + __DBL_EPSILON__));
                break;
        }
    }
}

// set all inputs at once via vector, for 2d input layer, with auto-scaling
void NeuralNet::set_inputs(vector<vector<double>> input_vector){
    int size_x=input_vector.size();
    int size_y=input_vector[0].size();
    for (int x=0;x<fmin(size_x,layer[0].neurons_x);x++){
        for (int y=0;y<fmin(size_y,layer[0].neurons_y);y++){
            if (training_mode){
                layer[0].neuron2d[x][y].input_min = fmin(input_vector[x][y],layer[0].neuron2d[x][y].input_min);
                layer[0].neuron2d[x][y].input_max = fmax(input_vector[x][y],layer[0].neuron2d[x][y].input_max);
                layer[0].neuron2d[x][y].input_maxabs = fmax(layer[0].neuron2d[x][y].input_maxabs,abs(input_vector[x][y]));
                layer[0].neuron2d[x][y].input_rolling_average = vnum(layer[0].neuron2d[x][y].input_rolling_average*0.99+input_vector[x][y]*0.01);
                layer[0].neuron2d[x][y].input_mdev2 = vnum(pow((input_vector[x][y]-layer[0].neuron2d[x][y].input_rolling_average),2));
                layer[0].neuron2d[x][y].input_variance = vnum(layer[0].neuron2d[x][y].input_variance*0.99+layer[0].neuron2d[x][y].input_mdev2*0.01);
                layer[0].neuron2d[x][y].input_stddev = vnum(sqrt(layer[0].neuron2d[x][y].input_variance));
            }
            switch (scaling_method){
                case none:
                    layer[0].neuron2d[x][y].h = input_vector[x][y];
                    break;
                case maxabs:
                    // -1 to 1
                    layer[0].neuron2d[x][y].h = vnum(input_vector[x][y] / (layer[0].neuron2d[x][y].input_maxabs + __DBL_EPSILON__));
                    break;
                case normalized:
                    // 0 to 1
                    layer[0].neuron2d[x][y].h = vnum(input_vector[x][y] - layer[0].neuron2d[x][y].input_min);
                    layer[0].neuron2d[x][y].h = vnum(layer[0].neuron2d[x][y].h / (layer[0].neuron2d[x][y].input_max - layer[0].neuron2d[x][y].input_min + __DBL_EPSILON__));
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[0].neuron2d[x][y].h = vnum(input_vector[x][y] - layer[0].neuron2d[x][y].input_rolling_average); 
                    layer[0].neuron2d[x][y].h = vnum(layer[0].neuron2d[x][y].h / (layer[0].neuron2d[x][y].input_stddev + __DBL_EPSILON__));
                    break;
            }
        }
    }
}

// set all inputs at once via vector, for 3d input layer, with auto-scaling
void NeuralNet::set_inputs(vector<vector<vector<double>>> input_vector){
    int size_x=input_vector.size();
    int size_y=input_vector[0].size();
    int size_z=input_vector[0][0].size();
    for (int x=0;x<fmin(size_x,layer[0].neurons_x);x++){
        for (int y=0;y<fmin(size_y,layer[0].neurons_y);y++){
            for (int z=0;z<fmin(size_z,layer[0].neurons_z);z++){
                if (training_mode){
                    layer[0].neuron3d[x][y][z].input_min = fmin(input_vector[x][y][z],layer[0].neuron3d[x][y][z].input_min);
                    layer[0].neuron3d[x][y][z].input_max = fmax(input_vector[x][y][z],layer[0].neuron3d[x][y][z].input_max);
                    layer[0].neuron3d[x][y][z].input_maxabs = fmax(layer[0].neuron3d[x][y][z].input_maxabs,abs(input_vector[x][y][z]));
                    layer[0].neuron3d[x][y][z].input_rolling_average = vnum(layer[0].neuron3d[x][y][z].input_rolling_average*0.99+input_vector[x][y][z]*0.01);
                    layer[0].neuron3d[x][y][z].input_mdev2 = vnum(pow((input_vector[x][y][z]-layer[0].neuron3d[x][y][z].input_rolling_average),2));
                    layer[0].neuron3d[x][y][z].input_variance = vnum(layer[0].neuron3d[x][y][z].input_variance*0.99+layer[0].neuron3d[x][y][z].input_mdev2*0.01);
                    layer[0].neuron3d[x][y][z].input_stddev = vnum(sqrt(layer[0].neuron3d[x][y][z].input_variance));
                }
                switch (scaling_method){
                    case none:
                        layer[0].neuron3d[x][y][z].h = input_vector[x][y][z];
                        break;
                    case maxabs:
                        // -1 to 1
                        layer[0].neuron3d[x][y][z].h = vnum(input_vector[x][y][z] / (layer[0].neuron3d[x][y][z].input_maxabs + __DBL_EPSILON__));
                        break;
                    case normalized:
                        // 0 to 1
                        layer[0].neuron3d[x][y][z].h = vnum(input_vector[x][y][z] - layer[0].neuron3d[x][y][z].input_min);
                        layer[0].neuron3d[x][y][z].h = vnum(layer[0].neuron3d[x][y][z].h / (layer[0].neuron3d[x][y][z].input_max - layer[0].neuron3d[x][y][z].input_min + __DBL_EPSILON__));
                        break;
                    default:
                        // standardized (µ=0, sigma=1)
                        layer[0].neuron3d[x][y][z].h = vnum(input_vector[x][y][z] - layer[0].neuron3d[x][y][z].input_rolling_average); 
                        layer[0].neuron3d[x][y][z].h = vnum(layer[0].neuron3d[x][y][z].h / (layer[0].neuron3d[x][y][z].input_stddev + __DBL_EPSILON__));
                        break;
                }
            }
        }
    }
}

// return a single output value via 1-3d index
double NeuralNet::get_output(vector<int> output_index){
    int dimensions=output_index.size();
    if (dimensions!=layer[layers-1].dimensions){return NAN;}
    if (dimensions==1){
        int x=output_index[0];
        return layer[layers-1].neuron1d[x].output;
    }
    else if (dimensions==2){
        int x=output_index[0];
        int y=output_index[1];
        return layer[layers-1].neuron2d[x][y].output;
    }
    else if (dimensions==3){
        int x=output_index[0];
        int y=output_index[1];
        int z=output_index[2];
        return layer[layers-1].neuron3d[x][y][z].output;
    }
    return NAN;
}

// get output vector from 1d output layer
vector<double> NeuralNet::get_outputs_1d(){
    // initialize private outputs1d vector if this is the first call
    if (outputs1d.size()==0){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            outputs1d.push_back(0);
        }
    }
    // get outputs
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        outputs1d[x]=layer[layers-1].neuron1d[x].output;
    }
    return outputs1d;
}

// get output vector from 2d output layer
vector<vector<double>> NeuralNet::get_outputs_2d(){
    // initialize private outputs2d vector if this is the first call
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        vector<double> column;
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            column.push_back(0);
        }
        outputs2d.push_back(column);
    }
    // get outputs
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            outputs2d[x][y]=layer[layers-1].neuron2d[x][y].output;
        }
    }
    return outputs2d;
}   

// get output vector from 3d output layer
vector<vector<vector<double>>> NeuralNet::get_outputs_3d(){
    // initialize private outputs 3d vector if this is the first call
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        vector<vector<double>> column;
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            vector<double> stack;
            for (int z=0;z<layer[layers-1].neurons_z;z++){
                stack.push_back(0);
            }
            column.push_back(stack);
        }
        outputs3d.push_back(column);
    }
    // get outputs
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            for (int z=0;z<layer[layers-1].neurons_z;z++){
                outputs3d[x][y][z]=layer[layers-1].neuron3d[x][y][z].output;
            }
        }
    }
    return outputs3d;
}   

// get 'h' values for 1d hidden layer (e.g. for autoencoder bottleneck)
vector<double> NeuralNet::get_hidden_1d(int layer_index){
    vector<double> result(layer[layer_index].neurons_x);
    for (int x=0;x<layer[layer_index].neurons_x;x++){
        result[x]=layer[layer_index].neuron1d[x].h;        
    }
    return result;
}

// get 'h' values for 2d hidden layer (e.g. for autoencoder bottleneck)
vector<vector<double>> NeuralNet::get_hidden_2d(int layer_index){
    vector<vector<double>> result;
    for (int x=0;x<layer[layer_index].neurons_x;x++){
        vector<double> column;
        for (int y=0;y<layer[layer_index].neurons_y;y++){
            column.push_back(layer[layer_index].neuron2d[x][y].h);
        }
        result.push_back(column);
    }
    return result;
}

// get 'h' values for 3d hidden layer (e.g. for autoencoder bottleneck)
vector<vector<vector<double>>> NeuralNet::get_hidden_3d(int layer_index){
    vector<vector<vector<double>>> result;
    for (int x=0;x<layer[layer_index].neurons_x;x++){
        vector<vector<double>> column;
        for (int y=0;y<layer[layer_index].neurons_y;y++){
            vector<double> stack;
            for (int z=0;z<layer[layer_index].neurons_z;z++){
                stack.push_back(layer[layer_index].neuron3d[x][y][z].h);
            }
            column.push_back(stack);
        }
        result.push_back(column);
    }
    return result;
}

// set a single label via 1-3d index
void NeuralNet::set_label(vector<int> output_index, double label_value) {
    int dimensions=output_index.size();
    int x,y,z;
    x=output_index[0];
    if (dimensions>1){y=output_index[1];}
    if (dimensions>2){z=output_index[2];}
    if (training_mode){
        if (dimensions==1){
            layer[layers-1].neuron1d[x].label_min = fmin(label_value,layer[layers-1].neuron1d[x].label_min);
            layer[layers-1].neuron1d[x].label_max = fmax(label_value,layer[layers-1].neuron1d[x].label_max);
            layer[layers-1].neuron1d[x].label_maxabs = fmax(layer[layers-1].neuron1d[x].label_maxabs,abs(label_value));
            layer[layers-1].neuron1d[x].label_rolling_average = vnum(layer[layers-1].neuron1d[x].label_rolling_average*0.99+label_value*0.01);
            layer[layers-1].neuron1d[x].label_mdev2 = vnum(pow((label_value-layer[layers-1].neuron1d[x].label_rolling_average),2));
            layer[layers-1].neuron1d[x].label_variance = vnum(layer[layers-1].neuron1d[x].label_variance*0.99+layer[layers-1].neuron1d[x].label_mdev2*0.01);
            layer[layers-1].neuron1d[x].label_stddev = vnum(sqrt(layer[layers-1].neuron1d[x].label_variance));
        }

        if (dimensions==2){
            layer[layers-1].neuron2d[x][y].label_min = fmin(label_value,layer[layers-1].neuron2d[x][y].label_min);
            layer[layers-1].neuron2d[x][y].label_max = fmax(label_value,layer[layers-1].neuron2d[x][y].label_max);
            layer[layers-1].neuron2d[x][y].label_maxabs = fmax(layer[layers-1].neuron2d[x][y].label_maxabs,abs(label_value));
            layer[layers-1].neuron2d[x][y].label_rolling_average = layer[layers-1].neuron2d[x][y].label_rolling_average*0.99+label_value*0.01;
            layer[layers-1].neuron2d[x][y].label_mdev2 = vnum(pow((label_value-layer[layers-1].neuron2d[x][y].label_rolling_average),2));
            layer[layers-1].neuron2d[x][y].label_variance = vnum(layer[layers-1].neuron2d[x][y].label_variance*0.99+layer[layers-1].neuron2d[x][y].label_mdev2*0.01);
            layer[layers-1].neuron2d[x][y].label_stddev = vnum(sqrt(layer[layers-1].neuron2d[x][y].label_variance));
        }

        if (dimensions==3){
            layer[layers-1].neuron3d[x][y][z].label_min = fmin(label_value,layer[layers-1].neuron3d[x][y][z].label_min);
            layer[layers-1].neuron3d[x][y][z].label_max = fmax(label_value,layer[layers-1].neuron3d[x][y][z].label_max);
            layer[layers-1].neuron3d[x][y][z].label_maxabs = fmax(layer[layers-1].neuron3d[x][y][z].label_maxabs,abs(label_value));
            layer[layers-1].neuron3d[x][y][z].label_rolling_average = vnum(layer[layers-1].neuron3d[x][y][z].label_rolling_average*0.99+label_value*0.01);
            layer[layers-1].neuron3d[x][y][z].label_mdev2 = vnum(pow((label_value-layer[layers-1].neuron3d[x][y][z].label_rolling_average),2));
            layer[layers-1].neuron3d[x][y][z].label_variance = vnum(layer[layers-1].neuron3d[x][y][z].label_variance*0.99+layer[layers-1].neuron3d[x][y][z].label_mdev2*0.01);
            layer[layers-1].neuron3d[x][y][z].label_stddev = vnum(sqrt(layer[layers-1].neuron3d[x][y][z].label_variance));
        }                
    }
    if (dimensions==1){
        switch (scaling_method){
            case none:
                layer[layers-1].neuron1d[x].scaled_label = label_value;
                break;
            case maxabs:
                // -1 to 1
                layer[layers-1].neuron1d[x].scaled_label = vnum(label_value / (layer[layers-1].neuron1d[x].label_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[layers-1].neuron1d[x].scaled_label = vnum(label_value - layer[layers-1].neuron1d[x].label_min);
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label / (layer[layers-1].neuron1d[x].label_max - layer[layers-1].neuron1d[x].label_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label - layer[layers-1].neuron1d[x].label_rolling_average); 
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label / (layer[layers-1].neuron1d[x].label_stddev + __DBL_EPSILON__));
                break;
        }
    }
    if (dimensions==2){
        switch (scaling_method){
            case none:
                layer[layers-1].neuron2d[x][y].scaled_label = label_value;
                break;
            case maxabs:
                // -1 to 1
                layer[layers-1].neuron2d[x][y].scaled_label = vnum(label_value / (layer[layers-1].neuron2d[x][y].label_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[layers-1].neuron2d[x][y].scaled_label = vnum(label_value - layer[layers-1].neuron2d[x][y].label_min);
                layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label / (layer[layers-1].neuron2d[x][y].label_max - layer[layers-1].neuron2d[x][y].label_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label - layer[layers-1].neuron2d[x][y].label_rolling_average); 
                layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label / (layer[layers-1].neuron2d[x][y].label_stddev + __DBL_EPSILON__));
                break;
        }
    }
    if (dimensions==3){
        switch (scaling_method){
            case none:
                layer[layers-1].neuron3d[x][y][z].scaled_label = label_value;
                break;
            case maxabs:
                // -1 to 1
                layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(label_value / (layer[layers-1].neuron3d[x][y][z].label_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(label_value - layer[layers-1].neuron3d[x][y][z].label_min);
                layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label / (layer[layers-1].neuron3d[x][y][z].label_max - layer[layers-1].neuron3d[x][y][z].label_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label - layer[layers-1].neuron3d[x][y][z].label_rolling_average); 
                layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label / (layer[layers-1].neuron3d[x][y][z].label_stddev + __DBL_EPSILON__));
                break;
        }
    }        
}

// set labels for 1d outputs via vector
void NeuralNet::set_labels_1d(vector<double> label_vector) {
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        if (training_mode){
            layer[layers-1].neuron1d[x].label_min = fmin(label_vector[x],layer[layers-1].neuron1d[x].label_min);
            layer[layers-1].neuron1d[x].label_max = fmax(label_vector[x],layer[layers-1].neuron1d[x].label_max);
            layer[layers-1].neuron1d[x].label_maxabs = fmax(layer[layers-1].neuron1d[x].label_maxabs,abs(label_vector[x]));
            layer[layers-1].neuron1d[x].label_rolling_average = vnum(layer[layers-1].neuron1d[x].label_rolling_average*0.99+label_vector[x]*0.01);
            layer[layers-1].neuron1d[x].label_mdev2 = vnum(pow((label_vector[x]-layer[layers-1].neuron1d[x].label_rolling_average),2));
            layer[layers-1].neuron1d[x].label_variance = vnum(layer[layers-1].neuron1d[x].label_variance*0.99+layer[layers-1].neuron1d[x].label_mdev2*0.01);
            layer[layers-1].neuron1d[x].label_stddev = vnum(sqrt(layer[layers-1].neuron1d[x].label_variance));
        }
        switch (scaling_method){
            case none:
                layer[layers-1].neuron1d[x].scaled_label = label_vector[x];
                break;
            case maxabs:
                // -1 to 1
                layer[layers-1].neuron1d[x].scaled_label = vnum(label_vector[x] / (layer[layers-1].neuron1d[x].label_maxabs + __DBL_EPSILON__));
                break;
            case normalized:
                // 0 to 1
                layer[layers-1].neuron1d[x].scaled_label = vnum(label_vector[x] - layer[layers-1].neuron1d[x].label_min);
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label / (layer[layers-1].neuron1d[x].label_max - layer[layers-1].neuron1d[x].label_min + __DBL_EPSILON__));
                break;
            default:
                // standardized (µ=0, sigma=1)
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label - layer[layers-1].neuron1d[x].label_rolling_average); 
                layer[layers-1].neuron1d[x].scaled_label = vnum(layer[layers-1].neuron1d[x].scaled_label / (layer[layers-1].neuron1d[x].label_stddev + __DBL_EPSILON__));
                break;
        }
    }
}

// set labels for 2d outputs via vector
void NeuralNet::set_labels_2d(vector<vector<double>> label_vector){
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            if (training_mode){
                layer[layers-1].neuron2d[x][y].label_min = fmin(label_vector[x][y],layer[layers-1].neuron2d[x][y].label_min);
                layer[layers-1].neuron2d[x][y].label_max = fmax(label_vector[x][y],layer[layers-1].neuron2d[x][y].label_max);
                layer[layers-1].neuron2d[x][y].label_maxabs = fmax(layer[layers-1].neuron2d[x][y].label_maxabs,abs(label_vector[x][y]));
                layer[layers-1].neuron2d[x][y].label_rolling_average = vnum(layer[layers-1].neuron2d[x][y].label_rolling_average*0.99+label_vector[x][y]*0.01);
                layer[layers-1].neuron2d[x][y].label_mdev2 = vnum(pow((label_vector[x][y]-layer[layers-1].neuron2d[x][y].label_rolling_average),2));
                layer[layers-1].neuron2d[x][y].label_variance = vnum(layer[layers-1].neuron2d[x][y].label_variance*0.99+layer[layers-1].neuron2d[x][y].label_mdev2*0.01);
                layer[layers-1].neuron2d[x][y].label_stddev = vnum(sqrt(layer[layers-1].neuron2d[x][y].label_variance));
            }
            switch (scaling_method){
                case none:
                    layer[layers-1].neuron2d[x][y].scaled_label = label_vector[x][y];
                    break;
                case maxabs:
                    // -1 to 1
                    layer[layers-1].neuron2d[x][y].scaled_label = vnum(label_vector[x][y] / (layer[layers-1].neuron2d[x][y].label_maxabs + __DBL_EPSILON__));
                    break;
                case normalized:
                    // 0 to 1
                    layer[layers-1].neuron2d[x][y].scaled_label = vnum(label_vector[x][y] - layer[layers-1].neuron2d[x][y].label_min);
                    layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label / (layer[layers-1].neuron2d[x][y].label_max - layer[layers-1].neuron2d[x][y].label_min + __DBL_EPSILON__));
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label - layer[layers-1].neuron2d[x][y].label_rolling_average); 
                    layer[layers-1].neuron2d[x][y].scaled_label = vnum(layer[layers-1].neuron2d[x][y].scaled_label / (layer[layers-1].neuron2d[x][y].label_stddev + __DBL_EPSILON__));
                    break;
            }
        }
    }
}

// set labels for 3d outputs via vector
void NeuralNet::set_labels_3d(vector<vector<vector<double>>> label_vector){
    for (int x=0;x<layer[layers-1].neurons_x;x++){
        for (int y=0;y<layer[layers-1].neurons_y;y++){
            for (int z=0;z<layer[layers-1].neurons_z;z++){
                if (training_mode){
                    layer[layers-1].neuron3d[x][y][z].label_min = fmin(label_vector[x][y][z],layer[layers-1].neuron3d[x][y][z].label_min);
                    layer[layers-1].neuron3d[x][y][z].label_max = fmax(label_vector[x][y][z],layer[layers-1].neuron3d[x][y][z].label_max);
                    layer[layers-1].neuron3d[x][y][z].label_maxabs = fmax(layer[layers-1].neuron3d[x][y][z].label_maxabs,abs(label_vector[x][y][z]));
                    layer[layers-1].neuron3d[x][y][z].label_rolling_average = vnum(layer[layers-1].neuron3d[x][y][z].label_rolling_average*0.99+label_vector[x][y][z]*0.01);
                    layer[layers-1].neuron3d[x][y][z].label_mdev2 = vnum(pow((label_vector[x][y][z]-layer[layers-1].neuron3d[x][y][z].label_rolling_average),2));
                    layer[layers-1].neuron3d[x][y][z].label_variance = vnum(layer[layers-1].neuron3d[x][y][z].label_variance*0.99+layer[layers-1].neuron3d[x][y][z].label_mdev2*0.01);
                    layer[layers-1].neuron3d[x][y][z].label_stddev = vnum(sqrt(layer[layers-1].neuron3d[x][y][z].label_variance));
                }
                switch (scaling_method){
                    case none:
                        layer[layers-1].neuron3d[x][y][z].scaled_label = label_vector[x][y][z];
                        break;
                    case maxabs:
                        // -1 to 1
                        layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(label_vector[x][y][z] / (layer[layers-1].neuron3d[x][y][z].label_maxabs + __DBL_EPSILON__));
                        break;
                    case normalized:
                        // 0 to 1
                        layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(label_vector[x][y][z] - layer[layers-1].neuron3d[x][y][z].label_min);
                        layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label / (layer[layers-1].neuron3d[x][y][z].label_max - layer[layers-1].neuron3d[x][y][z].label_min + __DBL_EPSILON__));
                        break;
                    default:
                        // standardized (µ=0, sigma=1)
                        layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label - layer[layers-1].neuron3d[x][y][z].label_rolling_average); 
                        layer[layers-1].neuron3d[x][y][z].scaled_label = vnum(layer[layers-1].neuron3d[x][y][z].scaled_label / (layer[layers-1].neuron3d[x][y][z].label_stddev + __DBL_EPSILON__));
                        break;
                }
            }
        }
    }    
}

// autoencode (use inputs as labels)
void NeuralNet::autoencode(){
    // confirm that layer dimensions of input and output layer are identical
    if (layer[0].dimensions!=layer[layers-1].dimensions){return;}
    if (layer[0].neurons_x!=layer[layers-1].neurons_x || layer[0].neurons_y!=layer[layers-1].neurons_y || layer[0].neurons_z!=layer[layers-1].neurons_z){return;}
    // copy inputs to labels for 1d network
    if (layer[0].dimensions==1){
        for (int x=0;x<layer[0].neurons_x;x++){
            layer[layers-1].neuron1d[x].label_min = layer[0].neuron1d[x].input_min;
            layer[layers-1].neuron1d[x].label_max = layer[0].neuron1d[x].input_max;
            layer[layers-1].neuron1d[x].label_maxabs = layer[0].neuron1d[x].input_maxabs;
            layer[layers-1].neuron1d[x].label_rolling_average = layer[0].neuron1d[x].input_rolling_average;
            layer[layers-1].neuron1d[x].label_mdev2 = layer[0].neuron1d[x].input_mdev2;
            layer[layers-1].neuron1d[x].label_variance = layer[0].neuron1d[x].input_variance;
            layer[layers-1].neuron1d[x].label_stddev = layer[0].neuron1d[x].input_stddev;
            layer[layers-1].neuron1d[x].scaled_label = layer[0].neuron1d[x].h;
        }
    }
    else if (layer[0].dimensions==2){
        for (int x=0;x<layer[0].neurons_x;x++){
            for (int y=0;y<layer[0].neurons_y;y++){
                layer[layers-1].neuron2d[x][y].label_min = layer[0].neuron2d[x][y].input_min;
                layer[layers-1].neuron2d[x][y].label_max = layer[0].neuron2d[x][y].input_max;
                layer[layers-1].neuron2d[x][y].label_maxabs = layer[0].neuron2d[x][y].input_maxabs;
                layer[layers-1].neuron2d[x][y].label_rolling_average = layer[0].neuron2d[x][y].input_rolling_average;
                layer[layers-1].neuron2d[x][y].label_mdev2 = layer[0].neuron2d[x][y].input_mdev2;
                layer[layers-1].neuron2d[x][y].label_variance = layer[0].neuron2d[x][y].input_variance;
                layer[layers-1].neuron2d[x][y].label_stddev = layer[0].neuron2d[x][y].input_stddev;                
                layer[layers-1].neuron2d[x][y].scaled_label = layer[0].neuron2d[x][y].h;
            }
        }
    }
    else if (layer[0].dimensions==3){
        for (int x=0;x<layer[0].neurons_x;x++){
            for (int y=0;y<layer[0].neurons_y;y++){
                for (int z=0;z<layer[0].neurons_z;z++){
                    layer[layers-1].neuron3d[x][y][z].label_min = layer[0].neuron3d[x][y][z].input_min;
                    layer[layers-1].neuron3d[x][y][z].label_max = layer[0].neuron3d[x][y][z].input_max;
                    layer[layers-1].neuron3d[x][y][z].label_maxabs = layer[0].neuron3d[x][y][z].input_maxabs;
                    layer[layers-1].neuron3d[x][y][z].label_rolling_average = layer[0].neuron3d[x][y][z].input_rolling_average;
                    layer[layers-1].neuron3d[x][y][z].label_mdev2 = layer[0].neuron3d[x][y][z].input_mdev2;
                    layer[layers-1].neuron3d[x][y][z].label_variance = layer[0].neuron3d[x][y][z].input_variance;
                    layer[layers-1].neuron3d[x][y][z].label_stddev = layer[0].neuron3d[x][y][z].input_stddev;                       
                    layer[layers-1].neuron3d[x][y][z].scaled_label = layer[0].neuron3d[x][y][z].h;
                }
            }
        }
    }    
}
// run forward pass
void NeuralNet::forward_pass() {
    // cycle through layers
    for (int l=1;l<layers;l++){
        //  +=======================+
        //  |     FOR 1D LAYERS     |
        //  +=======================+
        if (layer[l].dimensions==1){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                // reset x
                layer[l].neuron1d[j_x].x=0;
                // get sum of weighted inputs from 1d preceding layer
                if (layer[l-1].dimensions==1){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].input_weight_1d[i_x] * layer[l-1].neuron1d[i_x].h;
                    }
                }
                // get sum of weighted inputs from 2d preceding layer
                else if (layer[l-1].dimensions==2){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] * layer[l-1].neuron2d[i_x][i_y].h;
                        }
                    }
                }            
                // get sum of weighted inputs from 3d preceding layer
                else if (layer[l-1].dimensions==3){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] * layer[l-1].neuron3d[i_x][i_y][i_z].h;
                            }
                        }
                    }
                }                    
                // add bias
                layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].bias_weight;
                // add recurrent input
                if (layer[l].type == recurrent || layer[l].type==mod_recurrent){
                    layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].m1 * layer[l].neuron1d[j_x].m1_weight;
                    layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].m2 * layer[l].neuron1d[j_x].m2_weight;
                    layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].m3 * layer[l].neuron1d[j_x].m3_weight;
                    layer[l].neuron1d[j_x].x += layer[l].neuron1d[j_x].m4 * layer[l].neuron1d[j_x].m4_weight;
                }
                // NaN Inf check
                layer[l].neuron1d[j_x].x = vnum(layer[l].neuron1d[j_x].x);
                // activate
                layer[l].neuron1d[j_x].h = vnum(activate(layer[l].neuron1d[j_x].x,layer[l].activation));
                // update recurrent inputs for next iteration
                if (layer[l].type == recurrent || layer[l].type == mod_recurrent){
                    layer[l].neuron1d[j_x].m1 = layer[l].neuron1d[j_x].h;
                    layer[l].neuron1d[j_x].m2 = vnum(layer[l].neuron1d[j_x].m2*0.9 + layer[l].neuron1d[j_x].h*0.1);
                    layer[l].neuron1d[j_x].m3 = vnum(layer[l].neuron1d[j_x].m3*0.99 + layer[l].neuron1d[j_x].h*0.01);
                    layer[l].neuron1d[j_x].m4 = vnum(layer[l].neuron1d[j_x].m4*0.999 + layer[l].neuron1d[j_x].h*0.001);
                }
                // rescale outputs
                if (l==layers-1){
                    switch (scaling_method){
                        case none:
                            layer[l].neuron1d[j_x].output = layer[l].neuron1d[j_x].h;
                            break;
                        case maxabs:
                            layer[l].neuron1d[j_x].output = vnum(layer[l].neuron1d[j_x].h * layer[l].neuron1d[j_x].label_maxabs);
                            break;
                        case normalized:
                            layer[l].neuron1d[j_x].output = vnum(layer[l].neuron1d[j_x].h * (layer[l].neuron1d[j_x].label_max - layer[l].neuron1d[j_x].label_min) + layer[l].neuron1d[j_x].label_min);
                            break;
                        default:
                            // =standardized
                            layer[l].neuron1d[j_x].output = vnum(layer[l].neuron1d[j_x].h * layer[l].neuron1d[j_x].label_stddev + layer[l].neuron1d[j_x].label_rolling_average);
                            break;
                    }
                }
            } 
        }
        //  +=======================+
        //  |     FOR 2D LAYERS     |
        //  +=======================+
        else if (layer[l].dimensions==2){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    // reset x
                    layer[l].neuron2d[j_x][j_y].x=0;
                    // get sum of weighted inputs from 1d preceding layer
                    if (layer[l-1].dimensions==1){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] * layer[l-1].neuron1d[i_x].h;
                        }
                    }
                    // get sum of weighted inputs from 2d preceding layer
                    else if (layer[l-1].dimensions==2){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] * layer[l-1].neuron2d[i_x][i_y].h;
                            }
                        }
                    }            
                    // get sum of weighted inputs from 3d preceding layer
                    else if (layer[l-1].dimensions==3){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                    layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] * layer[l-1].neuron3d[i_x][i_y][i_z].h;
                                }
                            }
                        }
                    }   
                    // add bias
                    layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].bias_weight;
                    // add recurrent input
                    if (layer[l].type == recurrent || layer[l].type==mod_recurrent){
                        layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].m1 * layer[l].neuron2d[j_x][j_y].m1_weight;
                        layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].m2 * layer[l].neuron2d[j_x][j_y].m2_weight;
                        layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].m3 * layer[l].neuron2d[j_x][j_y].m3_weight;
                        layer[l].neuron2d[j_x][j_y].x += layer[l].neuron2d[j_x][j_y].m4 * layer[l].neuron2d[j_x][j_y].m4_weight;
                    }
                    // NaN Inf check
                    layer[l].neuron2d[j_x][j_y].x = vnum(layer[l].neuron2d[j_x][j_y].x);
                    // activate
                    layer[l].neuron2d[j_x][j_y].h = vnum(activate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation));
                    // update recurrent inputs for next iteration
                    if (layer[l].type == recurrent || layer[l].type == mod_recurrent){
                        layer[l].neuron2d[j_x][j_y].m1 = layer[l].neuron2d[j_x][j_y].h;
                        layer[l].neuron2d[j_x][j_y].m2 = vnum(layer[l].neuron2d[j_x][j_y].m2*0.9 + layer[l].neuron2d[j_x][j_y].h*0.1);
                        layer[l].neuron2d[j_x][j_y].m3 = vnum(layer[l].neuron2d[j_x][j_y].m3*0.99 + layer[l].neuron2d[j_x][j_y].h*0.01);
                        layer[l].neuron2d[j_x][j_y].m4 = vnum(layer[l].neuron2d[j_x][j_y].m4*0.999 + layer[l].neuron2d[j_x][j_y].h*0.001);
                    }
                    // rescale outputs
                    if (l==layers-1){
                        switch (scaling_method){
                            case none:
                                layer[l].neuron2d[j_x][j_y].output = layer[l].neuron2d[j_x][j_y].h;
                                break;
                            case maxabs:
                                layer[l].neuron2d[j_x][j_y].output = vnum(layer[l].neuron2d[j_x][j_y].h * layer[l].neuron2d[j_x][j_y].label_maxabs);
                                break;
                            case normalized:
                                layer[l].neuron2d[j_x][j_y].output = vnum(layer[l].neuron2d[j_x][j_y].h * (layer[l].neuron2d[j_x][j_y].label_max - layer[l].neuron2d[j_x][j_y].label_min) + layer[l].neuron2d[j_x][j_y].label_min);
                                break;
                            default:
                                // =standardized
                                layer[l].neuron2d[j_x][j_y].output = vnum(layer[l].neuron2d[j_x][j_y].h * layer[l].neuron2d[j_x][j_y].label_stddev + layer[l].neuron2d[j_x][j_y].label_rolling_average);
                                break;
                        }
                    }
                }
            } 
        }
        //  +=======================+
        //  |     FOR 3D LAYERS     |
        //  +=======================+
        else if (layer[l].dimensions==3){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    for (int j_z=0;j_z<layer[l].neurons_z;j_z++){
                        // reset x
                        layer[l].neuron3d[j_x][j_y][j_z].x=0;
                        // get sum of weighted inputs from 1d preceding layer
                        if (layer[l-1].dimensions==1){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] * layer[l-1].neuron1d[i_x].h;
                            }
                        }
                        // get sum of weighted inputs from 2d preceding layer
                        else if (layer[l-1].dimensions==2){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] * layer[l-1].neuron2d[i_x][i_y].h;
                                }
                            }
                        }            
                        // get sum of weighted inputs from 3d preceding layer
                        else if (layer[l-1].dimensions==3){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                        layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] * layer[l-1].neuron3d[i_x][i_y][i_z].h;
                                    }
                                }
                            }
                        }   
                        // add bias
                        layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].bias_weight;
                        // add recurrent input
                        if (layer[l].type == recurrent || layer[l].type==mod_recurrent){
                            layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].m1 * layer[l].neuron3d[j_x][j_y][j_z].m1_weight;
                            layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].m2 * layer[l].neuron3d[j_x][j_y][j_z].m2_weight;
                            layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].m3 * layer[l].neuron3d[j_x][j_y][j_z].m3_weight;
                            layer[l].neuron3d[j_x][j_y][j_z].x += layer[l].neuron3d[j_x][j_y][j_z].m4 * layer[l].neuron3d[j_x][j_y][j_z].m4_weight;
                        }
                        // NaN Inf check
                        layer[l].neuron3d[j_x][j_y][j_z].x = vnum(layer[l].neuron3d[j_x][j_y][j_z].x);
                        // activate
                        layer[l].neuron3d[j_x][j_y][j_z].h = vnum(activate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation));
                        // update recurrent inputs for next iteration
                        if (layer[l].type == recurrent || layer[l].type == mod_recurrent){
                            layer[l].neuron3d[j_x][j_y][j_z].m1 = layer[l].neuron3d[j_x][j_y][j_z].h;
                            layer[l].neuron3d[j_x][j_y][j_z].m2 = vnum(layer[l].neuron3d[j_x][j_y][j_z].m2*0.9 + layer[l].neuron3d[j_x][j_y][j_z].h*0.1);
                            layer[l].neuron3d[j_x][j_y][j_z].m3 = vnum(layer[l].neuron3d[j_x][j_y][j_z].m3*0.99 + layer[l].neuron3d[j_x][j_y][j_z].h*0.01);
                            layer[l].neuron3d[j_x][j_y][j_z].m4 = vnum(layer[l].neuron3d[j_x][j_y][j_z].m4*0.999 + layer[l].neuron3d[j_x][j_y][j_z].h*0.001);
                        }
                        // rescale outputs
                        if (l==layers-1){
                            switch (scaling_method){
                                case none:
                                    layer[l].neuron3d[j_x][j_y][j_z].output = layer[l].neuron3d[j_x][j_y][j_z].h;
                                    break;
                                case maxabs:
                                    layer[l].neuron3d[j_x][j_y][j_z].output = vnum(layer[l].neuron3d[j_x][j_y][j_z].h * layer[l].neuron3d[j_x][j_y][j_z].label_maxabs);
                                    break;
                                case normalized:
                                    layer[l].neuron3d[j_x][j_y][j_z].output = vnum(layer[l].neuron3d[j_x][j_y][j_z].h * (layer[l].neuron3d[j_x][j_y][j_z].label_max - layer[l].neuron3d[j_x][j_y][j_z].label_min) + layer[l].neuron3d[j_x][j_y][j_z].label_min);
                                    break;
                                default:
                                    // =standardized
                                    layer[l].neuron3d[j_x][j_y][j_z].output = vnum(layer[l].neuron3d[j_x][j_y][j_z].h * layer[l].neuron3d[j_x][j_y][j_z].label_stddev + layer[l].neuron3d[j_x][j_y][j_z].label_rolling_average);
                                    break;
                            }
                        }
                    }
                }
            } 
        }        
    }
}

// run backpropagation
void NeuralNet::backpropagate() {
    backprop_iterations++;

    // cycle backwards through layers
    for (int l=layers-1;l>=1;l--){

        //  +=======================+
        //  |     FOR 1D LAYERS     |
        //  +=======================+
        if (layer[l].dimensions==1){
            if (l==layers-1){
                // step 1: output layer, global errors
                layer[l].rSSG = 0; // rSSG = root of sum of squared gradients
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    // derivative of the 0.5err^2 loss function: scaled_label-output
                    layer[l].neuron1d[j_x].gradient = vnum(layer[l].neuron1d[j_x].scaled_label - layer[l].neuron1d[j_x].h);

                    // gradient clipping
                    if (gradient_clipping){
                        layer[l].neuron1d[j_x].gradient = fmin(layer[l].neuron1d[j_x].gradient, gradient_clipping_threshold);
                        layer[l].neuron1d[j_x].gradient = fmax(layer[l].neuron1d[j_x].gradient, -gradient_clipping_threshold);
                    }

                    // 0.5err^2 loss
                    layer[l].neuron1d[j_x].loss = vnum(0.5 * layer[l].neuron1d[j_x].gradient * layer[l].neuron1d[j_x].gradient);

                    // cumulative loss (per neuron)
                    layer[l].neuron1d[j_x].loss_sum = vnum(layer[l].neuron1d[j_x].loss_sum + layer[l].neuron1d[j_x].loss);
                
                    // sum of squared gradients (this iteration)
                    layer[l].rSSG = vnum(layer[l].rSSG + pow(layer[l].neuron1d[j_x].gradient, 2));
                }
                // layer gradient (root of sum of squared gradients) as rolling average
                layer[l].rSSG=vnum(sqrt(layer[l].rSSG));
                layer[l].rSSG_average = vnum(layer[l].rSSG*0.999 + layer[l].rSSG*0.001);

                // adjust learning rate according to current loss relative to average
                if (gradient_dependent_attention){
                    lr=vnum(learning_rate*pow(layer[l].rSSG/(layer[l].rSSG_average+__DBL_EPSILON__),2));
                }
                else{
                    lr=learning_rate;
                }
            }
            else{
                // step 2.1: hidden errors, i.e. SUM_k[err_k*w_j_xk]
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    layer[l].neuron1d[j_x].gradient=0;
                    if (layer[l+1].dimensions==1){
                        for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                            layer[l].neuron1d[j_x].gradient = vnum(layer[l].neuron1d[j_x].gradient + layer[l+1].neuron1d[k_x].gradient * layer[l+1].neuron1d[k_x].input_weight_1d[j_x]);
                        }
                    }
                    else if (layer[l+1].dimensions==2){
                        for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                            for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                layer[l].neuron1d[j_x].gradient = vnum(layer[l].neuron1d[j_x].gradient + layer[l+1].neuron2d[k_x][k_y].gradient * layer[l+1].neuron2d[k_y][k_y].input_weight_1d[j_x]);
                            }
                        }
                    }
                    else if (layer[l+1].dimensions==3){
                        for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                            for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                for (int k_z=0;k_z<layer[l+1].neurons_z;k_z++){
                                    layer[l].neuron1d[j_x].gradient = vnum(layer[l].neuron1d[j_x].gradient + layer[l+1].neuron3d[k_x][k_y][k_z].gradient * layer[l+1].neuron3d[k_y][k_y][k_z].input_weight_1d[j_x]);
                                }
                            }
                        }
                    }                    
                }               
            }
        }

        //  +=======================+
        //  |     FOR 2D LAYERS     |
        //  +=======================+
        else if (layer[l].dimensions==2){
            if (l==layers-1){
                // step 1: output layer, global errors
                layer[l].rSSG = 0; // rSSG = root of sum of squared gradients
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                        // derivative of the 0.5err^2 loss function: scaled_label-output
                        layer[l].neuron2d[j_x][j_y].gradient = vnum(layer[l].neuron2d[j_x][j_y].scaled_label - layer[l].neuron2d[j_x][j_y].h);

                        // gradient clipping
                        if (gradient_clipping){
                            layer[l].neuron2d[j_x][j_y].gradient = fmin(layer[l].neuron2d[j_x][j_y].gradient, gradient_clipping_threshold);
                            layer[l].neuron2d[j_x][j_y].gradient = fmax(layer[l].neuron2d[j_x][j_y].gradient, -gradient_clipping_threshold);
                        }

                        // 0.5err^2 loss
                        layer[l].neuron2d[j_x][j_y].loss = vnum(0.5 * layer[l].neuron2d[j_x][j_y].gradient * layer[l].neuron2d[j_x][j_y].gradient);

                        // cumulative loss (per neuron)
                        layer[l].neuron2d[j_x][j_y].loss_sum = vnum(layer[l].neuron2d[j_x][j_y].loss_sum + layer[l].neuron2d[j_x][j_y].loss);

                        // sum of squared gradients
                        layer[l].rSSG = vnum(layer[l].rSSG + pow(layer[l].neuron2d[j_x][j_y].gradient,2));
                    }
                }
                
                // layer gradient (root of sum of squared gradients) as rolling average
                layer[l].rSSG = vnum(sqrt(layer[l].rSSG));
                layer[l].rSSG_average = vnum(layer[l].rSSG_average*0.999 + layer[l].rSSG*0.001);
                
                // adjust learning rate according to current loss relative to average
                if (gradient_dependent_attention){
                    lr=vnum(learning_rate*pow(layer[l].rSSG/(layer[l].rSSG_average+__DBL_EPSILON__),2));
                }
                else{
                    lr=learning_rate;
                }
            }
            else{
                // step 2.1: hidden errors, i.e. SUM_k[err_k*w_j_xk]
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                        layer[l].neuron2d[j_x][j_y].gradient=0;
                        if (layer[l+1].dimensions==1){
                            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                layer[l].neuron2d[j_x][j_y].gradient = vnum(layer[l].neuron2d[j_x][j_y].gradient + layer[l+1].neuron1d[k_x].gradient * layer[l+1].neuron1d[k_x].input_weight_2d[j_x][j_y]);
                            }
                        }
                        else if (layer[l+1].dimensions==2){
                            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                    layer[l].neuron2d[j_x][j_y].gradient = vnum(layer[l].neuron2d[j_x][j_y].gradient + layer[l+1].neuron2d[k_x][k_y].gradient * layer[l+1].neuron2d[k_y][k_y].input_weight_2d[j_x][j_y]);
                                }
                            }
                        }
                        else if (layer[l+1].dimensions==3){
                            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                    for (int k_z=0;k_z<layer[l+1].neurons_z;k_z++){
                                        layer[l].neuron2d[j_x][j_y].gradient = vnum(layer[l].neuron2d[j_x][j_y].gradient + layer[l+1].neuron3d[k_x][k_y][k_z].gradient * layer[l+1].neuron3d[k_y][k_y][k_z].input_weight_2d[j_x][j_y]);
                                    }
                                }
                            }
                        }     
                    }               
                }              
            }
        }

        //  +=======================+
        //  |     FOR 3D LAYERS     |
        //  +=======================+
        else if (layer[l].dimensions==3){
            if (l==layers-1){
                // step 1: output layer, global errors
                layer[l].rSSG = 0; // rSSG = root of sum of squared gradients
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                        for (int j_z=0;j_z<layer[l].neurons_z;j_z++){
                            // derivative of the 0.5err^2 loss function: scaled_label-output
                            layer[l].neuron3d[j_x][j_y][j_z].gradient = vnum(layer[l].neuron3d[j_x][j_y][j_z].scaled_label - layer[l].neuron3d[j_x][j_y][j_z].h);

                            // gradient clipping
                            if (gradient_clipping){
                                layer[l].neuron3d[j_x][j_y][j_z].gradient = fmin(layer[l].neuron3d[j_x][j_y][j_z].gradient, gradient_clipping_threshold);
                                layer[l].neuron3d[j_x][j_y][j_z].gradient = fmax(layer[l].neuron3d[j_x][j_y][j_z].gradient, -gradient_clipping_threshold);
                            }

                            // 0.5err^2 loss
                            layer[l].neuron3d[j_x][j_y][j_z].loss = vnum(0.5 * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l].neuron3d[j_x][j_y][j_z].gradient);

                            // cumulative loss (per neuron)
                            layer[l].neuron3d[j_x][j_y][j_z].loss_sum = vnum(layer[l].neuron3d[j_x][j_y][j_z].loss_sum + layer[l].neuron3d[j_x][j_y][j_z].loss);

                            // sum of squared gradients
                            layer[l].rSSG = vnum(layer[l].rSSG + pow(layer[l].neuron3d[j_x][j_y][j_z].gradient,2));
                        }
                    }
                }
                
                // layer gradient (root of sum of squared gradients) as rolling average
                layer[l].rSSG = vnum(sqrt(layer[l].rSSG));
                layer[l].rSSG_average = vnum(layer[l].rSSG_average*0.999 + layer[l].rSSG*0.001);
                
                // adjust learning rate according to current loss relative to average
                if (gradient_dependent_attention){
                    lr=vnum(learning_rate*pow(layer[l].rSSG/(layer[l].rSSG_average+__DBL_EPSILON__),2));
                }
                else{
                    lr=learning_rate;
                }
            }
            else{
                // step 2.1: hidden errors, i.e. SUM_k[err_k*w_j_xk]
                for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                    for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                        for (int j_z=0;j_z<layer[l].neurons_z;j_z++){
                            layer[l].neuron3d[j_x][j_y][j_z].gradient=0;
                            if (layer[l+1].dimensions==1){
                                for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                    layer[l].neuron3d[j_x][j_y][j_z].gradient = vnum(layer[l].neuron3d[j_x][j_y][j_z].gradient + layer[l+1].neuron1d[k_x].gradient * layer[l+1].neuron1d[k_x].input_weight_3d[j_x][j_y][j_z]);
                                }
                            }
                            else if (layer[l+1].dimensions==2){
                                for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                    for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                        layer[l].neuron3d[j_x][j_y][j_z].gradient = vnum(layer[l].neuron3d[j_x][j_y][j_z].gradient + layer[l+1].neuron2d[k_x][k_y].gradient * layer[l+1].neuron2d[k_y][k_y].input_weight_3d[j_x][j_y][j_z]);
                                    }
                                }
                            }
                            else if (layer[l+1].dimensions==3){
                                for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                                    for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                                        for (int k_z=0;k_z<layer[l+1].neurons_z;k_z++){
                                            layer[l].neuron3d[j_x][j_y][j_z].gradient = vnum(layer[l].neuron3d[j_x][j_y][j_z].gradient + layer[l+1].neuron3d[k_x][k_y][k_z].gradient * layer[l+1].neuron3d[k_y][k_y][k_z].input_weight_3d[j_x][j_y][j_z]);
                                        }
                                    }
                                }
                            }         
                        }
                    }           
                }              
            }
        }

        // step 2.2: bias error
        layer[l].bias_error=0;
        if (layer[l+1].dimensions==1){
            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                layer[l].bias_error = vnum(layer[l].bias_error + layer[l+1].neuron1d[k_x].gradient * layer[l+1].neuron1d[k_x].bias_weight);
            }
        }
        else if (layer[l+1].dimensions==2){
            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                    layer[l].bias_error = vnum(layer[l].bias_error + layer[l+1].neuron2d[k_x][k_y].gradient * layer[l+1].neuron2d[k_x][k_y].bias_weight);
                }
            }
        }
        else if (layer[l+1].dimensions==3){
            for (int k_x=0;k_x<layer[l+1].neurons_x;k_x++){
                for (int k_y=0;k_y<layer[l+1].neurons_y;k_y++){
                    for (int k_z=0;k_z<layer[l+1].neurons_z;k_z++){
                        layer[l].bias_error = vnum(layer[l].bias_error + layer[l+1].neuron3d[k_x][k_y][k_z].gradient * layer[l+1].neuron3d[k_x][k_y][k_z].bias_weight);
                    }
                }
            }
        }          
    }
    // step 3.1: weight updates
    for (int l=layers-1;l>=1;l--){
        // delta rule for output neurons: delta w_ij=lr*(label-output)*act'(net_inp_j)*out_i
        // delta rule for hidden neurons: delta w_ij=lr*SUM_k[err_k*w_jk]*act'(net_inp_j)*out_i
        // --> general rule: delta w_ij=lr*error*act'(net_inp_j)*out_i

        // for 1d layer
        if (layer[l].dimensions==1){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){

                // ...with 1d preceding layer
                if (layer[l-1].dimensions==1){                    
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                        
                        if (opt_method==Vanilla){
                            layer[l].neuron1d[j_x].delta_w_1d[i_x] = vnum(
                                    (lr_momentum*layer[l].neuron1d[j_x].delta_w_1d[i_x])
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron1d[j_x].gradient
                                * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                * layer[l-1].neuron1d[i_x].h);
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x] + layer[l].neuron1d[j_x].delta_w_1d[i_x]);
                        }

                        else if (opt_method==Nesterov){
                            // lookahead step
                            double lookahead = vnum(
                                    (lr_momentum*layer[l].neuron1d[j_x].delta_w_1d[i_x])
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron1d[j_x].gradient
                                * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                * layer[l-1].neuron1d[i_x].h);
                            // momentum step
                            layer[l].neuron1d[j_x].delta_w_1d[i_x] = vnum(
                                    (lr_momentum*lookahead)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron1d[j_x].gradient
                                * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                * layer[l-1].neuron1d[i_x].h);
                            // update step
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x] + layer[l].neuron1d[j_x].delta_w_1d[i_x]);
                        }

                        else if (opt_method==RMSprop){
                            layer[l].neuron1d[j_x].opt_v_1d[i_x] = vnum(
                                    lr_momentum*layer[l].neuron1d[j_x].opt_v_1d[i_x]
                                + (1-lr_momentum)*pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation),2)
                                * layer[l].neuron1d[j_x].gradient);
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(
                                    layer[l].neuron1d[j_x].input_weight_1d[i_x]
                                + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron1d[j_x].opt_v_1d[i_x]+1e-8)+__DBL_MIN__))
                                * (layer[l].neuron1d[j_x].h * (1-layer[l].neuron1d[j_x].h))
                                * layer[l].neuron1d[j_x].gradient
                                * layer[l-1].neuron1d[i_x].h);
                        }

                        else if (opt_method==ADADELTA){
                            layer[l].neuron1d[j_x].opt_v_1d[i_x] = vnum(
                                    opt_beta1 * layer[l].neuron1d[j_x].opt_v_1d[i_x]
                                + (1-opt_beta1) * pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                * layer[l].neuron1d[j_x].gradient
                                * layer[l-1].neuron1d[i_x].h,2));
                            layer[l].neuron1d[j_x].opt_w_1d[i_x] = vnum(
                                    (opt_beta1 * pow(layer[l].neuron1d[j_x].opt_w_1d[i_x],2))
                                + (1-opt_beta1)*pow(layer[l].neuron1d[j_x].delta_w_1d[i_x],2));
                            layer[l].neuron1d[j_x].delta_w_1d[i_x] = vnum(
                                    sqrt(layer[l].neuron1d[j_x].opt_w_1d[i_x]+1e-8)/(sqrt(layer[l].neuron1d[j_x].opt_v_1d[i_x]+1e-8)+__DBL_MIN__)
                                * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron1d[i_x].h);
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x] + layer[l].neuron1d[j_x].delta_w_1d[i_x]);
                        }

                        else if (opt_method==ADAM){ // =ADAM without minibatch
                            layer[l].neuron1d[j_x].opt_v_1d[i_x] = vnum(
                                    opt_beta1 * layer[l].neuron1d[j_x].opt_v_1d[i_x]
                                + (1-opt_beta1) * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron1d[i_x].h);
                            layer[l].neuron1d[j_x].opt_w_1d[i_x] = vnum(
                                    opt_beta2 * layer[l].neuron1d[j_x].opt_w_1d[i_x]
                                * pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron1d[i_x].h,2));
                            double v_t = vnum(layer[l].neuron1d[j_x].opt_v_1d[i_x]/(1-opt_beta1));
                            double w_t = vnum(layer[l].neuron1d[j_x].opt_w_1d[i_x]/(1-opt_beta2));
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                        }

                        else if (opt_method==AdaGrad){
                            layer[l].neuron1d[j_x].opt_v_1d[i_x] = vnum(
                                    layer[l].neuron1d[j_x].opt_v_1d[i_x]
                                + pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron1d[i_x].h,2));
                            layer[l].neuron1d[j_x].input_weight_1d[i_x] = vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x] + 
                                    ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron1d[j_x].opt_v_1d[i_x]+1e-8))
                                * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                * layer[l].neuron1d[j_x].gradient
                                * layer[l-1].neuron1d[i_x].h);
                        }
                    }           
                }
            

                // ... with 2d preceding layer
                if (layer[l-1].dimensions==2){                    
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){

                            if (opt_method==Vanilla){
                                layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y] = vnum(
                                        (lr_momentum*layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y])
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron1d[j_x].gradient
                                    * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                    * layer[l-1].neuron2d[i_x][i_y].h);
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] + layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y]);
                            }

                            else if (opt_method==Nesterov){
                                // lookahead step
                                double lookahead = vnum(
                                        (lr_momentum*layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y])
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron1d[j_x].gradient
                                    * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                    * layer[l-1].neuron2d[i_x][i_y].h);
                                // momentum step
                                layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y] = vnum(
                                        (lr_momentum*lookahead)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron1d[j_x].gradient
                                    * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                    * layer[l-1].neuron2d[i_x][i_y].h);
                                // update step
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] + layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y]);
                            }

                            else if (opt_method==RMSprop){
                                layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y] = vnum(
                                        lr_momentum*layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]
                                    + (1-lr_momentum)*pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation),2)
                                    * layer[l].neuron1d[j_x].gradient);
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(
                                        layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y]
                                    + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__))
                                    * (layer[l].neuron1d[j_x].h * (1-layer[l].neuron1d[j_x].h))
                                    * layer[l].neuron1d[j_x].gradient
                                    * layer[l-1].neuron2d[i_x][i_y].h);
                            }

                            else if (opt_method==ADADELTA){
                                layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y] = vnum(
                                        opt_beta1 * layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]
                                    + (1-opt_beta1) * pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                    * layer[l].neuron1d[j_x].gradient
                                    * layer[l-1].neuron2d[i_x][i_y].h,2));
                                layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y] = vnum(
                                        (opt_beta1 * pow(layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y],2))
                                    + (1-opt_beta1)*pow(layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y],2));
                                layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y] = vnum(
                                        sqrt(layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y]+1e-8)/(sqrt(layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__)
                                    * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                    * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] + layer[l].neuron1d[j_x].delta_w_2d[i_x][i_y]);
                            }

                            else if (opt_method==ADAM){ // =ADAM without minibatch
                                layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y] = vnum(
                                        opt_beta1 * layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]
                                    + (1-opt_beta1) * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                    * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y] = vnum(
                                        opt_beta2 * layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y]
                                    * pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                double v_t = vnum(layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]/(1-opt_beta1));
                                double w_t = vnum(layer[l].neuron1d[j_x].opt_w_2d[i_x][i_y]/(1-opt_beta2));
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                            }

                            else if (opt_method==AdaGrad){
                                layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y] = vnum(
                                        layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]
                                    + pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y] +
                                        ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron1d[j_x].opt_v_2d[i_x][i_y]+1e-8))
                                    * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                    * layer[l].neuron1d[j_x].gradient
                                    * layer[l-1].neuron2d[i_x][i_y].h);
                            }
                        }
                    }           
                }

                // ... with 3d preceding layer
                if (layer[l-1].dimensions==3){                    
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){

                                if (opt_method==Vanilla){
                                    layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z] = vnum(
                                            (lr_momentum*layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron1d[j_x].gradient
                                        * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    layer[l].neuron1d[j_x].input_weight_3d[i_y][i_y][i_z] = vnum(layer[l].neuron1d[j_x].input_weight_3d[i_y][i_y][i_z] + layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z]);
                                }

                                else if (opt_method==Nesterov){
                                    // lookahead step
                                    double lookahead = vnum(
                                            (lr_momentum*layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron1d[j_x].gradient
                                        * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    // momentum step
                                    layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z] = vnum(
                                            (lr_momentum*lookahead)
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron1d[j_x].gradient
                                        * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    // update step
                                    layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z]);
                                }

                                else if (opt_method==RMSprop){
                                    layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z] = vnum(
                                            lr_momentum*layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]
                                        + (1-lr_momentum)*pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation),2)
                                        * layer[l].neuron1d[j_x].gradient);
                                    layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] = vnum(
                                            layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z]
                                        + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__))
                                        * (layer[l].neuron1d[j_x].h * (1-layer[l].neuron1d[j_x].h))
                                        * layer[l].neuron1d[j_x].gradient
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                }

                                else if (opt_method==ADADELTA){
                                    layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z] = vnum(
                                            opt_beta1 * layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]
                                        + (1-opt_beta1) * pow(deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                        * layer[l].neuron1d[j_x].gradient
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                    layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z] = vnum(
                                            (opt_beta1 * pow(layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z],2))
                                        + (1-opt_beta1)*pow(layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z],2));
                                    layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z] = vnum(
                                            sqrt(layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z]+1e-8)/(sqrt(layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__)
                                        * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                                        * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron1d[j_x].delta_w_3d[i_x][i_y][i_z]);
                                }

                                else if (opt_method==ADAM){ // =ADAM without minibatch
                                    layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z] = vnum(
                                            opt_beta1 * layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]
                                        + (1-opt_beta1) * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                        * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z] = vnum(
                                            opt_beta2 * layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z]
                                        * pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                    double v_t = vnum(layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]/(1-opt_beta1));
                                    double w_t = vnum(layer[l].neuron1d[j_x].opt_w_3d[i_x][i_y][i_z]/(1-opt_beta2));
                                    layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                }

                                else if (opt_method==AdaGrad){
                                    layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z] = vnum(
                                            layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]
                                        + pow(deactivate(layer[l].neuron1d[j_x].x, layer[l].activation) * layer[l].neuron1d[j_x].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                    layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z] +
                                            ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron1d[j_x].opt_v_3d[i_x][i_y][i_z]+1e-8))
                                        * deactivate(layer[l].neuron1d[j_x].x, layer[l].activation)
                                        * layer[l].neuron1d[j_x].gradient
                                        * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                }
                            }
                        }
                    }           
                }
                

                // step 3.2: bias updates (for simplicity reasons always using Vanilla method for this purpose)
                layer[l].neuron1d[j_x].delta_b = vnum( 
                        (lr_momentum*layer[l].neuron1d[j_x].delta_b)
                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                        * layer[l].bias_error
                        * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation));
                layer[l].neuron1d[j_x].bias_weight = vnum(layer[l].neuron1d[j_x].bias_weight + layer[l].neuron1d[j_x].delta_b);

                // step 3.3: recurrent weight updates (Vanilla method only)
                if (layer[l].type==recurrent || layer[l].type==mod_recurrent){
                    // update m1 weight
                    layer[l].neuron1d[j_x].delta_m1 = vnum(
                            (lr_momentum*layer[l].neuron1d[j_x].delta_m1)
                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                            * layer[l].neuron1d[j_x].gradient
                            * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                            * layer[l].neuron1d[j_x].m1);
                    layer[l].neuron1d[j_x].m1_weight = vnum(layer[l].neuron1d[j_x].m1_weight + layer[l].neuron1d[j_x].delta_m1);
                }            
                if (layer[l].type==mod_recurrent){
                    // update m2 weight
                    layer[l].neuron1d[j_x].delta_m2 = vnum(
                            (lr_momentum*layer[l].neuron1d[j_x].delta_m2)
                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                            * layer[l].neuron1d[j_x].gradient
                            * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                            * layer[l].neuron1d[j_x].m2);
                    layer[l].neuron1d[j_x].m2_weight = vnum(layer[l].neuron1d[j_x].m2_weight + layer[l].neuron1d[j_x].delta_m2);
                    // update m3 weight
                    layer[l].neuron1d[j_x].delta_m3 = vnum( 
                            (lr_momentum*layer[l].neuron1d[j_x].delta_m3)
                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                            * layer[l].neuron1d[j_x].gradient
                            * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                            * layer[l].neuron1d[j_x].m3);
                    layer[l].neuron1d[j_x].m3_weight = vnum(layer[l].neuron1d[j_x].m3_weight + layer[l].neuron1d[j_x].delta_m3);
                    // update m4 weight
                    layer[l].neuron1d[j_x].delta_m4 = vnum( 
                            (lr_momentum*layer[l].neuron1d[j_x].delta_m4)
                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                            * layer[l].neuron1d[j_x].gradient
                            * deactivate(layer[l].neuron1d[j_x].x,layer[l].activation)
                            * layer[l].neuron1d[j_x].m4);
                    layer[l].neuron1d[j_x].m4_weight = vnum(layer[l].neuron1d[j_x].m4_weight + layer[l].neuron1d[j_x].delta_m4);
                }    
            }
        }

        // ===========================
        // weight updates for 2d layer
        // ===========================
        else if (layer[l].dimensions==2){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){

                    // ...with 1d preceding layer
                    if (layer[l-1].dimensions==1){                    
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                            
                            if (opt_method==Vanilla){
                                layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x] = vnum( 
                                        (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x])
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                    * layer[l-1].neuron1d[i_x].h);
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] + layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x]);
                            }

                            else if (opt_method==Nesterov){
                                // lookahead step
                                double lookahead = vnum(
                                        (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x])
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                    * layer[l-1].neuron1d[i_x].h);
                                // momentum step
                                layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x] = vnum(
                                        (lr_momentum*lookahead)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                    * layer[l-1].neuron1d[i_x].h);
                                // update step
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] + layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x]);
                            }

                            else if (opt_method==RMSprop){
                                layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x] = vnum(
                                        lr_momentum*layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]
                                    + (1-lr_momentum)*pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation),2)
                                    * layer[l].neuron2d[j_x][j_y].gradient);
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(
                                        layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x]
                                    + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]+1e-8)+__DBL_MIN__))
                                    * (layer[l].neuron2d[j_x][j_y].h * (1-layer[l].neuron2d[j_x][j_y].h))
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * layer[l-1].neuron1d[i_x].h);
                            }

                            else if (opt_method==ADADELTA){
                                layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x] = vnum(
                                        opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]
                                    + (1-opt_beta1) * pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * layer[l-1].neuron1d[i_x].h,2));
                                layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x] = vnum(
                                        (opt_beta1 * pow(layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x],2))
                                    + (1-opt_beta1)*pow(layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x],2));
                                layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x] = vnum(
                                        sqrt(layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x]+1e-8)/(sqrt(layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]+1e-8)+__DBL_MIN__)
                                    * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                    * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron1d[i_x].h);
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] + layer[l].neuron2d[j_x][j_y].delta_w_1d[i_x]);
                            }

                            else if (opt_method==ADAM){ // =ADAM without minibatch
                                layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x] = vnum(
                                        opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]
                                    + (1-opt_beta1) * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                    * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron1d[i_x].h);
                                layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x] = vnum(
                                        opt_beta2 * layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x]
                                    * pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron1d[i_x].h,2));
                                double v_t = vnum(layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]/(1-opt_beta1));
                                double w_t = vnum(layer[l].neuron2d[j_x][j_y].opt_w_1d[i_x]/(1-opt_beta2));
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                            }

                            else if (opt_method==AdaGrad){
                                layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x] = vnum(
                                        layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]
                                    + pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron1d[i_x].h,2));
                                layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x] +
                                        ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron2d[j_x][j_y].opt_v_1d[i_x]+1e-8))
                                    * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                    * layer[l].neuron2d[j_x][j_y].gradient
                                    * layer[l-1].neuron1d[i_x].h);
                            }
                        }           
                    }
                

                    // ... with 2d preceding layer
                    if (layer[l-1].dimensions==2){                    
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){

                                if (opt_method==Vanilla){
                                    layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y] = vnum(
                                            (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                        * layer[l-1].neuron2d[i_x][i_y].h);
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] + layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y]);
                                }

                                else if (opt_method==Nesterov){
                                    // lookahead step
                                    double lookahead = vnum(
                                            (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                        * layer[l-1].neuron2d[i_x][i_y].h);
                                    // momentum step
                                    layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y] = vnum( 
                                            (lr_momentum*lookahead)
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                        * layer[l-1].neuron2d[i_x][i_y].h);
                                    // update step
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] + layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y]);
                                }

                                else if (opt_method==RMSprop){
                                    layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y] = vnum(
                                            lr_momentum*layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]
                                        + (1-lr_momentum)*pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation),2)
                                        * layer[l].neuron2d[j_x][j_y].gradient);
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(
                                            layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y]
                                        + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__))
                                        * (layer[l].neuron2d[j_x][j_y].h * (1-layer[l].neuron2d[j_x][j_y].h))
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * layer[l-1].neuron2d[i_x][i_y].h);
                                }

                                else if (opt_method==ADADELTA){
                                    layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y] = vnum(
                                            opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]
                                        + (1-opt_beta1) * pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * layer[l-1].neuron2d[i_x][i_y].h,2));
                                    layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y] = vnum(
                                            (opt_beta1 * pow(layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y],2))
                                        + (1-opt_beta1)*pow(layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y],2));
                                    layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y] = vnum(
                                            sqrt(layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y]+1e-8)/(sqrt(layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__)
                                        * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                        * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] + layer[l].neuron2d[j_x][j_y].delta_w_2d[i_x][i_y]);
                                }

                                else if (opt_method==ADAM){ // =ADAM without minibatch
                                    layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y] = vnum(
                                            opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]
                                        + (1-opt_beta1) * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                        * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                    layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y] = vnum(
                                            opt_beta2 * layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y]
                                        * pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                    double v_t = vnum(layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]/(1-opt_beta1));
                                    double w_t = vnum(layer[l].neuron2d[j_x][j_y].opt_w_2d[i_x][i_y]/(1-opt_beta2));
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                }

                                else if (opt_method==AdaGrad){
                                    layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y] = vnum(
                                            layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]
                                        + pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                    layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y] +
                                            ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron2d[j_x][j_y].opt_v_2d[i_x][i_y]+1e-8))
                                        * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                        * layer[l].neuron2d[j_x][j_y].gradient
                                        * layer[l-1].neuron2d[i_x][i_y].h);
                                }
                            }
                        }           
                    }

                    // ... with 3d preceding layer
                    if (layer[l-1].dimensions==3){                    
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){

                                    if (opt_method==Vanilla){
                                        layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z])
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_y][i_y][i_z] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_y][i_y][i_z] + layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z]);
                                    }

                                    else if (opt_method==Nesterov){
                                        // lookahead step
                                        double lookahead = vnum(
                                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z])
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        // momentum step
                                        layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                (lr_momentum*lookahead)
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        // update step
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z]);
                                    }

                                    else if (opt_method==RMSprop){
                                        layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                lr_momentum*layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]
                                            + (1-lr_momentum)*pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation),2)
                                            * layer[l].neuron2d[j_x][j_y].gradient);
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] = vnum(
                                                layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z]
                                            + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__))
                                            * (layer[l].neuron2d[j_x][j_y].h * (1-layer[l].neuron2d[j_x][j_y].h))
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    }

                                    else if (opt_method==ADADELTA){
                                        layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]
                                            + (1-opt_beta1) * pow(deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                        layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z] = vnum( 
                                                (opt_beta1 * pow(layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z],2))
                                            + (1-opt_beta1)*pow(layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z],2));
                                        layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                sqrt(layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z]+1e-8)/(sqrt(layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__)
                                            * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                            * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron2d[j_x][j_y].delta_w_3d[i_x][i_y][i_z]);
                                    }

                                    else if (opt_method==ADAM){ // =ADAM without minibatch
                                        layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                opt_beta1 * layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]
                                            + (1-opt_beta1) * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                            * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z] = vnum(
                                                opt_beta2 * layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z]
                                            * pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                        double v_t = vnum(layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]/(1-opt_beta1));
                                        double w_t = vnum(layer[l].neuron2d[j_x][j_y].opt_w_3d[i_x][i_y][i_z]/(1-opt_beta2));
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                    }

                                    else if (opt_method==AdaGrad){
                                        layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]
                                            + pow(deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation) * layer[l].neuron2d[j_x][j_y].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                        layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z] +
                                                ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron2d[j_x][j_y].opt_v_3d[i_x][i_y][i_z]+1e-8))
                                            * deactivate(layer[l].neuron2d[j_x][j_y].x, layer[l].activation)
                                            * layer[l].neuron2d[j_x][j_y].gradient
                                            * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                    }
                                }
                            }
                        }           
                    }
                    

                    // step 3.2: bias updates (for simplicity reasons always using Vanilla method for this purpose)
                    layer[l].neuron2d[j_x][j_y].delta_b = vnum(
                            (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_b)
                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                            * layer[l].bias_error
                            * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation));
                    layer[l].neuron2d[j_x][j_y].bias_weight = vnum(layer[l].neuron2d[j_x][j_y].bias_weight + layer[l].neuron2d[j_x][j_y].delta_b);

                    // step 3.3: recurrent weight updates (Vanilla method only)
                    if (layer[l].type==recurrent || layer[l].type==mod_recurrent){
                        // update m1 weight
                        layer[l].neuron2d[j_x][j_y].delta_m1 = vnum(
                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_m1)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron2d[j_x][j_y].gradient
                                * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                * layer[l].neuron2d[j_x][j_y].m1);
                        layer[l].neuron2d[j_x][j_y].m1_weight = vnum(layer[l].neuron2d[j_x][j_y].m1_weight + layer[l].neuron2d[j_x][j_y].delta_m1);
                    }            
                    if (layer[l].type==mod_recurrent){
                        // update m2 weight
                        layer[l].neuron2d[j_x][j_y].delta_m2 = vnum(
                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_m2)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron2d[j_x][j_y].gradient
                                * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                * layer[l].neuron2d[j_x][j_y].m2);
                        layer[l].neuron2d[j_x][j_y].m2_weight = vnum(layer[l].neuron2d[j_x][j_y].m2_weight + layer[l].neuron2d[j_x][j_y].delta_m2);
                        // update m3 weight
                        layer[l].neuron2d[j_x][j_y].delta_m3 = vnum(
                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_m3)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron2d[j_x][j_y].gradient
                                * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                * layer[l].neuron2d[j_x][j_y].m3);
                        layer[l].neuron2d[j_x][j_y].m3_weight = vnum(layer[l].neuron2d[j_x][j_y].m3_weight + layer[l].neuron2d[j_x][j_y].delta_m3);
                        // update m4 weight
                        layer[l].neuron2d[j_x][j_y].delta_m4 = vnum(
                                (lr_momentum*layer[l].neuron2d[j_x][j_y].delta_m4)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].neuron2d[j_x][j_y].gradient
                                * deactivate(layer[l].neuron2d[j_x][j_y].x,layer[l].activation)
                                * layer[l].neuron2d[j_x][j_y].m4);
                        layer[l].neuron2d[j_x][j_y].m4_weight = vnum(layer[l].neuron2d[j_x][j_y].m4_weight + layer[l].neuron2d[j_x][j_y].delta_m4);
                    }   
                }        
            }
        }

        // ===========================
        // weight updates for 3d layer
        // ===========================
        else if (layer[l].dimensions==3){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    for (int j_z=0;j_z<layer[l].neurons_z;j_z++){

                        // ...with 1d preceding layer
                        if (layer[l-1].dimensions==1){                    
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                                
                                if (opt_method==Vanilla){
                                    layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x] = vnum( 
                                            (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                        * layer[l-1].neuron1d[i_x].h);
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x]);
                                }

                                else if (opt_method==Nesterov){
                                    // lookahead step
                                    double lookahead = vnum(
                                            (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x])
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                        * layer[l-1].neuron1d[i_x].h);
                                    // momentum step
                                    layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x] = vnum( 
                                            (lr_momentum*lookahead)
                                        + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                        * layer[l-1].neuron1d[i_x].h);
                                    // update step
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x]);
                                }

                                else if (opt_method==RMSprop){
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x] = vnum(
                                            lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]
                                        + (1-lr_momentum)*pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation),2)
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient);
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x]
                                        + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]+1e-8)+__DBL_MIN__))
                                        * (layer[l].neuron3d[j_x][j_y][j_z].h * (1-layer[l].neuron3d[j_x][j_y][j_z].h))
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * layer[l-1].neuron1d[i_x].h);
                                }

                                else if (opt_method==ADADELTA){
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x] = vnum(
                                            opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]
                                        + (1-opt_beta1) * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * layer[l-1].neuron1d[i_x].h,2));
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x] = vnum(
                                            (opt_beta1 * pow(layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x],2))
                                        + (1-opt_beta1)*pow(layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x],2));
                                    layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x] = vnum(
                                            sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x]+1e-8)/(sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]+1e-8)+__DBL_MIN__)
                                        * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron1d[i_x].h);
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_1d[i_x]);
                                }

                                else if (opt_method==ADAM){ // =ADAM without minibatch
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x] = vnum(
                                            opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]
                                        + (1-opt_beta1) * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron1d[i_x].h);
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x] = vnum(
                                            opt_beta2 * layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x]
                                        * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron1d[i_x].h,2));
                                    double v_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]/(1-opt_beta1));
                                    double w_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_w_1d[i_x]/(1-opt_beta2));
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                }

                                else if (opt_method==AdaGrad){
                                    layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x] = vnum(
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]
                                        + pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron1d[i_x].h,2));
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x] +
                                            ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_1d[i_x]+1e-8))
                                        * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                        * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                        * layer[l-1].neuron1d[i_x].h);
                                }
                            }           
                        }
                    
                        // ... with 2d preceding layer
                        if (layer[l-1].dimensions==2){                    
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){

                                    if (opt_method==Vanilla){
                                        layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y] = vnum(
                                                (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y])
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                            * layer[l-1].neuron2d[i_x][i_y].h);
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y]);
                                    }

                                    else if (opt_method==Nesterov){
                                        // lookahead step
                                        double lookahead = vnum( 
                                                (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y])
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                            * layer[l-1].neuron2d[i_x][i_y].h);
                                        // momentum step
                                        layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y] = vnum(
                                                (lr_momentum*lookahead)
                                            + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                            * layer[l-1].neuron2d[i_x][i_y].h);
                                        // update step
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y]);
                                    }

                                    else if (opt_method==RMSprop){
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y] = vnum(
                                                lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]
                                            + (1-lr_momentum)*pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation),2)
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient);
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(
                                                layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y]
                                            + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__))
                                            * (layer[l].neuron3d[j_x][j_y][j_z].h * (1-layer[l].neuron3d[j_x][j_y][j_z].h))
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * layer[l-1].neuron2d[i_x][i_y].h);
                                    }

                                    else if (opt_method==ADADELTA){
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y] = vnum(
                                                opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]
                                            + (1-opt_beta1) * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * layer[l-1].neuron2d[i_x][i_y].h,2));
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y] = vnum(
                                                (opt_beta1 * pow(layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y],2))
                                            + (1-opt_beta1)*pow(layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y],2));
                                        layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y] = vnum(
                                                sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y]+1e-8)/(sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]+1e-8)+__DBL_MIN__)
                                            * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_2d[i_x][i_y]);
                                    }

                                    else if (opt_method==ADAM){ // =ADAM without minibatch
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y] = vnum(
                                                opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]
                                            + (1-opt_beta1) * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron2d[i_x][i_y].h);
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y] = vnum(
                                                opt_beta2 * layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y]
                                            * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                        double v_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]/(1-opt_beta1));
                                        double w_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_w_2d[i_x][i_y]/(1-opt_beta2));
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                    }

                                    else if (opt_method==AdaGrad){
                                        layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y] = vnum(
                                                layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]
                                            + pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron2d[i_x][i_y].h,2));
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y] + 
                                                ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_2d[i_x][i_y]+1e-8))
                                            * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                            * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                            * layer[l-1].neuron2d[i_x][i_y].h);
                                    }
                                }
                            }           
                        }

                        // ... with 3d preceding layer
                        if (layer[l-1].dimensions==3){                    
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){ 
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){

                                        if (opt_method==Vanilla){
                                            layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z])
                                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_y][i_y][i_z] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_y][i_y][i_z] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z]);
                                        }

                                        else if (opt_method==Nesterov){
                                            // lookahead step
                                            double lookahead = vnum(
                                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z])
                                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                            // momentum step
                                            layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                    (lr_momentum*lookahead)
                                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                            // update step
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z]);
                                        }

                                        else if (opt_method==RMSprop){
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                    lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]
                                                + (1-lr_momentum)*pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation),2)
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient);
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] = vnum(
                                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z]
                                                + ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__))
                                                * (layer[l].neuron3d[j_x][j_y][j_z].h * (1-layer[l].neuron3d[j_x][j_y][j_z].h))
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        }

                                        else if (opt_method==ADADELTA){
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                    opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]
                                                + (1-opt_beta1) * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z] = vnum(
                                                    (opt_beta1 * pow(layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z],2))
                                                + (1-opt_beta1)*pow(layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z],2));
                                            layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z] = vnum(
                                                    sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z]+1e-8)/(sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]+1e-8)+__DBL_MIN__)
                                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] + layer[l].neuron3d[j_x][j_y][j_z].delta_w_3d[i_x][i_y][i_z]);
                                        }

                                        else if (opt_method==ADAM){ // =ADAM without minibatch
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                    opt_beta1 * layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]
                                                + (1-opt_beta1) * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z] = vnum(
                                                    opt_beta2 * layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z]
                                                * pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                            double v_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]/(1-opt_beta1));
                                            double w_t = vnum(layer[l].neuron3d[j_x][j_y][j_z].opt_w_3d[i_x][i_y][i_z]/(1-opt_beta2));
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] + (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                                        }

                                        else if (opt_method==AdaGrad){
                                            layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z] = vnum(
                                                    layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]
                                                + pow(deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation) * layer[l].neuron3d[j_x][j_y][j_z].gradient * layer[l-1].neuron3d[i_x][i_y][i_z].h,2));
                                            layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] = vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z] +
                                                    ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron3d[j_x][j_y][j_z].opt_v_3d[i_x][i_y][i_z]+1e-8))
                                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x, layer[l].activation)
                                                * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                                * layer[l-1].neuron3d[i_x][i_y][i_z].h);
                                        }
                                    }
                                }
                            }           
                        }
                        
                        // step 3.2: bias updates (for simplicity reasons always using Vanilla method for this purpose)
                        layer[l].neuron3d[j_x][j_y][j_z].delta_b = vnum( 
                                (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_b)
                                + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                * layer[l].bias_error
                                * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation));
                        layer[l].neuron3d[j_x][j_y][j_z].bias_weight = vnum(layer[l].neuron3d[j_x][j_y][j_z].bias_weight + layer[l].neuron3d[j_x][j_y][j_z].delta_b);

                        // step 3.3: recurrent weight updates (Vanilla method only)
                        if (layer[l].type==recurrent || layer[l].type==mod_recurrent){
                            // update m1 weight
                            layer[l].neuron3d[j_x][j_y][j_z].delta_m1 = vnum(
                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_m1)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                    * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                    * layer[l].neuron3d[j_x][j_y][j_z].m1);
                            layer[l].neuron3d[j_x][j_y][j_z].m1_weight = vnum(layer[l].neuron3d[j_x][j_y][j_z].m1_weight + layer[l].neuron3d[j_x][j_y][j_z].delta_m1);
                        }            
                        if (layer[l].type==mod_recurrent){
                            // update m2 weight
                            layer[l].neuron3d[j_x][j_y][j_z].delta_m2 = vnum(
                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_m2)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                    * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                    * layer[l].neuron3d[j_x][j_y][j_z].m2);
                            layer[l].neuron3d[j_x][j_y][j_z].m2_weight = vnum(layer[l].neuron3d[j_x][j_y][j_z].m2_weight + layer[l].neuron3d[j_x][j_y][j_z].delta_m2);
                            // update m3 weight
                            layer[l].neuron3d[j_x][j_y][j_z].delta_m3 = vnum(
                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_m3)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                    * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                    * layer[l].neuron3d[j_x][j_y][j_z].m3);
                            layer[l].neuron3d[j_x][j_y][j_z].m3_weight = vnum(layer[l].neuron3d[j_x][j_y][j_z].m3_weight + layer[l].neuron3d[j_x][j_y][j_z].delta_m3);
                            // update m4 weight
                            layer[l].neuron3d[j_x][j_y][j_z].delta_m4 = vnum(
                                    (lr_momentum*layer[l].neuron3d[j_x][j_y][j_z].delta_m4)
                                    + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations))
                                    * layer[l].neuron3d[j_x][j_y][j_z].gradient
                                    * deactivate(layer[l].neuron3d[j_x][j_y][j_z].x,layer[l].activation)
                                    * layer[l].neuron3d[j_x][j_y][j_z].m4);
                            layer[l].neuron3d[j_x][j_y][j_z].m4_weight = vnum(layer[l].neuron3d[j_x][j_y][j_z].m4_weight + layer[l].neuron3d[j_x][j_y][j_z].delta_m4);     
                        }    
                    }
                }
            }
        }
    }
}

double NeuralNet::get_loss_avg(){
    double result=0;
    // for 1d output layer
    if (layer[layers-1].dimensions==1){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            result+=layer[layers-1].neuron1d[x].loss_sum/backprop_iterations;
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 2d output layer
    else if (layer[layers-1].dimensions==2){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                result+=layer[layers-1].neuron2d[x][y].loss_sum/backprop_iterations;
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 3d output layer
    else if (layer[layers-1].dimensions==3){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                for (int z=0;z<layer[layers-1].neurons_z;z++){
                    result+=layer[layers-1].neuron3d[x][y][z].loss_sum/backprop_iterations;
                }
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // default
    return result;
}

double NeuralNet::get_avg_h(){
    double result=0;
    // for 1d output layer
    if (layer[layers-1].dimensions==1){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            result+=layer[layers-1].neuron1d[x].h;
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 2d output layer
    else if (layer[layers-1].dimensions==2){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                result+=layer[layers-1].neuron2d[x][y].h;
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 3d output layer
    else if (layer[layers-1].dimensions==3){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                for (int z=0;z<layer[layers-1].neurons_z;z++){
                    result+=layer[layers-1].neuron3d[x][y][z].h;
                }
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // default
    return result;
}

double NeuralNet::get_avg_output(){
    double result=0;
    // for 1d output layer
    if (layer[layers-1].dimensions==1){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            result+=layer[layers-1].neuron1d[x].output;
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 2d output layer
    else if (layer[layers-1].dimensions==2){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                result+=layer[layers-1].neuron2d[x][y].output;
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // for 3d output layer
    else if (layer[layers-1].dimensions==3){
        for (int x=0;x<layer[layers-1].neurons_x;x++){
            for (int y=0;y<layer[layers-1].neurons_y;y++){
                for (int z=0;z<layer[layers-1].neurons_z;z++){
                    result+=layer[layers-1].neuron3d[x][y][z].output;
                }
            }
        }
        result/=layer[layers-1].neurons_total;
        return vnum(result);
    }
    // default
    return result;
}

// balance weights between layers
void NeuralNet::balance_weights(){
    // step 1: get average sum of weights per neuron
    double total=0;
    for (int l=1;l<layers;l++){
        double result=0;
        if (layer[l].dimensions==1){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                if (layer[l-1].dimensions==1){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        result+=layer[l].neuron1d[j_x].input_weight_1d[i_x];
                    }
                }
                else if(layer[l-1].dimensions==2){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            result+=layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y];
                        }
                    }
                }
                else if (layer[l-1].dimensions==3){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                result+=layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z];
                            }
                        }
                    }
                }
            }
        }
        else if (layer[l].dimensions==2){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    if (layer[l-1].dimensions==1){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            result+=layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x];
                        }
                    }
                    else if(layer[l-1].dimensions==2){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                result+=layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y];
                            }
                        }
                    }
                    else if (layer[l-1].dimensions==3){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                    result+=layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z];
                                }
                            }
                        }                        
                    }      
                }
            }      
        }
        else if (layer[l].dimensions==3){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    for (int j_z=0;j_z<layer[l].neurons_z;j_z++){
                        if (layer[l-1].dimensions==1){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                result+=layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x];
                            }
                        }
                        else if(layer[l-1].dimensions==2){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    result+=layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y];
                                }
                            }
                        }
                        else if (layer[l-1].dimensions==3){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                        result+=layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z];
                                    }
                                }
                            }       
                        }   
                    }   
                }
            } 
        }
        layer[l].avg_sum_of_weights=vnum(result/layer[l].neurons_total);
        total=vnum(total+layer[l].avg_sum_of_weights);
    }
    double balanced_sum_of_weight=total/(layers-1);
    // step 2: balance weights between layers / layerwise rescaling according to imbalances
    for (int l=1;l<layers;l++){
        double scale_factor=vnum(balanced_sum_of_weight/layer[l].avg_sum_of_weights);
        if (layer[l].dimensions==1){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                if (layer[l-1].dimensions==1){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        layer[l].neuron1d[j_x].input_weight_1d[i_x]=vnum(layer[l].neuron1d[j_x].input_weight_1d[i_x]*scale_factor);
                    }
                }
                else if(layer[l-1].dimensions==2){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y]=vnum(layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y]*scale_factor);
                        }
                    }
                }
                else if (layer[l-1].dimensions==3){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z]=vnum(layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z]*scale_factor);
                            }
                        }
                    }
                }
            }
        }
        else if (layer[l].dimensions==2){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    if (layer[l-1].dimensions==1){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x]=vnum(layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x]*scale_factor);
                        }
                    }
                    else if(layer[l-1].dimensions==2){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y]=vnum(layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y]*scale_factor);
                            }
                        }
                    }
                    else if (layer[l-1].dimensions==3){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                    layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z]=vnum(layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z]*scale_factor);
                                }
                            }
                        }                        
                    }      
                }
            }      
        }
        else if (layer[l].dimensions==3){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    for (int j_z=0;j_z<layer[l].neurons_z;j_y++){
                        if (layer[l-1].dimensions==1){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x]=vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x]*scale_factor);
                            }
                        }
                        else if(layer[l-1].dimensions==2){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y]=vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y]*scale_factor);
                                }
                            }
                        }
                        else if (layer[l-1].dimensions==3){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z]=vnum(layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z]*scale_factor);
                                    }
                                }
                            }       
                        }   
                    }   
                }
            } 
        }
    }    
}

// reset weights
void NeuralNet::reset_weights(){
    // using He initialization with normal distribution
    for (int l=1;l<layers;l++){
        int fan_in=layer[l-1].neurons_total;
        int fan_out=1;
        if (l!=layers-1){ // =if not output layer
            fan_out=layer[l+1].neurons_total;
        }
        double sigma=sqrt(2.0/(fan_in+fan_out));
        if (layer[l].dimensions==1){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                if (layer[l-1].dimensions==1){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        layer[l].neuron1d[j_x].input_weight_1d[i_x]=rand_norm(0,sigma);
                    }
                }
                else if(layer[l-1].dimensions==2){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            layer[l].neuron1d[j_x].input_weight_2d[i_x][i_y]=rand_norm(0,sigma);
                        }
                    }
                }
                else if (layer[l-1].dimensions==3){
                    for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                        for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                            for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                layer[l].neuron1d[j_x].input_weight_3d[i_x][i_y][i_z]=rand_norm(0,sigma);
                            }
                        }
                    }
                }
            }
        }
        else if (layer[l].dimensions==2){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    if (layer[l-1].dimensions==1){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            layer[l].neuron2d[j_x][j_y].input_weight_1d[i_x]=rand_norm(0,sigma);
                        }
                    }
                    else if(layer[l-1].dimensions==2){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                layer[l].neuron2d[j_x][j_y].input_weight_2d[i_x][i_y]=rand_norm(0,sigma);
                            }
                        }
                    }
                    else if (layer[l-1].dimensions==3){
                        for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                            for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                    layer[l].neuron2d[j_x][j_y].input_weight_3d[i_x][i_y][i_z]=rand_norm(0,sigma);
                                }
                            }
                        }                        
                    }      
                }
            }      
        }
        else if (layer[l].dimensions==3){
            for (int j_x=0;j_x<layer[l].neurons_x;j_x++){
                for (int j_y=0;j_y<layer[l].neurons_y;j_y++){
                    for (int j_z=0;j_z<layer[l].neurons_z;j_z++){
                        if (layer[l-1].dimensions==1){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                layer[l].neuron3d[j_x][j_y][j_z].input_weight_1d[i_x]=rand_norm(0,sigma);
                            }
                        }
                        else if(layer[l-1].dimensions==2){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    layer[l].neuron3d[j_x][j_y][j_z].input_weight_2d[i_x][i_y]=rand_norm(0,sigma);
                                }
                            }
                        }
                        else if (layer[l-1].dimensions==3){
                            for (int i_x=0;i_x<layer[l-1].neurons_x;i_x++){
                                for (int i_y=0;i_y<layer[l-1].neurons_y;i_y++){
                                    for (int i_z=0;i_z<layer[l-1].neurons_z;i_z++){
                                        layer[l].neuron3d[j_x][j_y][j_z].input_weight_3d[i_x][i_y][i_z]=rand_norm(0,sigma);
                                    }
                                }
                            }       
                        }   
                    }   
                }
            } 
        }
    }    
}

// save network data into file
void NeuralNet::save(string filename) {

}


/*
To-Do:
- adaptation for different neuron types (lstm,gru, etc)

Idea for image recognition algorithm (requires 2d network for autoencoder inputs in x+y dimension):
- input filter for images: scan small squares with 2d autoencoder into a single value (repeat for all 3 colors), then for each area look for the value that was
 associated with the lowest error -> take this as the essential feature for this region of the picture (pooling)
 -> next layer will have reduced number of neurons (depending to the pooling raster);
  each layer will have it's own autoencoder in order to regognize patterns that are typical for this layer
  */