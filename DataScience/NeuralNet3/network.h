#pragma once
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/neuron.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/input.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/output.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet3/layer.h>
#include <vector>
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include </home/christian/Documents/own_code/c++/DataScience/distributions/random_distributions.h>
#include </home/christian/Documents/own_code/c++/DataScience/activation_functions.h>
using namespace std;

class Network{
    private:
        vector<Layer> layer;
        OPTIMIZATION_METHOD method=Vanilla;
        double lr=0.001;
        double lr_momentum=0.0;
        double lr_decay=0.0;
        double opt_beta1=0.9;
        double opt_beta2=0.99;        
        int backprop_iterations=0;
        int layers=0;
        void make_connections(int layer_index, int neuron_index);
        void validate(double& expression);
    protected:
    public:
        void add_inputlayer(int inputs, SCALING scaling_method=normalized);
        void add_inputs(int amount,SCALING scaling_method=normalized);
        void add_input(SCALING scaling_method=normalized){add_inputs(1,scaling_method);}
        void add_outputlayer(int outputs,  int lower_connections, int level_connections=0, SCALING scaling_method=normalized);
        void add_outputs(int amount,int lower_connections, int level_connections=0,SCALING scaling_method=normalized);
        void add_output(int lower_connections, int level_connections=0,SCALING scaling_method=normalized){add_outputs(1,level_connections,scaling_method);}
        void add_hiddenlayer(int neurons, int lower_connections, int level_connections=0, ACTIVATION_FUNC act_func=f_LReLU);
        void add_neurons(int layer_index, int amount, int lower_connections, int level_connections=0, ACTIVATION_FUNC act_func=f_LReLU);
        void widen(int amount, int lower_connections, int level_connections=0, ACTIVATION_FUNC act_func=f_LReLU, SCALING scaling_method=normalized);
        void widen_hidden(int amount, int lower_connections, int level_connections=0, ACTIVATION_FUNC act_func=f_LReLU);
        void set_learning_rate(double value){lr=fmin(fmax(0,value),1.0);}
        void set_learning_rate_decay(double value){lr_decay=fmin(fmax(0,value),1.0);}
        void set_learning_momentum(double value){lr_momentum=fmin(fmax(0,value),1.0);}   
        void set_optimizer(OPTIMIZATION_METHOD method){this->method=method;}     
        void set_input(int index, double value);
        double get_input(int index){return layer[0].input[index].get_input();}
        double get_output(int index){return layer[layers-1].output[index].get_output();}
        double get_loss_avg(int index){return layer[layers-1].output[index].get_loss_avg();}
        void set_label(int index, double value);
        void run();
        void backprop(bool remap=true);
        void autoencode();
        // constructor
        Network(){};
        // destructor
        ~Network(){};
};

// private helper function for number validation
void Network::validate(double& expression){
    // NaN protection
    if (expression!=expression){
        expression=__DBL_MIN__;
    }
    // inf protection
    if (isinf(expression)){
        if (expression>__DBL_MAX__){
            expression=__DBL_MAX__;
        }
        else if (expression<-__DBL_MAX__){
            expression=-__DBL_MAX__;
        }
    }
}

// add input layer at position 0 of the layer stack
void Network::add_inputlayer(int inputs, SCALING scaling_method){
    layer.insert(layer.begin(),Layer());
    layer[0].index=0;
    layers++;
    // shift indices for higher layers
    for (int l=1;l<layers;l++){
        layer[l].index++;
        for (int j=0;j<layer[l].neurons;j++){
            for (int i=0;i<layer[l].neuron[j].incoming.size();i++){
                layer[l].neuron[j].incoming[i].from_layer++;
            }
            for (int k=0;k<layer[l].neuron[j].outgoing.size();k++){
                layer[l].neuron[j].outgoing[k].to_layer++;
            }
        }
    }
    // populate layer with inputs and associated neurons
    layer[0].add_inputs(inputs, scaling_method);
    // attach subsequent layer
    if (layers>1){
        for (int j=0;j<layer[1].neurons;j++){
            // erase previous connections
            layer[1].neuron[j].incoming.clear();
            for (int k=0;k<layer[1].neuron[j].outgoing.size();k++){
                while (layer[1].neuron[j].outgoing[k].to_layer==1){
                    layer[1].neuron[j].outgoing.erase(layer[1].neuron[j].outgoing.begin()+k);
                }
            }
            // make new connections
            make_connections(1,j);
        }
    }
}

void Network::add_inputs(int amount,SCALING scaling_method){
    // create (/inset) a new input layer if no layers exist
    // or if lowest layer is a hidden layer
    if (layers==0 || layer[0].inputs==0){
        add_inputlayer(amount,scaling_method);
        return;
    }
    // else: add inputs to existing input layer
    else {
        for (int n=0;n<amount;n++){
            layer[0].add_input(scaling_method);
        }
    }
}

// add new output layer to the top of the layer stack
void Network::add_outputlayer(int outputs,  int lower_connections, int level_connections, SCALING scaling_method){
    layer.push_back(Layer());
    layers++;
    int this_layer=layers-1;
    layer[this_layer].index=this_layer;
    layer[this_layer].add_outputs(outputs,scaling_method);
    // make connections
    for (int j=0;j<outputs;j++){
        layer[this_layer].neuron[j].lower_connections=lower_connections;
        layer[this_layer].neuron[j].level_connections=level_connections;
        make_connections(this_layer,j);
    }
}

// add more outputs to top level layer
void Network::add_outputs(int amount,int lower_connections, int level_connections,SCALING scaling_method){
    // create new output layer if no layers exist
    if (layers==0){
        add_outputlayer(amount,lower_connections, level_connections, scaling_method);
        return;
    }
    // else: create new output layer if there's nothing but an input layer
    else if (layers==1 && layer[0].inputs>0){
        add_outputlayer(amount, lower_connections, level_connections, scaling_method);
        return;
    }
    // else: add outputs to preexisting top layer if this is an output or hidden layer
    else{
        int this_layer=layers-1;
        for (int n=0;n<amount;n++){
            layer[this_layer].add_output(scaling_method);
            layer[this_layer].neuron[layer[this_layer].outputs-1].lower_connections=lower_connections;
            layer[this_layer].neuron[layer[this_layer].outputs-1].level_connections=level_connections;
            make_connections(this_layer,layer[this_layer].neurons-1);
        }
    }
}

// add hidden layer
void Network::add_hiddenlayer(int neurons, int lower_connections, int level_connections, ACTIVATION_FUNC act_func){
    int this_layer;
    // add to the top of the layer stack if there's no output layer yet
    if (layers==0 || layer[layers-1].outputs==0){
        layer.push_back(Layer());
        layers++;
        this_layer=layers-1;
        layer[this_layer].index=this_layer;
        // populate with neurons
        layer[this_layer].add_hidden(neurons,act_func);
        // make connections
        for (int j=0;j<neurons;j++){
            layer[this_layer].neuron[j].lower_connections=lower_connections;
            layer[this_layer].neuron[j].level_connections=level_connections;
            make_connections(this_layer,j);
        }
        return;
    }
    // else: insert below output layer
    else {
        layer.insert(layer.end()-1,Layer());
        layers++;
        this_layer=layers-2;
        layer[this_layer].index=this_layer;
        // populate new layer with neurons
        layer[this_layer].add_hidden(neurons,act_func);
        // erase old connections from preceding layer
        if (this_layer-1>=0){
            for (int i=0;i<layer[this_layer-1].neurons;i++){
                for (int k=0;k<layer[this_layer-1].neuron[i].outgoing.size();k++){
                    while (layer[this_layer-1].neuron[i].outgoing[k].to_layer==this_layer){
                        layer[this_layer-1].neuron[i].outgoing.erase(layer[this_layer-1].neuron[i].outgoing.begin()+k);
                    }
                }
            }
        }
        // make new connections
        for (int j=0;j<neurons;j++){
            layer[this_layer].neuron[j].lower_connections=lower_connections;
            layer[this_layer].neuron[j].level_connections=level_connections;
            make_connections(this_layer,j);
        }
        // reattach output layer
        for (int k=0;k<layer[layers-1].neurons;k++){
            layer[layers-1].neuron[k].incoming.clear();
            layer[layers-1].neuron[k].outgoing.clear();
        }
        for (int k=0;k<layer[layers-1].neurons;k++){
            make_connections(layers-1,k);
        }
    }
}

// make weight conections
// lower_connections: incoming connections (per neuron!) from preceding layer
// level_connections: incoming connections (per neuron!) from same layer
void Network::make_connections(int layer_index, int neuron_index){
    int this_layer=layer_index;
    int lower_connections=layer[this_layer].neuron[neuron_index].lower_connections;
    int level_connections=layer[this_layer].neuron[neuron_index].level_connections;
    // make connections to preceding layer
    for (int i=0;i<lower_connections;i++){
        // safety exit if this is already the lowest layer
        if (this_layer==0){break;}
        // get index of a random neuron from preceding layer
        int from_index=floor(rand_uni()*layer[this_layer-1].neurons);
        // assign an incoming connection from this random neuron (initialize=true)
        int weight_index=layer[this_layer].neuron[neuron_index].add_incoming(this_layer-1,from_index,true);
        // add a corresponding outgoing connection at the random neuron to current neuron
        layer[this_layer-1].neuron[from_index].add_outgoing(this_layer,neuron_index,weight_index);
    }
    // make connections within the same layer
    for (int j=0;j<level_connections;j++){
        // safety exit if layer has no more than 1 neuron
        if (layer[this_layer].neurons<=1){break;}
        // get index of a random neuron from current layer
        int from_index;
        do {
            from_index=floor(rand_uni()*layer[this_layer].neurons);
        }
        // avoid connection of neuron to itself + avoid invalid index
        while (from_index==neuron_index || from_index>=layer[this_layer].neurons);
        // assign an incoming connection from the randomly selected neuron (initialize=true)
        int weight_index=layer[this_layer].neuron[neuron_index].add_incoming(this_layer,from_index,true);
        // add a corresponding outgoing connection at the random neuron to current neuron
        layer[this_layer].neuron[from_index].add_outgoing(this_layer,neuron_index,weight_index);
    }        
}

// add hidden neurons
void Network::add_neurons(int layer_index, int amount, int lower_connections, int level_connections, ACTIVATION_FUNC act_func){
    // safety exit if layer index is invalid
    if (layer_index>=layers-1 || layer_index<0){return;}
    // add neurons
    int previous_neurons=layer[layer_index].neurons;
    layer[layer_index].add_hidden(amount,act_func);
    for (int i=previous_neurons;i<layer[layer_index].neurons;i++){
        layer[layer_index].neuron[i].lower_connections=lower_connections;
        layer[layer_index].neuron[i].level_connections=level_connections;
        make_connections(layer_index,i);
    }
}

// widen all layers (including inputs and outputs) by specified amount of extra neurons
void Network::widen(int amount, int lower_connections, int level_connections, ACTIVATION_FUNC act_func, SCALING scaling_method){
    for (int l=0;l<layers;l++){
        // if this is an input layer
        if (layer[l].inputs>0){
            layer[l].add_inputs(amount,scaling_method);
        }
        // if this is an output layer
        else if (layer[l].outputs>0){
            layer[l].add_outputs(amount,scaling_method);
        }
        // default: hidden layer
        else {
            layer[l].add_hidden(amount,act_func);
        }
    }
}

// widen all hidden(!) layers (without inputs and outputs) by specified amount of extra neurons
void Network::widen_hidden(int amount, int lower_connections, int level_connections, ACTIVATION_FUNC act_func){
    for (int l=0;l<layers;l++){
        // skip if this is an input or output layer
        if (layer[l].inputs>0 || layer[l].outputs>0){
            continue;
        }
        // default: hidden layer
        else {
            layer[l].add_hidden(amount,act_func);
        }
    }
}

// set individual network input
void Network::set_input(int index, double value){
    layer[0].input[index].set_input(value);
}

// set individual output label
void Network::set_label(int index, double value){
    layer[layers-1].output[index].set_label(value);
}

// run forward pass
void Network::run(){
    // write scaled inputs to associated neurons
    for (int j=0;j<layer[0].inputs;j++){
        int target=layer[0].input[j].to_index;
        layer[0].neuron[target].h=layer[0].input[j].get_scaled_input();
    }
    // cycle through all layers
    for (int l=0;l<layers;l++){
        for (int j=0;j<layer[l].neurons;j++){
            // reset x
            layer[l].neuron[j].x=0;
            // add sum of weighted inputs
            int n=layer[l].neuron[j].incoming.size();
            for (int i=0;i<n;i++){
                if (layer[l].neuron[j].incoming[i].from_layer==l-1){
                    layer[l].neuron[j].x+=
                    layer[l].neuron[j].incoming[i].weight*
                    layer[l-1].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                }
            }
            // add bias
            layer[l].neuron[j].x+=layer[l].neuron[j].bias_weight;
            // add weighted recurrent values from last iteration
            layer[l].neuron[j].x+=layer[l].neuron[j].m1*layer[l].neuron[j].m1_weight;
            layer[l].neuron[j].x+=layer[l].neuron[j].m2*layer[l].neuron[j].m2_weight;
            layer[l].neuron[j].x+=layer[l].neuron[j].m3*layer[l].neuron[j].m3_weight;
            layer[l].neuron[j].x+=layer[l].neuron[j].m4*layer[l].neuron[j].m4_weight;
            validate(layer[l].neuron[j].x);
            // activate
            layer[l].neuron[j].h=activate(layer[l].neuron[j].x,layer[l].neuron[j].act_func);
        }
        // get weighted h values from same layer
        for (int j=0;j<layer[l].neurons;j++){
            layer[l].neuron[j].y=0;
            int n=layer[l].neuron[j].incoming.size();
            for (int i=0;i<n;i++){
                if (layer[l].neuron[j].incoming[i].from_layer==l){
                    layer[l].neuron[j].y+=
                    layer[l].neuron[j].incoming[i].weight*
                    layer[l].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                }
            }
        }
        // update x,
        // update activation,
        // update recurrent values for next iteration
        for (int j=0;j<layer[l].neurons;j++){
            layer[l].neuron[j].x+=layer[l].neuron[j].y;
            validate(layer[l].neuron[j].x);
            layer[l].neuron[j].h=activate(layer[l].neuron[j].x, layer[l].neuron[j].act_func);
            layer[l].neuron[j].m1=layer[l].neuron[j].h;
            layer[l].neuron[j].m2=layer[l].neuron[j].m2*0.9+layer[l].neuron[j].h*0.1;
            layer[l].neuron[j].m3=layer[l].neuron[j].m3*0.99+layer[l].neuron[j].h*0.01;
            layer[l].neuron[j].m4=layer[l].neuron[j].m4*0.999+layer[l].neuron[j].h*0.001;
        }       
    }
    // write results to outputs
    for (int j=0;j<layer[layers-1].outputs;j++){
        layer[layers-1].output[j].set_result(layer[layers-1].neuron[layer[layers-1].output[j].from_index].h);
    }
}

// run backpropagation
void Network::backprop(bool remap){
    backprop_iterations++;
    // read output errors
    for (int j=0;j<layer[layers-1].outputs;j++){
        layer[layers-1].neuron[layer[layers-1].output[j].from_index].gradient=layer[layers-1].output[j].get_gradient();
    }
    // push gradients through lower layers (map hidden error gradients)
    for (int l=layers-2;l>=1;l--){
        // get weighted errors from subsequent layer
        for (int j=0;j<layer[l].neurons;j++){
            // reset gradient
            layer[l].neuron[j].gradient=0;
            // get SUM_k[err_k*w_jk]
            for (int k=0;k<layer[l].neuron[j].outgoing.size();k++){
                if (layer[l].neuron[j].outgoing[k].to_layer==l+1){
                    int neuron_k=layer[l].neuron[j].outgoing[k].to_neuron;
                    int weight_jk_index=layer[l].neuron[j].outgoing[k].to_weight;
                    layer[l].neuron[j].gradient+=layer[l+1].neuron[neuron_k].gradient*layer[l+1].neuron[neuron_k].incoming[weight_jk_index].weight;
                }
            }
        }
        // get weighted errors from same layer (temporarily store as 'z')
        for (int j=0;j<layer[l].neurons;j++){
            layer[l].neuron[j].z=0;
            // get SUM_k[err_k*w_jk]
            for (int k=0;k<layer[l].neuron[j].outgoing.size();k++){
                if (layer[l].neuron[j].outgoing[k].to_layer==l){
                    int neuron_k=layer[l].neuron[j].outgoing[k].to_neuron;
                    int weight_jk_index=layer[l].neuron[j].outgoing[k].to_weight;
                    layer[l].neuron[j].z+=layer[l].neuron[neuron_k].gradient*layer[l].neuron[neuron_k].incoming[weight_jk_index].weight;
                }
            }            
        }
        // correct gradients by adding value of 'z'
        for (int j=0;j<layer[l].neurons;j++){
            layer[l].neuron[j].gradient+=layer[l].neuron[j].z;
        }
    }
    // update weights; general rule: delta w_ij=lr*error*act'(net_inp_j)*out_i   
    for (int l=layers-1;l>=1;l--){
        for (int j=0;j<layer[l].neurons;j++){
            int inputs=layer[l].neuron[j].incoming.size();
            for (int i=0;i<inputs;i++){

                if (method==Vanilla){
                    // get delta
                    layer[l].neuron[j].incoming[i].weight_delta = lr_momentum*layer[l].neuron[j].incoming[i].weight_delta + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x, layer[l].neuron[j].act_func) * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // update
                    layer[l].neuron[j].incoming[i].weight += layer[l].neuron[j].incoming[i].weight_delta;
                }
                else if (method==Nesterov){
                    // lookahead step
                    double lookahead = (lr_momentum*layer[l].neuron[j].incoming[i].weight_delta) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // momentum step
                    layer[l].neuron[j].incoming[i].weight_delta = (lr_momentum*lookahead) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // update step
                    layer[l].neuron[j].incoming[i].weight += layer[l].neuron[j].incoming[i].weight_delta;
                }
                else if (method==RMSprop){
                    // opt_v update
                    layer[l].neuron[j].incoming[i].opt_v = lr_momentum*layer[l].neuron[j].incoming[i].opt_v + (1-lr_momentum)*pow(deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func),2) * layer[l].neuron[j].gradient;
                    // get delta
                    layer[l].neuron[j].incoming[i].weight_delta = (lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l].neuron[j].incoming[i].opt_v+1e-8)+__DBL_MIN__) * pow(layer[l].neuron[j].h,2) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // update
                    layer[l].neuron[j].incoming[i].weight += layer[l].neuron[j].incoming[i].weight_delta;
                }
                else if (method==ADADELTA){
                    // opt_v update
                    layer[l].neuron[j].incoming[i].opt_v = opt_beta1 * layer[l].neuron[j].incoming[i].opt_v + (1-opt_beta1) * pow(deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h,2);
                    // opt_w update
                    layer[l].neuron[j].incoming[i].opt_w = (opt_beta1 * pow(layer[l].neuron[j].incoming[i].opt_w,2)) + (1-opt_beta1)*pow(layer[l].neuron[j].incoming[i].weight_delta,2);
                    // get delta
                    layer[l].neuron[j].incoming[i].weight_delta = sqrt(layer[l].neuron[j].incoming[i].opt_w+1e-8)/(sqrt(layer[l].neuron[j].incoming[i].opt_v+1e-8)+__DBL_MIN__) * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // update
                    layer[l].neuron[j].incoming[i].weight+=layer[l].neuron[j].incoming[i].weight_delta;
                }
                else if (method==ADAM){ // =ADAM without minibatch
                    // opt_v update
                    layer[l].neuron[j].incoming[i].opt_v = opt_beta1 * layer[l].neuron[j].incoming[i].opt_v + (1-opt_beta1) * deactivate(layer[l].neuron[j].x, layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // opt_w update
                    layer[l].neuron[j].incoming[i].opt_w = opt_beta2 * layer[l].neuron[j].incoming[i].opt_w * pow(deactivate(layer[l].neuron[j].x, layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h,2);
                    // get delta
                    double v_t = layer[l].neuron[j].incoming[i].opt_v/(1-opt_beta1);
                    double w_t = layer[l].neuron[j].incoming[i].opt_w/(1-opt_beta2);
                    layer[l].neuron[j].incoming[i].weight_delta = (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__);
                    // update
                    layer[l].neuron[j].incoming[i].weight+=layer[l].neuron[j].incoming[i].weight_delta;
                }
                else if (method==AdaGrad){
                    // opt_v update
                    layer[l].neuron[j].incoming[i].opt_v = layer[l].neuron[j].incoming[i].opt_v + pow(deactivate(layer[l].neuron[j].x, layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h,2);
                    // get delta
                    layer[l].neuron[j].incoming[i].weight_delta = ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l].neuron[j].incoming[i].opt_v +1e-8)) * deactivate(layer[l].neuron[j].x, layer[l].neuron[j].act_func) * layer[l].neuron[j].gradient * layer[layer[l].neuron[j].incoming[i].from_layer].neuron[layer[l].neuron[j].incoming[i].from_neuron].h;
                    // update
                    layer[l].neuron[j].incoming[i].weight+=layer[l].neuron[j].incoming[i].weight_delta;
                }     
                // validate
                validate(layer[l].neuron[j].incoming[i].weight);
            }
            // update bias weights (Vanilla)
            layer[l].neuron[j].delta_b = (lr_momentum*layer[l].neuron[j].delta_b) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func);
            layer[l].neuron[j].bias_weight += layer[l].neuron[j].delta_b;
            validate(layer[l].neuron[j].bias_weight);

            // update recurrent weights (Vanilla)
            layer[l].neuron[j].delta_m1 = (lr_momentum*layer[l].neuron[j].delta_m1) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].m1;
            layer[l].neuron[j].delta_m2 = (lr_momentum*layer[l].neuron[j].delta_m2) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].m2;
            layer[l].neuron[j].delta_m3 = (lr_momentum*layer[l].neuron[j].delta_m3) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].m3;
            layer[l].neuron[j].delta_m4 = (lr_momentum*layer[l].neuron[j].delta_m4) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l].neuron[j].gradient * deactivate(layer[l].neuron[j].x,layer[l].neuron[j].act_func) * layer[l].neuron[j].m4;
            layer[l].neuron[j].m1_weight += layer[l].neuron[j].delta_m1; validate(layer[l].neuron[j].m1_weight);
            layer[l].neuron[j].m2_weight += layer[l].neuron[j].delta_m2; validate(layer[l].neuron[j].m2_weight);
            layer[l].neuron[j].m3_weight += layer[l].neuron[j].delta_m3; validate(layer[l].neuron[j].m3_weight);
            layer[l].neuron[j].m4_weight += layer[l].neuron[j].delta_m4; validate(layer[l].neuron[j].m4_weight);
        }
    }
    // remap least significant weights
    int x=10; // perform remapping every x iterations
    if (remap && backprop_iterations%x==0){
        for (int l=1;l<layers;l++){
            for (int first_neuron_j=0;first_neuron_j<layer[l].neurons;first_neuron_j++){
                if (layer[l].neuron[first_neuron_j].lower_connections>0){
                    // find smallest weight from preceding layer
                    double smallest_weight=__DBL_MAX__;
                    int first_weight_j=0;
                    int first_neuron_i=0;
                    for (int i=0;i<layer[l].neuron[first_neuron_j].incoming.size();i++){
                        if (layer[l].neuron[first_neuron_j].incoming[i].from_layer<l){
                            if (fabs(layer[l].neuron[first_neuron_j].incoming[i].weight)<smallest_weight){
                                smallest_weight=fabs(layer[l].neuron[first_neuron_j].incoming[i].weight);
                                first_weight_j=i;
                                first_neuron_i=layer[l].neuron[first_neuron_j].incoming[i].from_neuron;
                            }
                        }
                    }
                    // get outgoing weight index
                    int first_weight_i=0;
                    for (int k=0;k<layer[l-1].neuron[first_neuron_i].outgoing.size();k++){
                        if (layer[l-1].neuron[first_neuron_i].outgoing[k].to_layer==l){
                            if (layer[l-1].neuron[first_neuron_i].outgoing[k].to_neuron==first_neuron_j){
                                first_weight_i=k;
                                break;
                            }
                        }
                    }
                    // find smallest weight (from preceding layer) of a second neuron
                    int second_neuron_j=floor(rand_uni()*layer[l].neurons);
                    smallest_weight=__DBL_MAX__;
                    int second_weight_j=0;
                    int second_neuron_i=0;
                    for (int i=0;i<layer[l].neuron[second_neuron_j].incoming.size();i++){
                        if (layer[l].neuron[second_neuron_j].incoming[i].from_layer<l){
                            if (fabs(layer[l].neuron[second_neuron_j].incoming[i].weight)<smallest_weight){
                                smallest_weight=fabs(layer[l].neuron[second_neuron_j].incoming[i].weight);
                                second_weight_j=i;
                                second_neuron_i=layer[l].neuron[second_neuron_j].incoming[i].from_neuron;
                            }
                        }
                    }         
                    // get outgoing weight index
                    int second_weight_i=0;
                    for (int k=0;k<layer[l-1].neuron[second_neuron_i].outgoing.size();k++){
                        if (layer[l-1].neuron[second_neuron_i].outgoing[k].to_layer==l){
                            if (layer[l-1].neuron[second_neuron_i].outgoing[k].to_neuron==second_neuron_j){
                                second_weight_i=k;
                                break;
                            }
                        }
                    }                
                    // swap connections
                    layer[l-1].neuron[first_neuron_i].outgoing[first_weight_i].to_neuron=second_neuron_j;
                    layer[l-1].neuron[first_neuron_i].outgoing[first_weight_i].to_weight=second_weight_j;
                    layer[l].neuron[second_neuron_j].incoming[second_weight_j].from_neuron=first_neuron_i;

                    layer[l-1].neuron[second_neuron_i].outgoing[second_weight_i].to_neuron=first_neuron_j;
                    layer[l-1].neuron[second_neuron_i].outgoing[second_weight_i].to_weight=first_weight_j;
                    layer[l].neuron[first_neuron_j].incoming[first_weight_j].from_neuron=second_neuron_i;
                }

                // repeat steps for weight from same layer
                if (layer[l].neuron[first_neuron_j].level_connections>0){
                    // find smallest weight from same layer
                    double smallest_weight=__DBL_MAX__;
                    int first_weight_j=0;
                    int first_neuron_i=0;
                    for (int i=0;i<layer[l].neuron[first_neuron_j].incoming.size();i++){
                        if (layer[l].neuron[first_neuron_j].incoming[i].from_layer==l){
                            if (fabs(layer[l].neuron[first_neuron_j].incoming[i].weight)<smallest_weight){
                                smallest_weight=fabs(layer[l].neuron[first_neuron_j].incoming[i].weight);
                                first_weight_j=i;
                                first_neuron_i=layer[l].neuron[first_neuron_j].incoming[i].from_neuron;
                            }
                        }
                    }
                    // get outgoing weight index
                    int first_weight_i=0;
                    for (int k=0;k<layer[l].neuron[first_neuron_i].outgoing.size();k++){
                        if (layer[l].neuron[first_neuron_i].outgoing[k].to_layer==l){
                            if (layer[l].neuron[first_neuron_i].outgoing[k].to_neuron==first_neuron_j){
                                first_weight_i=k;
                                break;
                            }
                        }
                    }
                    // find smallest weight (from same layer) of a second neuron
                    int second_neuron_j=floor(rand_uni()*layer[l].neurons);
                    smallest_weight=__DBL_MAX__;
                    int second_weight_j=0;
                    int second_neuron_i=0;
                    for (int i=0;i<layer[l].neuron[second_neuron_j].incoming.size();i++){
                        if (layer[l].neuron[second_neuron_j].incoming[i].from_layer==l){
                            if (fabs(layer[l].neuron[second_neuron_j].incoming[i].weight)<smallest_weight){
                                smallest_weight=fabs(layer[l].neuron[second_neuron_j].incoming[i].weight);
                                second_weight_j=i;
                                second_neuron_i=layer[l].neuron[second_neuron_j].incoming[i].from_neuron;
                            }
                        }
                    }         
                    // get outgoing weight index
                    int second_weight_i=0;
                    for (int k=0;k<layer[l].neuron[second_neuron_i].outgoing.size();k++){
                        if (layer[l-1].neuron[second_neuron_i].outgoing[k].to_layer==l){
                            if (layer[l].neuron[second_neuron_i].outgoing[k].to_neuron==second_neuron_j){
                                second_weight_i=k;
                                break;
                            }
                        }
                    }                
                    // swap connections
                    layer[l].neuron[first_neuron_i].outgoing[first_weight_i].to_neuron=second_neuron_j;
                    layer[l].neuron[first_neuron_i].outgoing[first_weight_i].to_weight=second_weight_j;
                    layer[l].neuron[second_neuron_j].incoming[second_weight_j].from_neuron=first_neuron_i;

                    layer[l].neuron[second_neuron_i].outgoing[second_weight_i].to_neuron=first_neuron_j;
                    layer[l].neuron[second_neuron_i].outgoing[second_weight_i].to_weight=first_weight_j;
                    layer[l].neuron[first_neuron_j].incoming[first_weight_j].from_neuron=second_neuron_i;
                }
            }
        }  
    }
}

void Network::autoencode(){
    for (int n=0;n<fmin(layer[0].inputs, layer[layers-1].outputs);n++){
        layer[layers-1].output[n].set_label(layer[0].input[n].get_input());
    }
}