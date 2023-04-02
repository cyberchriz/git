// author: 'cyberchriz' (Christian Suer)
// this is a neural network library for flexible topologies

// neural network optimization methods
enum OPTIMIZATION_METHOD
  {
   Vanilla=1,           // Vanilla Stochastic Gradient Descent
   Nesterov=2,          // Nesterov Accelerated Gradient (NAG)
   RMSprop=3,           // RMSprop
   ADADELTA=4,          // ADADELTA
   ADAM=5,              // ADAM
   AdaGrad=6            // AdaGrad
  };

// neural network feature and label scaling methods
enum SCALING
  {
   none,              // no scaling
   standardized,      // standard deviation method (µ=0, sigma=1)
   normalized,        // minmax, range 0 to 1
   maxabs             // minmax, range -1 to +1
  };

#pragma once
#include "../../DataScience/array.h"
#include "../../weight_init.h"
#include "layer.h"
#include "../../activation_functions.h"
#include <cmath>
#define MAX_LAYERS 50

class Network{
    private:
        int layers=0;
        Layer* layer[MAX_LAYERS];
        void load(string filename);
        int backprop_iterations=0;
        SCALING scaling_method=none;
        double lr=0.001;
        double lr_momentum=0.0;
        double lr_decay=0.0;
        double opt_beta1=0.9;
        double opt_beta2=0.99;
        bool training_mode=true;
        bool gradient_clipping=false;
        double gradient_clipping_threshold=0.999;
        string filename;      
    protected:
    public:
        int add_layer(int* _layer_shape, OPTIMIZATION_METHOD _opt_method=ADADELTA, ACTIVATION_FUNC _activation=f_LReLU, bool _self_attention=false, bool _recurrent=false);
        void initialize_weights(); // manually reset weights (note: weights are automatically set whenever a layer is added)
        void feedforward();
        void backpropagate();
        void set_input(int index, double value); // set single input via 1d index
        void set_input(int* nd_index,double value); // set single input via n-dimensional index
        void set_inputs(Array* nd_array); // set all inputs at once via n-dimensional data array
        double get_input(int index, int l=0){return layer[l]->neuron[index]->x;}
        double get_output(int index); // set single output via 1d index
        double get_output(int* nd_index); // set single input via n-dimensional index
        Array* get_outputs(); // get all outputs at once as n-dimensional data array        
        Array* get_hidden(int layer_index); // get 'h' values as array for hidden layer (e.g. for autoencoder bottleneck)
        double get_hidden(int index,int layer_index); // get a single 'h' from a hidden layer via 1d index
        double get_hidden(int* nd_index, int layer_index); // get a single 'h' value from a hidden layer via n-dimensional index
        void set_label(int index, double value); // set a single label value via 1d output neuron index
        void set_label(int* nd_index, double value); // set a single label value via n-dimensional output neuron index
        void set_labels(Array* nd_array); // set all labels at once via n-dimensional label data array
        void autoencode(); // enable autoencoding by copying the inputs to the labels
        void save(string filename="network_backup.dat");
        bool network_exists(){return layers!=0;}
        double get_loss_avg();
        void set_training_mode(bool active=true){training_mode=active;}
        void set_learning_rate(double value){lr=fmin(fmax(0,value),1.0);}
        void set_learning_rate_decay(double value){lr_decay=fmin(fmax(0,value),1.0);}
        void set_learning_momentum(double value){lr_momentum=fmin(fmax(0,value),1.0);}
        void set_scaling_method(SCALING method=normalized){scaling_method=method;}
        void set_gradient_clipping(bool active,double threshold=0.999){gradient_clipping=active;gradient_clipping_threshold=threshold;}
        double get_avg_h();
        double get_avg_output();     
        // constructor
        Network(string filename="");
        // destructor
        ~Network();
};

// constructor
Network::Network(string filename){
    this->filename=filename;
    //if (filename!=""){load(filename);}
}

// destructor
Network::~Network(){
    for (int i=0;i<layers;i++){
        delete layer[i];
    }
}

// add new network layer
// return value: current number of total layers
int Network::add_layer(int* _layer_shape, OPTIMIZATION_METHOD _opt_method, ACTIVATION_FUNC _activation, bool _self_attention, bool _recurrent){
    int* input_shape;
    if (layers==0){
        input_shape = new int[1];
        input_shape[0]=0;
    }
    else{
        input_shape = layer[layers-1]->get_layer_shape();
    }
    layer[layers] = new Layer(_layer_shape,input_shape,_opt_method,_activation,_self_attention,_recurrent);
    layers++;
    Network::initialize_weights();
    return layers;
}

// initialize weights:
// automatically sets appropriate method for a given activation function
void Network::initialize_weights(){
    for (int l=1;l<layers;l++){
        int fan_in=1;
        int fan_out=1;
        if (l>1){
            fan_in=layer[l-1]->get_neurons(); // input neurons
            fan_in+=1; // bias weight from preceding layer
            if (layer[l]->is_recurrent()){fan_in+=4;} // recurrent weights M1-M4 from current layer
            if (layer[l]->has_selfattention()){fan_in+=layer[l]->get_neurons();} // self-attention weights from current layer
            if (l<layers-1){ // =if not last layer
                fan_out=layer[l+1]->get_neurons();
            }
        }
        ACTIVATION_FUNC act_func = layer[l]->get_activation();
        switch (act_func){
            case f_ReLU:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_He_ReLU(fan_in));
                    }
                    layer[l]->neuron[j]->bias_weight=f_He_ReLU(fan_in);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m2_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m3_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m4_weight=f_He_ReLU(fan_in);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_He_ReLU(fan_in));
                        }
                    }                    
                }
                break;
            case f_LReLU:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_He_ReLU(fan_in));
                    }
                    layer[l]->neuron[j]->bias_weight=f_He_ReLU(fan_in);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m2_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m3_weight=f_He_ReLU(fan_in);
                        layer[l]->neuron[j]->m4_weight=f_He_ReLU(fan_in);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_He_ReLU(fan_in));
                        }
                    }                    
                }
                break;                
            case f_tanh:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_Xavier_uniform(fan_in,fan_out));
                    }
                    layer[l]->neuron[j]->bias_weight=f_Xavier_uniform(fan_in,fan_out);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_Xavier_uniform(fan_in,fan_out);
                        layer[l]->neuron[j]->m2_weight=f_Xavier_uniform(fan_in,fan_out);
                        layer[l]->neuron[j]->m3_weight=f_Xavier_uniform(fan_in,fan_out);
                        layer[l]->neuron[j]->m4_weight=f_Xavier_uniform(fan_in,fan_out);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_Xavier_uniform(fan_in,fan_out));
                        }
                    }                    
                }            
                break;
            case f_sigmoid:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_Xavier_sigmoid(fan_in,fan_out));
                    }
                    layer[l]->neuron[j]->bias_weight=f_Xavier_sigmoid(fan_in,fan_out);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_Xavier_sigmoid(fan_in,fan_out);
                        layer[l]->neuron[j]->m2_weight=f_Xavier_sigmoid(fan_in,fan_out);
                        layer[l]->neuron[j]->m3_weight=f_Xavier_sigmoid(fan_in,fan_out);
                        layer[l]->neuron[j]->m4_weight=f_Xavier_sigmoid(fan_in,fan_out);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_Xavier_sigmoid(fan_in,fan_out));
                        }
                    }                    
                }            
                break;
            case f_ELU:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_He_ELU(fan_in));
                    }
                    layer[l]->neuron[j]->bias_weight=f_He_ELU(fan_in);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_He_ELU(fan_in);
                        layer[l]->neuron[j]->m2_weight=f_He_ELU(fan_in);
                        layer[l]->neuron[j]->m3_weight=f_He_ELU(fan_in);
                        layer[l]->neuron[j]->m4_weight=f_He_ELU(fan_in);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_He_ELU(fan_in));
                        }
                    }                    
                }            
                break;
            default:
                for (int j=0;j<layer[l]->get_neurons();j++){
                    for (int i=0;i<layer[l-1]->get_neurons();i++){
                        layer[l]->neuron[j]->input_weight->set(i,f_Xavier_normal(fan_in,fan_out));
                    }
                    layer[l]->neuron[j]->bias_weight=f_Xavier_normal(fan_in,fan_out);
                    if (layer[l]->is_recurrent()){
                        layer[l]->neuron[j]->m1_weight=f_Xavier_normal(fan_in,fan_out);
                        layer[l]->neuron[j]->m2_weight=f_Xavier_normal(fan_in,fan_out);
                        layer[l]->neuron[j]->m3_weight=f_Xavier_normal(fan_in,fan_out);
                        layer[l]->neuron[j]->m4_weight=f_Xavier_normal(fan_in,fan_out);
                    }
                    if (layer[l]->has_selfattention()){
                        for (int jj=0;jj<layer[l]->get_neurons();jj++){
                            layer[l]->neuron[j]->self_weight->set(jj,f_Xavier_normal(fan_in,fan_out));
                        }
                    }                    
                }              
        }
    }
}

// forward propagation
void Network::feedforward(){
    // cycle through layers
    for (int l=1;l<layers;l++){
        int layer_neurons=layer[l]->get_neurons();
        for (int j=0;j<layer_neurons;j++){
            // get sum of weighted inputs
            layer[l]->neuron[j]->x=0;
            int input_neurons=layer[l-1]->get_neurons();
            for (int i=0;i<input_neurons;i++){
                layer[l]->neuron[j]->x+=layer[l-1]->neuron[i]->h*layer[l]->neuron[j]->input_weight->get(i);
            }
            // add weighted bias
            layer[l]->neuron[j]->x+=1.0*layer[l]->neuron[j]->bias_weight;
            // add weighted recurrent neurons
            if (layer[l]->is_recurrent()){
                layer[l]->neuron[j]->x+=layer[l]->neuron[j]->m1*layer[l]->neuron[j]->m1_weight;
                layer[l]->neuron[j]->x+=layer[l]->neuron[j]->m2*layer[l]->neuron[j]->m2_weight;
                layer[l]->neuron[j]->x+=layer[l]->neuron[j]->m3*layer[l]->neuron[j]->m3_weight;
                layer[l]->neuron[j]->x+=layer[l]->neuron[j]->m4*layer[l]->neuron[j]->m4_weight;
            }
            // activate
            layer[l]->neuron[j]->h = activate(layer[l]->neuron[j]->x,layer[l]->get_activation());
        }
        // add weighted self-attention
        if (layer[l]->has_selfattention()){
            for (int j=0;j<layer_neurons;j++){
                for (int jj=0;jj<layer_neurons;jj++){
                    layer[l]->neuron[j]->x+=layer[l]->neuron[jj]->h*layer[l]->neuron[j]->self_weight->get(jj);
                }
            }
            // activate again, now including self-attention
            for (int j=0;j<layer_neurons;j++){
                layer[l]->neuron[j]->h = activate(layer[l]->neuron[j]->x,layer[l]->get_activation());
            }
        }            
        // update recurrent values
        if (layer[l]->is_recurrent()){
            for (int j=0;j<layer_neurons;j++){
                layer[l]->neuron[j]->m1=layer[l]->neuron[j]->h;
                layer[l]->neuron[j]->m2=0.9*layer[l]->neuron[j]->m2+0.1*layer[l]->neuron[j]->h;
                layer[l]->neuron[j]->m3=0.99*layer[l]->neuron[j]->m3+0.01*layer[l]->neuron[j]->h;
                layer[l]->neuron[j]->m4=0.999*layer[l]->neuron[j]->m4+0.001*layer[l]->neuron[j]->h;
            }
        }
        // rescale outputs
        if (l==layers-1){
            for (int j=0;j<layer_neurons;j++){
                switch (scaling_method){
                    case none:
                        layer[l]->neuron[j]->output = layer[l]->neuron[j]->h;
                        break;
                    case maxabs:
                        layer[l]->neuron[j]->output = layer[l]->neuron[j]->h * layer[l]->neuron[j]->label_maxabs;
                        break;
                    case normalized:
                        layer[l]->neuron[j]->output = layer[l]->neuron[j]->h * (layer[l]->neuron[j]->label_max - layer[l]->neuron[j]->label_min) + layer[l]->neuron[j]->label_min;
                        break;
                    default:
                        // =standardized
                        layer[l]->neuron[j]->output = layer[l]->neuron[j]->h * layer[l]->neuron[j]->label_stddev + layer[l]->neuron[j]->label_rolling_average;
                        break;
                }
            }
        }        
    }
}

// backpropagation
void Network::backpropagate(){
    if (!training_mode){return;}
    backprop_iterations++;

    // (I) cycle backwards through layers
    for (int l=layers-1;l>=1;l--){
        int layer_neurons=layer[l]->get_neurons();
        // get global errors from output layer
        if (l==layers-1){
            for (int j=0;j<layer_neurons;j++){
                // derivative of the 0.5err^2 loss function: scaled_label-output
                layer[l]->neuron[j]->gradient = layer[l]->neuron[j]->scaled_label - layer[l]->neuron[j]->h;

                // gradient clipping
                if (gradient_clipping){
                    layer[l]->neuron[j]->gradient = fmin(layer[l]->neuron[j]->gradient, gradient_clipping_threshold);
                    layer[l]->neuron[j]->gradient = fmax(layer[l]->neuron[j]->gradient, -gradient_clipping_threshold);
                }

                // 0.5err^2 loss
                layer[l]->neuron[j]->loss = 0.5 * layer[l]->neuron[j]->gradient * layer[l]->neuron[j]->gradient;

                // cumulative loss (per neuron)
                layer[l]->neuron[j]->loss_sum += layer[l]->neuron[j]->loss;
            }
        }
        // get hidden errors, i.e. SUM_k[err_k*w_jk]
        else{
            for (int j=0;j<layer_neurons;j++){
                layer[l]->neuron[j]->gradient=0;
                int fan_out=layer[l+1]->get_neurons();
                for (int k=0;k<fan_out;k++){
                    layer[l]->neuron[j]->gradient+=layer[l+1]->neuron[k]->gradient*layer[l+1]->neuron[k]->input_weight->get(j);
                }
            }
        }
    }
    // (II) cycle through layers a second time in order to update the weights
    // (=with respect to how much they each influenced the (hidden) error)
    // - delta rule for output neurons: delta w_ij=lr*(label-output)*act'(net_inp_j)*out_i
    // - delta rule for hidden neurons: delta w_ij=lr*SUM_k[err_k*w_jk]*act'(net_inp_j)*out_i
    // - general rule: delta w_ij=lr*error*act'(net_inp_j)*out_i    
    for (int l=layers-1;l>=1;l--){
        int layer_neurons=layer[l]->get_neurons();
        int input_neurons=layer[l-1]->get_neurons();
        OPTIMIZATION_METHOD method=layer[l]->get_opt_method();
        for (int j=0;j<layer_neurons;j++){
            for (int i=0;i<input_neurons;i++){
                if (method==Vanilla){
                    // get delta
                    layer[l]->neuron[j]->input_weight_delta->set(i, (lr_momentum*layer[l]->neuron[j]->input_weight_delta->get(i)) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l-1]->neuron[i]->h);
                    // update
                    layer[l]->neuron[j]->input_weight->set(i,layer[l]->neuron[j]->input_weight->get(i)+layer[l]->neuron[j]->input_weight_delta->get(i));
                }
                else if (method==Nesterov){
                    // lookahead step
                    double lookahead = (lr_momentum*layer[l]->neuron[j]->input_weight_delta->get(i)) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l-1]->neuron[i]->h;
                    // momentum step
                    layer[l]->neuron[j]->input_weight_delta->set(i, (lr_momentum*lookahead) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l-1]->neuron[i]->h);
                    // update step
                    layer[l]->neuron[j]->input_weight->set(i, layer[l]->neuron[j]->input_weight->get(i) + layer[l]->neuron[j]->input_weight_delta->get(i));
                }
                else if (method==RMSprop){
                    // opt_v update
                    layer[l]->neuron[j]->opt_v->set(i, lr_momentum*layer[l]->neuron[j]->opt_v->get(i) + (1-lr_momentum)*pow(deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()),2) * layer[l]->neuron[j]->gradient);
                    // get delta
                    layer[l]->neuron[j]->input_weight_delta->set(i, ((lr/(1+lr_decay*backprop_iterations)) / (sqrt(layer[l]->neuron[j]->opt_v->get(i)+1e-8)+__DBL_MIN__)) * pow(layer[l]->neuron[j]->h,2) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h);
                    // update
                    layer[l]->neuron[j]->input_weight->set(i,layer[l]->neuron[j]->input_weight->get(i)+layer[l]->neuron[j]->input_weight_delta->get(i));
                }
                else if (method==ADADELTA){
                    // opt_v update
                    layer[l]->neuron[j]->opt_v->set(i, opt_beta1 * layer[l]->neuron[j]->opt_v->get(i) + (1-opt_beta1) * pow(deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h,2));
                    // opt_w update
                    layer[l]->neuron[j]->opt_w->set(i, (opt_beta1 * pow(layer[l]->neuron[j]->opt_w->get(i),2)) + (1-opt_beta1)*pow(layer[l]->neuron[j]->input_weight_delta->get(i),2));
                    // get delta
                    layer[l]->neuron[j]->input_weight_delta->set(i, sqrt(layer[l]->neuron[j]->opt_w->get(i)+1e-8)/(sqrt(layer[l]->neuron[j]->opt_v->get(i)+1e-8)+__DBL_MIN__) * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h);
                    // update
                    layer[l]->neuron[j]->input_weight->set(i,layer[l]->neuron[j]->input_weight->get(i)+layer[l]->neuron[j]->input_weight_delta->get(i));
                }
                else if (method==ADAM){ // =ADAM without minibatch
                    // opt_v update
                    layer[l]->neuron[j]->opt_v->set(i, opt_beta1 * layer[l]->neuron[j]->opt_v->get(i) + (1-opt_beta1) * deactivate(layer[l]->neuron[j]->x, layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h);
                    // opt_w update
                    layer[l]->neuron[j]->opt_w->set(i,opt_beta2 * layer[l]->neuron[j]->opt_w->get(i) * pow(deactivate(layer[l]->neuron[j]->x, layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h,2));
                    // get delta
                    double v_t = layer[l]->neuron[j]->opt_v->get(i)/(1-opt_beta1);
                    double w_t = layer[l]->neuron[j]->opt_w->get(i)/(1-opt_beta2);
                    layer[l]->neuron[j]->input_weight_delta->set(i, (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__));
                    // update
                    layer[l]->neuron[j]->input_weight->set(i, layer[l]->neuron[j]->input_weight->get(i)  + layer[l]->neuron[j]->input_weight_delta->get(i));
                }
                else if (method==AdaGrad){
                    // opt_v update
                    layer[l]->neuron[j]->opt_v->set(i, layer[l]->neuron[j]->opt_v->get(i) + pow(deactivate(layer[l]->neuron[j]->x, layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h,2));
                    // get delta
                    layer[l]->neuron[j]->input_weight_delta->set(i, ((lr/(1+lr_decay*backprop_iterations)) / sqrt(layer[l]->neuron[j]->opt_v->get(i) +1e-8)) * deactivate(layer[l]->neuron[j]->x, layer[l]->get_activation()) * layer[l]->neuron[j]->gradient * layer[l-1]->neuron[i]->h);
                    // update
                    layer[l]->neuron[j]->input_weight->set(i, layer[l]->neuron[j]->input_weight->get(i)  + layer[l]->neuron[j]->input_weight_delta->get(i));
                }                
            }
            // update bias weights (Vanilla)
            layer[l]->neuron[j]->delta_b = (lr_momentum*layer[l]->neuron[j]->delta_b) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation());
            layer[l]->neuron[j]->bias_weight += layer[l]->neuron[j]->delta_b;
            // update recurrent weights (Vanilla)
            if (layer[l]->is_recurrent()){
                layer[l]->neuron[j]->delta_m1 = (lr_momentum*layer[l]->neuron[j]->delta_m1) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->m1;
                layer[l]->neuron[j]->delta_m2 = (lr_momentum*layer[l]->neuron[j]->delta_m2) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->m2;
                layer[l]->neuron[j]->delta_m3 = (lr_momentum*layer[l]->neuron[j]->delta_m3) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->m3;
                layer[l]->neuron[j]->delta_m4 = (lr_momentum*layer[l]->neuron[j]->delta_m4) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[j]->m4;
                layer[l]->neuron[j]->m1_weight += layer[l]->neuron[j]->delta_m1;
                layer[l]->neuron[j]->m2_weight += layer[l]->neuron[j]->delta_m2;
                layer[l]->neuron[j]->m3_weight += layer[l]->neuron[j]->delta_m3;
                layer[l]->neuron[j]->m4_weight += layer[l]->neuron[j]->delta_m4;
            }
            // update self-attention weights (Vanilla)
            if (layer[l]->has_selfattention()){
                for (int jj=0;jj<layer_neurons;jj++){
                    // get delta
                    layer[l]->neuron[j]->self_weight_delta->set(jj, (lr_momentum*layer[l]->neuron[j]->self_weight_delta->get(jj)) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * layer[l]->neuron[j]->gradient * deactivate(layer[l]->neuron[j]->x,layer[l]->get_activation()) * layer[l]->neuron[jj]->h);
                    // update
                    layer[l]->neuron[j]->self_weight->set(jj,layer[l]->neuron[j]->self_weight->get(jj)+layer[l]->neuron[j]->self_weight_delta->get(jj));                
                }
            }
        }
    }                
}

// set a single input value via 1-dimensional index (with auto-scaling)
void Network::set_input(int index, double value){
    layer[0]->neuron[index]->x = value;
    layer[0]->neuron[index]->input_min = fmin(value,layer[0]->neuron[index]->input_min);
    layer[0]->neuron[index]->input_max = fmax(value,layer[0]->neuron[index]->input_max);
    layer[0]->neuron[index]->input_maxabs = fmax(layer[0]->neuron[index]->input_maxabs,abs(value));
    if (backprop_iterations<10){
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.5+value*0.5;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.5+layer[0]->neuron[index]->input_mdev2*0.5;     
    }
    else if (backprop_iterations<200){
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.95+value*0.05;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.95+layer[0]->neuron[index]->input_mdev2*0.05;     
    }
    else{
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.999+value*0.001;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.999+layer[0]->neuron[index]->input_mdev2*0.001;         
    } 
    layer[0]->neuron[index]->input_stddev = sqrt(layer[0]->neuron[index]->input_variance);

    switch (scaling_method){
        case none:
            layer[0]->neuron[index]->h = value;
            break;
        case maxabs:
            // -1 to 1
            layer[0]->neuron[index]->h = value / (layer[0]->neuron[index]->input_maxabs + __DBL_EPSILON__);
            break;
        case normalized:
            // 0 to 1
            layer[0]->neuron[index]->h = value - layer[0]->neuron[index]->input_min;
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_max - layer[0]->neuron[index]->input_min + __DBL_EPSILON__);
            break;
        default:
            // standardized (µ=0, sigma=1)
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h - layer[0]->neuron[index]->input_rolling_average; 
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_stddev + __DBL_EPSILON__);
            break;
    }
}

// set a single input via n-dimensional index
void Network::set_input(int* nd_index,double value){
    int index = layer[0]->neuronindex(nd_index);
    layer[0]->neuron[index]->x = value;
    layer[0]->neuron[index]->input_min = fmin(value,layer[0]->neuron[index]->input_min);
    layer[0]->neuron[index]->input_max = fmax(value,layer[0]->neuron[index]->input_max);
    layer[0]->neuron[index]->input_maxabs = fmax(layer[0]->neuron[index]->input_maxabs,abs(value));
    if (backprop_iterations<10){
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.5+value*0.5;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.5+layer[0]->neuron[index]->input_mdev2*0.5;     
    }
    else if (backprop_iterations<200){
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.95+value*0.05;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.95+layer[0]->neuron[index]->input_mdev2*0.05;     
    }
    else{
        layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.999+value*0.001;
        layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
        layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.999+layer[0]->neuron[index]->input_mdev2*0.001;         
    } 
    layer[0]->neuron[index]->input_stddev = sqrt(layer[0]->neuron[index]->input_variance);   

    switch (scaling_method){
        case none:
            layer[0]->neuron[index]->h = value;
            break;
        case maxabs:
            // -1 to 1
            layer[0]->neuron[index]->h = value / (layer[0]->neuron[index]->input_maxabs + __DBL_EPSILON__);
            break;
        case normalized:
            // 0 to 1
            layer[0]->neuron[index]->h = value - layer[0]->neuron[index]->input_min;
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_max - layer[0]->neuron[index]->input_min + __DBL_EPSILON__);
            break;
        default:
            // standardized (µ=0, sigma=1)
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h - layer[0]->neuron[index]->input_rolling_average; 
            layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_stddev + __DBL_EPSILON__);
            break;
    }
}

// set all inputs at once via n-dimensional data array, with auto-scaling
void Network::set_inputs(Array* nd_array){
    int* nd_index = new int[nd_array->get_dimensions()];
    for (int d=0;d<nd_array->get_dimensions();d++){
        for (int i=0;i<nd_array->get_size(d);i++){
            nd_index[d]=i;
            int index=nd_array->get_element(nd_index);
            double value=nd_array->get(index);
            layer[0]->neuron[index]->x = value;
            layer[0]->neuron[index]->input_min = fmin(value,layer[0]->neuron[index]->input_min);
            layer[0]->neuron[index]->input_max = fmax(value,layer[0]->neuron[index]->input_max);
            layer[0]->neuron[index]->input_maxabs = fmax(layer[0]->neuron[index]->input_maxabs,abs(value));
            if (backprop_iterations<10){
                layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.5+value*0.5;
                layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
                layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.5+layer[0]->neuron[index]->input_mdev2*0.5;     
            }
            else if (backprop_iterations<200){
                layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.95+value*0.05;
                layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
                layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.95+layer[0]->neuron[index]->input_mdev2*0.05;     
            }
            else{
                layer[0]->neuron[index]->input_rolling_average = layer[0]->neuron[index]->input_rolling_average*0.999+value*0.001;
                layer[0]->neuron[index]->input_mdev2 = pow((value-layer[0]->neuron[index]->input_rolling_average),2);
                layer[0]->neuron[index]->input_variance = layer[0]->neuron[index]->input_variance*0.999+layer[0]->neuron[index]->input_mdev2*0.001;         
            } 
            layer[0]->neuron[index]->input_stddev = sqrt(layer[0]->neuron[index]->input_variance);   

            switch (scaling_method){
                case none:
                    layer[0]->neuron[index]->h = value;
                    break;
                case maxabs:
                    // -1 to 1
                    layer[0]->neuron[index]->h = value / (layer[0]->neuron[index]->input_maxabs + __DBL_EPSILON__);
                    break;
                case normalized:
                    // 0 to 1
                    layer[0]->neuron[index]->h = value - layer[0]->neuron[index]->input_min;
                    layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_max - layer[0]->neuron[index]->input_min + __DBL_EPSILON__);
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[0]->neuron[index]->h = layer[0]->neuron[index]->h - layer[0]->neuron[index]->input_rolling_average; 
                    layer[0]->neuron[index]->h = layer[0]->neuron[index]->h / (layer[0]->neuron[index]->input_stddev + __DBL_EPSILON__);
                    break;
            }
        }
    }
}

// get single output via 1d index
double Network::get_output(int index){
    return layer[layers-1]->neuron[index]->output;
}

// get single input via n-dimensional index
double Network::get_output(int* nd_index){
    int index = layer[layers-1]->neuronindex(nd_index);
    return layer[layers-1]->neuron[index]->output;
}

// get all outputs at once as n-dimensional data array
Array* Network::get_outputs(){
    int* shape=layer[layers-1]->get_layer_shape();
    Array* result=new Array(shape);
    int *nd_index=new int[result->get_dimensions()];
    for (int d=0;d<result->get_dimensions();d++){
        for (int i=0;i<result->get_size(d);i++){
            nd_index[d]=i;
            result->set(nd_index,layer[layers-1]->neuron[layer[layers-1]->neuronindex(nd_index)]->output);
        }
    }
    return result;
}
   
// get a single 'h' from a hidden layer via 1d index (e.g. for autoencoder bottleneck)      
double Network::get_hidden(int index,int layer_index){
    return layer[layer_index]->neuron[index]->h;
}

// get a single 'h' value from a hidden layer via n-dimensional index
// (e.g. for autoencoder bottleneck)
double Network::get_hidden(int* nd_index, int layer_index){
    int index = layer[layer_index]->neuronindex(nd_index);
    return layer[layer_index]->neuron[index]->h;
}

// get 'h' values as array for hidden layer (e.g. for autoencoder bottleneck)
Array* Network::get_hidden(int layer_index){
    int* shape=layer[layer_index]->get_layer_shape();
    Array* result=new Array(shape);
    int *nd_index=new int[result->get_dimensions()];
    for (int d=0;d<result->get_dimensions();d++){
        for (int i=0;i<result->get_size(d);i++){
            nd_index[d]=i;
            result->set(nd_index,layer[layer_index]->neuron[layer[layer_index]->neuronindex(nd_index)]->h);
        }
    }
    return result;    
}

// set a single label value via 1-dimensional index (with auto-scaling)
void Network::set_label(int index, double value){
    layer[layers-1]->neuron[index]->x = value;
    layer[layers-1]->neuron[index]->label_min = fmin(value,layer[layers-1]->neuron[index]->label_min);
    layer[layers-1]->neuron[index]->label_max = fmax(value,layer[layers-1]->neuron[index]->label_max);
    layer[layers-1]->neuron[index]->label_maxabs = fmax(layer[layers-1]->neuron[index]->label_maxabs,abs(value));
    if (backprop_iterations<10){
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.5+value*0.5;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.5+layer[layers-1]->neuron[index]->label_mdev2*0.5;
    }
    else if (backprop_iterations<200){
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.95+value*0.05;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.95+layer[layers-1]->neuron[index]->label_mdev2*0.05;
    }
    else{
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.999+value*0.001;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.999+layer[layers-1]->neuron[index]->label_mdev2*0.001;
    }
    layer[layers-1]->neuron[index]->label_stddev = sqrt(layer[layers-1]->neuron[index]->label_variance);    

    switch (scaling_method){
        case none:
            layer[layers-1]->neuron[index]->scaled_label = value;
            break;
        case maxabs:
            // -1 to 1
            layer[layers-1]->neuron[index]->scaled_label = value / (layer[layers-1]->neuron[index]->label_maxabs + __DBL_EPSILON__);
            break;
        case normalized:
            // 0 to 1
            layer[layers-1]->neuron[index]->scaled_label = value - layer[layers-1]->neuron[index]->label_min;
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_max - layer[layers-1]->neuron[index]->label_min + __DBL_EPSILON__);
            break;
        default:
            // standardized (µ=0, sigma=1)
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h - layer[layers-1]->neuron[index]->label_rolling_average; 
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_stddev + __DBL_EPSILON__);
            break;
    }
}

// set a single label via n-dimensional index
void Network::set_label(int* nd_index,double value){
    int index = layer[layers-1]->neuronindex(nd_index);
    layer[layers-1]->neuron[index]->x = value;
    layer[layers-1]->neuron[index]->label_min = fmin(value,layer[layers-1]->neuron[index]->label_min);
    layer[layers-1]->neuron[index]->label_max = fmax(value,layer[layers-1]->neuron[index]->label_max);
    layer[layers-1]->neuron[index]->label_maxabs = fmax(layer[layers-1]->neuron[index]->label_maxabs,abs(value));
    if (backprop_iterations<10){
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.5+value*0.5;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.5+layer[layers-1]->neuron[index]->label_mdev2*0.5;
    }
    else if (backprop_iterations<200){
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.95+value*0.05;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.95+layer[layers-1]->neuron[index]->label_mdev2*0.05;
    }
    else{
        layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.999+value*0.001;
        layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
        layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.999+layer[layers-1]->neuron[index]->label_mdev2*0.001;
    }
    layer[layers-1]->neuron[index]->label_stddev = sqrt(layer[layers-1]->neuron[index]->label_variance);    

    switch (scaling_method){
        case none:
            layer[layers-1]->neuron[index]->scaled_label = value;
            break;
        case maxabs:
            // -1 to 1
            layer[layers-1]->neuron[index]->scaled_label = value / (layer[layers-1]->neuron[index]->label_maxabs + __DBL_EPSILON__);
            break;
        case normalized:
            // 0 to 1
            layer[layers-1]->neuron[index]->scaled_label = value - layer[layers-1]->neuron[index]->label_min;
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_max - layer[layers-1]->neuron[index]->label_min + __DBL_EPSILON__);
            break;
        default:
            // standardized (µ=0, sigma=1)
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h - layer[layers-1]->neuron[index]->label_rolling_average; 
            layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_stddev + __DBL_EPSILON__);
            break;
    }
}

// set all labels at once via n-dimensional data array, with auto-scaling
void Network::set_labels(Array* nd_array){
    int* nd_index = new int[nd_array->get_dimensions()];
    for (int d=0;d<nd_array->get_dimensions();d++){
        for (int i=0;i<nd_array->get_size(d);i++){
            nd_index[d]=i;
            int index=nd_array->get_element(nd_index);
            double value=nd_array->get(index);
            layer[layers-1]->neuron[index]->x = value;
            layer[layers-1]->neuron[index]->label_min = fmin(value,layer[layers-1]->neuron[index]->label_min);
            layer[layers-1]->neuron[index]->label_max = fmax(value,layer[layers-1]->neuron[index]->label_max);
            layer[layers-1]->neuron[index]->label_maxabs = fmax(layer[layers-1]->neuron[index]->label_maxabs,abs(value));
            if (backprop_iterations<10){
                layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.5+value*0.5;
                layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
                layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.5+layer[layers-1]->neuron[index]->label_mdev2*0.5;
            }
            else if (backprop_iterations<200){
                layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.95+value*0.05;
                layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
                layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.95+layer[layers-1]->neuron[index]->label_mdev2*0.05;
            }
            else{
                layer[layers-1]->neuron[index]->label_rolling_average = layer[layers-1]->neuron[index]->label_rolling_average*0.999+value*0.001;
                layer[layers-1]->neuron[index]->label_mdev2 = pow((value-layer[layers-1]->neuron[index]->label_rolling_average),2);
                layer[layers-1]->neuron[index]->label_variance = layer[layers-1]->neuron[index]->label_variance*0.999+layer[layers-1]->neuron[index]->label_mdev2*0.001;
            }
            layer[layers-1]->neuron[index]->label_stddev = sqrt(layer[layers-1]->neuron[index]->label_variance);       

            switch (scaling_method){
                case none:
                    layer[layers-1]->neuron[index]->scaled_label = value;
                    break;
                case maxabs:
                    // -1 to 1
                    layer[layers-1]->neuron[index]->scaled_label = value / (layer[layers-1]->neuron[index]->label_maxabs + __DBL_EPSILON__);
                    break;
                case normalized:
                    // 0 to 1
                    layer[layers-1]->neuron[index]->scaled_label = value - layer[layers-1]->neuron[index]->label_min;
                    layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_max - layer[layers-1]->neuron[index]->label_min + __DBL_EPSILON__);
                    break;
                default:
                    // standardized (µ=0, sigma=1)
                    layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h - layer[layers-1]->neuron[index]->label_rolling_average; 
                    layer[layers-1]->neuron[index]->scaled_label = layer[layers-1]->neuron[index]->h / (layer[layers-1]->neuron[index]->label_stddev + __DBL_EPSILON__);
                    break;
            }
        }
    }
}

// autoencode (set inputs as labels)
void Network::autoencode(){
    if (layer[0]->get_dimensions()!=layer[layers-1]->get_dimensions() || layer[0]->get_neurons()!=layer[layers-1]->get_neurons()){return;}
    int items=layer[0]->get_neurons();
    for (int j=0;j<items;j++){
        set_label(j,layer[0]->neuron[j]->x);
    }
}

// get average loss per output neuron across all backprop iterations
double Network::get_loss_avg(){
    double result=0;
    if (backprop_iterations==0){return result;}
    int n=layer[layers-1]->get_neurons();
    for (int j=0;j<n;j++){
        result+=layer[layers-1]->neuron[j]->loss_sum/backprop_iterations;
    }
    result/=n;
    return result;
}

// get average 'h'
//      (=result of activation function)
//      (=output without rescaling)
double Network::get_avg_h(){
    double result=0;
    int n=layer[layers-1]->get_neurons();
    for (int j=0;j<n;j++){
        result+=layer[layers-1]->neuron[j]->h;
    }
    result/=n;
    return result;
}

// get average output (=result after rescaling of 'h')
double Network::get_avg_output(){
    double result=0;
    int n=layer[layers-1]->get_neurons();
    for (int j=0;j<n;j++){
        result+=layer[layers-1]->neuron[j]->output;
    }
    result/=n;
    return result;
}

// save network data into file
void Network::save(string filename) {

}