// author: 'cyberchriz' (Christian Suer)
// to be used as helper class in conjunction with <neuralnet2.h>
#pragma once
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include </home/christian/Documents/own_code/c++/DataScience/NeuralNet2/neuron2.h>
#define MAX_NEURONS 1000
using namespace std;

class Layer{
    private:
        int* layer_shape;
        int* input_shape;
        int neurons;
        OPTIMIZATION_METHOD opt_method;
        ACTIVATION_FUNC activation;
        bool self_attention;
        bool recurrent;
        int layer_dimensions;
        int input_dimensions;
    protected:
    public:
        Neuron* neuron[MAX_NEURONS];    
        int get_neurons(){return neurons;};
        OPTIMIZATION_METHOD get_opt_method(){return opt_method;};
        int neuronindex(const int *index);
        bool is_recurrent(){return recurrent;};
        bool has_selfattention(){return self_attention;};
        ACTIVATION_FUNC get_activation(){return activation;};
        int* get_layer_shape(){return this->layer_shape;};
        int get_dimensions(){return layer_dimensions;};
        // constructor
        Layer(int* _layer_shape, int* _input_shape, OPTIMIZATION_METHOD _opt_method=Vanilla, ACTIVATION_FUNC _activation=f_tanh, bool _self_attention=false, bool _recurrent=false);
        // destructor
        ~Layer();
};

// constructor definition
Layer::Layer(int* _layer_shape, int* _input_shape, OPTIMIZATION_METHOD _opt_method, ACTIVATION_FUNC _activation, bool _self_attention, bool _recurrent){
    this->input_shape=_input_shape;
    this->opt_method=_opt_method;
    this->activation=_activation;
    this->self_attention=_self_attention;
    this->recurrent=_recurrent;
    // setup layer shape
    layer_dimensions=sizeof(_layer_shape)/__SIZEOF_POINTER__;
    layer_shape = new int[layer_dimensions];
    layer_shape[0]=_layer_shape[0];
    neurons=layer_shape[0];
    for (int i=1;i<layer_dimensions;i++){
        layer_shape[i]=_layer_shape[i];
        neurons*=layer_shape[i];
    }
    // setup input shape
    input_dimensions=sizeof(_input_shape)/__SIZEOF_POINTER__;
    input_shape = new int[input_dimensions];
    input_shape[0]=_input_shape[0];
    for (int i=1;i<input_dimensions;i++){
        input_shape[i]=_input_shape[i];
    }
    // setup neurons
    for (int i=0;i<neurons;i++){
        neuron[i] = new Neuron(input_shape, layer_shape, opt_method, self_attention, recurrent);
    }
}

// destructor definition
Layer::~Layer(){
    for (int i=0;i<neurons;i++){
        delete neuron[i];
    }
    delete layer_shape;
    delete input_shape;
}

// get 1d neuron index from multidimensional index
int Layer::neuronindex(const int *index){
    if (sizeof(index)/__SIZEOF_POINTER__ > layer_dimensions){
        return -1;
    }
    int result=index[0];
    for (int d=1;d<layer_dimensions;d++){
        int add=index[d];
        for(int s=d;s>0;s--){
            add*=layer_shape[s-1];
        }
        result+=add;
    }
    return result;
};