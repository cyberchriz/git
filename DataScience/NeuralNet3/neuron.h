#pragma once
#include </home/christian/Documents/own_code/c++/DataScience/distributions/random_distributions.h>
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include <vector>
using namespace std;

struct IncomingStruct{
    int from_neuron=-1;
    int from_layer;
    double weight;
    double weight_delta=0;
    double opt_v=0;                   // used for RMSprop, ADADELTA and ADAM
    double opt_w=0;                   // used for ADADELTA and ADAM    
};

struct OutgoingStruct{
    int to_neuron=-1;
    int to_weight;
    int to_layer;
};

class Neuron{
    private:
    protected:
    public:
        double x=0; // holds sum of weighted inputs
        double y=0; // temporary storage for sum of weighted inputs from same layer
        double z=0; // temporary storage for errors from same layer
        double h=0; // result of activation function
        double m1=0; // recurrent input
        double m2=0; // recurent input (coeff. 0.9)
        double m3=0; // recurrent input (coeff. 0.99)
        double m4=0; // recurrent input (coeff. 0.999)
        double m1_weight;
        double m2_weight;
        double m3_weight;
        double m4_weight;
        double delta_m1=0;
        double delta_m2=0;
        double delta_m3=0;
        double delta_m4=0;
        double bias_weight;
        double delta_b;
        int lower_connections=0;
        int level_connections=0;
        int connections=5; // for memory weights and bias
        vector<IncomingStruct> incoming;
        vector<OutgoingStruct> outgoing;
        double gradient;
        ACTIVATION_FUNC act_func;
        // add an incoming weight and return the index of this new weight
        int add_incoming(int from_layer=0, int from_neuron=0, bool initialize=true){
            incoming.push_back(IncomingStruct());
            int weight_index=incoming.size()-1;
            incoming[weight_index].from_layer=from_layer;
            incoming[weight_index].from_neuron=from_neuron;
            connections++;
            if (initialize){
                incoming[weight_index].weight=rand_norm()/(connections);
            }
            return incoming.size()-1;
        }
        // add an outgoing weight and return the index of this new weight
        int add_outgoing(int to_layer=0, int to_neuron=0, int to_weight=0){
            outgoing.push_back(OutgoingStruct());
            int weight_index=outgoing.size()-1;
            outgoing[weight_index].to_layer=to_layer;
            outgoing[weight_index].to_neuron=to_neuron;
            outgoing[weight_index].to_weight=to_weight;
            return weight_index;
        }
        // randomly initialize the incoming weight with the specified index
        void initialize_weight(int weight_index){
            incoming[weight_index].weight=rand_norm()/(connections);
        }
        // constructor
        Neuron(ACTIVATION_FUNC act_func=f_LReLU){
            this->act_func=act_func;
            // initialize bias and recurrent weights
            m1_weight=rand_norm()/(connections);
            m2_weight=rand_norm()/(connections);
            m3_weight=rand_norm()/(connections);
            m4_weight=rand_norm()/(connections);
            bias_weight=rand_norm()/(connections);
        }
        // destructor
        ~Neuron(){
        }
};