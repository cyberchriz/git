#include </home/christian/Documents/own_code/c++/DataScience/FluidNet/fluidneuron.h>
#include </home/christian/Documents/own_code/c++/DataScience/FluidNet/receptor.h>
#include </home/christian/Documents/own_code/c++/DataScience/FluidNet/effector.h>
#include <vector>
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include </home/christian/Documents/own_code/c++/DataScience/distributions/random_distributions.h>
#include </home/christian/Documents/own_code/c++/DataScience/activation_functions.h>
using namespace std;

class FluidNet{
    private:
        int neurons;
        int inputs;
        int outputs;
        vector<FluidNeuron> neuron;
        vector<Receptor> receptor;
        vector<Effector> effector;
        void fire(int index);
        void push_gradient(int index);
        void remap();
        OPTIMIZATION_METHOD method=Vanilla;
        double lr=0.001;
        double lr_momentum=0.0;
        double lr_decay=0.0;
        double opt_beta1=0.9;
        double opt_beta2=0.99;        
        int backprop_iterations=0;
        bool is_connected=false;
    protected:
    public:
        void add_neurons(int amount, int connections, ACTIVATION_FUNC act_func=f_LReLU);
        void connect();
        void set_learning_rate(double value){lr=fmin(fmax(0,value),1.0);}
        void set_learning_rate_decay(double value){lr_decay=fmin(fmax(0,value),1.0);}
        void set_learning_momentum(double value){lr_momentum=fmin(fmax(0,value),1.0);}   
        void set_optimizer(OPTIMIZATION_METHOD method){this->method=method;}     
        void set_input(int index, double value);
        double get_input(int index){return receptor[index].get_input();}
        double get_output(int index){return effector[index].get_output();}
        double get_loss_avg(int index){return effector[index].get_loss_avg();}
        void set_label(int index, double value);
        void run();
        void backprop();
        void autoencode();
        // constructor
        FluidNet(int inputs, int outputs, int connections, SCALING input_scaling=normalized, SCALING label_scaling=normalized);
        // destructor
        ~FluidNet(){}
};

// constructor definition
FluidNet::FluidNet(int inputs, int outputs, int connections, SCALING input_scaling, SCALING label_scaling){
    for (int j=0;j<inputs;j++){
        receptor.push_back(Receptor(input_scaling));
        neuron.push_back(FluidNeuron(connections, f_ident));
        neuron[neurons].make_input();
        receptor[j].to_index=neurons;
        neurons++;
    }
    this->inputs=inputs;
    for (int j=0;j<outputs;j++){
        effector.push_back(Effector(label_scaling));
        neuron.push_back(FluidNeuron(connections, f_ident));
        neuron[neurons].make_output();
        effector[j].from_index=neurons;
        neurons++;
    }
    this->outputs=outputs;
}

// add hidden neurons
void FluidNet::add_neurons(int amount, int connections, ACTIVATION_FUNC act_func){
    for (int n=0;n<amount;n++){
        neuron.push_back(FluidNeuron(connections, act_func));
        neurons++;
    }
    is_connected=false;
}

// connect weights between neurons
void FluidNet::connect(){
    for (int target_neuron=0;target_neuron<neurons;target_neuron++){
        // skip if all inputs are already completely connected
        if (neuron[target_neuron].inputs_connected>=neuron[target_neuron].connections){
            continue;
        }
        for (int target_weight=0;target_weight<neuron[target_neuron].connections;target_weight++){
            // skip if this connection already exists
            if (neuron[target_neuron].incoming[target_weight].from_index>=0){
                continue;
            }
            // randomly select a source neuron for this weight
            int source_neuron=floor(rand_uni()*neurons);
            // default value of from_index=-1 indicates that this target weight isn't connected yet
            int neurons_checked=0;
            while(neuron[target_neuron].incoming[target_weight].from_index<0){
                // find a different source neuron if source+target are identical
                if (source_neuron==target_neuron){source_neuron++;}
                // restart a neuron index 0 if end of array is reached
                if (source_neuron>=neurons){source_neuron=0;}
                // check if source_neuron still has free output connections
                if (neuron[source_neuron].outputs_connected<neuron[source_neuron].connections){
                    // find index of free output connection
                    for (int i=0;i<neuron[source_neuron].connections;i++){
                        if (neuron[source_neuron].outgoing[i].to_index<0){
                            // write connection data
                            neuron[target_neuron].incoming[target_weight].from_index=source_neuron;
                            neuron[target_neuron].inputs_connected++;                            
                            neuron[source_neuron].outgoing[i].to_index=target_neuron;
                            neuron[source_neuron].outputs_connected++;
                            break; //breaks the for loop with established connection
                        }
                    }
                    break; // breaks the while loop after successful connection
                           // continue with next incoming weight of given neuron (outer for-loop)
                }
                // increment source neuron (and continue while loop) if connection wasn't successful
                source_neuron++;
                neurons_checked++;
                // safety exit (avoiding infinite loop) in case all neurons were checked without success
                if (neurons_checked>=neurons){
                    break;
                }
            }
        }
    }
    is_connected=true;
}

// remap least significant weights
void FluidNet::remap(){
    for (int this_neuron=0;this_neuron<neurons;this_neuron++){
        if (!neuron[this_neuron].is_input){
            // find smallest weight
            double smallest_weight=__DBL_MAX__;
            int this_index=0;
            for (int i=0;i<neuron[this_neuron].connections;i++){
                if (fabs(neuron[this_neuron].incoming[i].weight)<smallest_weight){
                    smallest_weight=fabs(neuron[this_neuron].incoming[i].weight);
                    this_index=i;
                }
            }
            // randomly select a second neuron
            int second_neuron;
            do {second_neuron=floor(rand_uni()*neurons);} while (!neuron[second_neuron].is_input);
            // find smallest weight of second neuron
            smallest_weight=__DBL_MAX__;
            int second_index=0;
            for (int i=0;i<neuron[second_neuron].connections;i++){
                if (fabs(neuron[second_neuron].incoming[i].weight)<smallest_weight){
                    smallest_weight=fabs(neuron[this_neuron].incoming[i].weight);
                    second_index=i;
                }
            }            
            // swap connections
            int temp=neuron[this_neuron].incoming[this_index].from_index;
            neuron[this_neuron].incoming[this_index].from_index = neuron[second_neuron].incoming[second_index].from_index;
            neuron[second_neuron].incoming[second_index].from_index = temp;
            // increase weights
            double factor=2.0;
            neuron[this_neuron].incoming[this_index].weight*=factor;
            neuron[second_neuron].incoming[second_index].weight*=factor;
        }
    }
}

// fire activation once all inputs are complete
void FluidNet::fire(int index){
    if (!neuron[index].is_input){
        // reset x
        neuron[index].x=0;
        // add sum of weighted inputs
        for (int i=0;i<neuron[index].connections;i++){
            neuron[index].x+=neuron[index].incoming[i].weight*neuron[neuron[index].incoming[i].from_index].h;
        }
        // add bias
        neuron[index].x+=neuron[index].bias_weight;
        // add weighted recurrent values from last iteration
        neuron[index].x+=neuron[index].m1*neuron[index].m1_weight;
        neuron[index].x+=neuron[index].m2*neuron[index].m2_weight;
        neuron[index].x+=neuron[index].m3*neuron[index].m3_weight;
        neuron[index].x+=neuron[index].m4*neuron[index].m4_weight;
        // activate
        neuron[index].h=activate(neuron[index].x,neuron[index].act_func);
        // update recurrent values
        neuron[index].m1=neuron[index].h;
        neuron[index].m2=neuron[index].m2*0.9+neuron[index].h*0.1;
        neuron[index].m3=neuron[index].m3*0.99+neuron[index].h*0.01;
        neuron[index].m4=neuron[index].m4*0.999+neuron[index].h*0.001;
    }
    // trigger subsequent neurons
    for (int n=0;n<neuron[index].connections;n++){
        int target=neuron[index].outgoing[n].to_index;
        // skip if connection is invalid
        if (target<0){continue;}
        neuron[target].inputs_received++;
        if (neuron[target].inputs_received==neuron[target].connections){
            fire(target);
        }
    }
}

// recursively push error gradient (i.e. get SUM_k[err_k*w_jk])
void FluidNet::push_gradient(int index){
    if (!neuron[index].gradient_complete()){return;}
    for (int i=0;i<neuron[index].connections;i++){
        // safety check for valid index of incoming neuron
        if (neuron[index].incoming[i].from_index<0 || neuron[index].incoming[i].from_index>=neurons){continue;}
        // push gradient to incoming neuron
        neuron[neuron[index].incoming[i].from_index].gradient+=neuron[index].gradient*neuron[index].incoming[i].weight;
        neuron[neuron[index].incoming[i].from_index].errors_received++;
        if (neuron[neuron[index].incoming[i].from_index].gradient_complete()){
            push_gradient(neuron[index].incoming[i].from_index);
        }
    }
}

// set individual network input
void FluidNet::set_input(int index, double value){
    receptor[index].set_input(value);
}

// set individual output label
void FluidNet::set_label(int index, double value){
    effector[index].set_label(value);
}

// recursively get all neurons to fire
void FluidNet::run(){
    // make neuron connections (in case the network is new or after adding neurons)
    if (!is_connected){connect();}
    // reset received inputs
    for (int n=0;n<neurons;n++){
        neuron[n].inputs_received=0;
    }
    // write scaled receptor inputs to associated neurons, then fire
    for (int n=0;n<inputs;n++){
        int target=receptor[n].to_index;
        neuron[target].h=receptor[n].get_scaled_input();
        neuron[target].inputs_received=neuron[target].connections;
        fire(target);
    }
    /* remap all neurons that didn't fire (so that they do)
    for (int n=0;n<neurons;n++){
        while (!neuron[n].inputs_complete()){
            // find a second neuron (indexed as 'm') that hasn't yet fired (but has at least one received input)
            int m=floor(rand_uni()*neurons);
            int neurons_checked=0;
            while (neuron[m].inputs_complete() || neuron[m].inputs_received==0){
                m++;
                neurons_checked++;
                if (m>neurons){m=0;}
                // safety exit in order to avoid infinite loop
                if (neurons_checked>neurons){break;}
            }
            // exit while loop if no suitable second neuron has been found
            if (neurons_checked>neurons){break;}
            // move a received connection from second neuron to current neuron
            int source_neuron;
            for (int i=0;i<neuron[m].connections;i++){
                if (neuron[neuron[m].incoming[i].from_index].inputs_complete()){
                    source_neuron=neuron[m].incoming[i].from_index;
                    // remove found connection from second neuron
                    neuron[m].inputs_received--;
                    break;
                }
            }
            // find an unused connection in current neuron
            for (int i=0;i<neuron[n].connections;i++){
                if (!neuron[neuron[n].incoming[i].from_index].inputs_complete()){
                    neuron[n].incoming[i].from_index=source_neuron;
                    neuron[n].inputs_received++;
                }
            }
            // fire neuron once all inputs are complete
            if (neuron[n].inputs_complete()){fire(n);}
        }
    } */
    // write results to effectors
    for (int n=0;n<outputs;n++){
        effector[n].set_result(neuron[effector[n].from_index].h);
    }
}

// run backpropagation
void FluidNet::backprop(){
    backprop_iterations++;
    // reset received errors
    for (int n=0;n<neurons;n++){
        neuron[n].errors_received=0;
        neuron[n].gradient=0;
    }
    // read effector errors to outputs
    for (int n=0;n<outputs;n++){
        neuron[effector[n].from_index].gradient=effector[n].get_gradient();
        neuron[effector[n].from_index].errors_received=neuron[effector[n].from_index].connections;
    }
    // push gradients (map hidden error gradients, i.e. SUM_k[err_k*w_jk])
    for (int n=0;n<outputs;n++){
        push_gradient(effector[n].from_index);
    }
    // update weights; general rule: delta w_ij=lr*error*act'(net_inp_j)*out_i   
    for (int n=0;n<neurons;n++){
        for (int i=0;i<neuron[n].connections;i++){
            if (neuron[n].incoming[i].from_index<0 || neuron[n].incoming[i].from_index>=neurons){continue;}
            if (method==Vanilla){
                // get delta
                neuron[n].incoming[i].weight_delta = lr_momentum*neuron[n].incoming[i].weight_delta + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x, neuron[n].act_func) * neuron[neuron[n].incoming[i].from_index].h;
                // update
                neuron[n].incoming[i].weight += neuron[n].incoming[i].weight_delta;
            }
            else if (method==Nesterov){
                // lookahead step
                double lookahead = (lr_momentum*neuron[n].incoming[i].weight_delta) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[neuron[n].incoming[i].from_index].h;
                // momentum step
                neuron[n].incoming[i].weight_delta = (lr_momentum*lookahead) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[neuron[n].incoming[i].from_index].h;
                // update step
                neuron[n].incoming[i].weight += neuron[n].incoming[i].weight_delta;
            }
            else if (method==RMSprop){
                // opt_v update
                neuron[n].incoming[i].opt_v = lr_momentum*neuron[n].incoming[i].opt_v + (1-lr_momentum)*pow(deactivate(neuron[n].x,neuron[n].act_func),2) * neuron[n].gradient;
                // get delta
                neuron[n].incoming[i].weight_delta = (lr/(1+lr_decay*backprop_iterations)) / (sqrt(neuron[n].incoming[i].opt_v+1e-8)+__DBL_MIN__) * pow(neuron[n].h,2) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h;
                // update
                neuron[n].incoming[i].weight += neuron[n].incoming[i].weight_delta;
            }
            else if (method==ADADELTA){
                // opt_v update
                neuron[n].incoming[i].opt_v = opt_beta1 * neuron[n].incoming[i].opt_v + (1-opt_beta1) * pow(deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h,2);
                // opt_w update
                neuron[n].incoming[i].opt_w = (opt_beta1 * pow(neuron[n].incoming[i].opt_w,2)) + (1-opt_beta1)*pow(neuron[n].incoming[i].weight_delta,2);
                // get delta
                neuron[n].incoming[i].weight_delta = sqrt(neuron[n].incoming[i].opt_w+1e-8)/(sqrt(neuron[n].incoming[i].opt_v+1e-8)+__DBL_MIN__) * deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h;
                // update
                neuron[n].incoming[i].weight+=neuron[n].incoming[i].weight_delta;
            }
            else if (method==ADAM){ // =ADAM without minibatch
                // opt_v update
                neuron[n].incoming[i].opt_v = opt_beta1 * neuron[n].incoming[i].opt_v + (1-opt_beta1) * deactivate(neuron[n].x, neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h;
                // opt_w update
                neuron[n].incoming[i].opt_w = opt_beta2 * neuron[n].incoming[i].opt_w * pow(deactivate(neuron[n].x, neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h,2);
                // get delta
                double v_t = neuron[n].incoming[i].opt_v/(1-opt_beta1);
                double w_t = neuron[n].incoming[i].opt_w/(1-opt_beta2);
                neuron[n].incoming[i].weight_delta = (lr/(1+lr_decay*backprop_iterations)) * (v_t/(sqrt(w_t+1e-8))+__DBL_MIN__);
                // update
                neuron[n].incoming[i].weight+=neuron[n].incoming[i].weight_delta;
            }
            else if (method==AdaGrad){
                // opt_v update
                neuron[n].incoming[i].opt_v = neuron[n].incoming[i].opt_v + pow(deactivate(neuron[n].x, neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h,2);
                // get delta
                neuron[n].incoming[i].weight_delta = ((lr/(1+lr_decay*backprop_iterations)) / sqrt(neuron[n].incoming[i].opt_v +1e-8)) * deactivate(neuron[n].x, neuron[n].act_func) * neuron[n].gradient * neuron[neuron[n].incoming[i].from_index].h;
                // update
                neuron[n].incoming[i].weight+=neuron[n].incoming[i].weight_delta;
            }                
        }
        // update bias weights (Vanilla)
        neuron[n].delta_b = (lr_momentum*neuron[n].delta_b) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func);
        neuron[n].bias_weight += neuron[n].delta_b;
        // update recurrent weights (Vanilla)
        neuron[n].delta_m1 = (lr_momentum*neuron[n].delta_m1) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].m1;
        neuron[n].delta_m2 = (lr_momentum*neuron[n].delta_m2) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].m2;
        neuron[n].delta_m3 = (lr_momentum*neuron[n].delta_m3) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].m3;
        neuron[n].delta_m4 = (lr_momentum*neuron[n].delta_m4) + (1-lr_momentum)*(lr/(1+lr_decay*backprop_iterations)) * neuron[n].gradient * deactivate(neuron[n].x,neuron[n].act_func) * neuron[n].m4;
        neuron[n].m1_weight += neuron[n].delta_m1;
        neuron[n].m2_weight += neuron[n].delta_m2;
        neuron[n].m3_weight += neuron[n].delta_m3;
        neuron[n].m4_weight += neuron[n].delta_m4;    
    }
    // remap least significant weights
    remap();
}

void FluidNet::autoencode(){
    for (int n=0;n<fmin(outputs, inputs);n++){
        effector[n].set_label(receptor[n].get_input());
    }
}