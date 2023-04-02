#include </home/christian/Documents/own_code/c++/DataScience/distributions/random_distributions.h>
#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
using namespace std;

struct IncomingStruct{
    int from_index=-1;
    double weight;
    double weight_delta;
    double opt_v;                   // used for RMSprop, ADADELTA and ADAM
    double opt_w;                   // used for ADADELTA and ADAM    
};

struct OutgoingStruct{
    int to_index=-1;
};

class FluidNeuron{
    private:
    protected:
    public:
        double x=0; // holds sum of weighted inputs
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
        IncomingStruct *incoming;
        OutgoingStruct *outgoing;
        double gradient;
        int inputs_received=0; // counter for received inputs (fire when complete)
        int errors_received=0; // counter for received hidden errors from subsequent neurons (for backprop)
        int outputs_connected=0; // counter for established outgoing connections
        int inputs_connected=0;
        bool is_input=false;
        bool is_output=false;
        int connections;
        ACTIVATION_FUNC act_func;
        bool inputs_complete(){return inputs_received==connections;}
        bool gradient_complete(){return errors_received==connections;}
        void make_input(){is_input=true;is_output=false;}
        void make_output(){is_output=true;is_input=false;outputs_connected=connections;}
        // constructor
        FluidNeuron(int connections, ACTIVATION_FUNC act_func=f_LReLU){
            this->connections = connections;
            this->act_func=act_func;
            // initialize weights (incoming, recurrent, bias)
            incoming = new IncomingStruct[connections];
            for (int j=0;j<connections;j++){
                incoming[j].weight=rand_norm()/(connections+5); // plus 5 in order to account for bias and recurrent inputs
            }
            m1_weight=rand_norm()/(connections+5);
            m2_weight=rand_norm()/(connections+5);
            m3_weight=rand_norm()/(connections+5);
            m4_weight=rand_norm()/(connections+5);
            bias_weight=rand_norm()/(connections+5);
            outgoing = new OutgoingStruct[connections];
        }
        // destructor
        ~FluidNeuron(){
            // delete incoming;
            // delete outgoing;
        }
};