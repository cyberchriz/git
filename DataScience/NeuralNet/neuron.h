#pragma once
#include <cmath>
#include <vector>
#include <enums.h>
#include <activation_functions.h>
using namespace std;

class NeuronObject
{
private:
public:
    // core parameters
    bool dropout=false;             // dropout for overtraining mitigation
    double x=0;                     // receives sum of weighted inputs
    double h=0;                     // for result of activation function (=output of hidden neurons; also used for input value of initial layer)
    double scaled_label;            // used for output error calculation
    double output;                  // used for rescaled output (rescaling of h)

    // recurrent inputs ("memory")    
    double m1=1;                    // stores h value from last iteration
    double m2=1;                    // stores rolling average with ratio 9:1 previous average versus new value for h
    double m3=1;                    // stores rolling average with ratio 99:1 previous average versus new value for h
    double m4=1;                    // stores rolling average with ratio 999:1 previous average versus new value for h

    // error calculations
    double gradient=0;              // used for output error (derivative of loss function) or hidden error, depending on the layer
    double loss=0;                  // for MSE loss
    double loss_sum=0;              // used for average loss (after dividing by number of backprop iterations) 

    // values for input and label scaling
    double input_min=__DBL_MAX__;   // used for minmax scaling
    double input_max=-__DBL_MAX__;   // used for minmax scaling
    double input_maxabs=__DBL_EPSILON__;// used for input scaling
    double input_rolling_average = 0; // used for input scaling
    double input_mdev2 = 0;         // used for input scaling
    double input_variance = 1;      // used for input scaling, calculated as approximation from rolling mdev2
    double input_stddev=1;          // used for standardized scaling
    double label;                   // true/target values (before scaling)
    double label_min=__DBL_MAX__;   // used for label minmax scaling
    double label_max=-__DBL_MAX__;  // used for label minmax scaling
    double label_maxabs=__DBL_EPSILON__;// used for label scaling
    double label_rolling_average = 0; // used for label scaling
    double label_mdev2 = 0;         // used for label scaling
    double label_variance = 1;      // used for label scaling, calculated as approximation from rolling mdev2
    double label_stddev =1;         // used for standardized label scaling

    // weights and weight-deltas for recurrent inputs
    double m1_weight=0;
    double m2_weight=0;
    double m3_weight=0;
    double m4_weight=0;
    double delta_m1=0;
    double delta_m2=0;
    double delta_m3=0;
    double delta_m4=0;    

    // bias weight
    double bias_weight=0;
    double delta_b=0;                 // calculated bias correction from backpropagation  

    // dimensions of incoming weight vector from lower layer
    int input_dimensions=0;
    int inputs_x=0;
    int inputs_y=0;
    int inputs_z=0;
    int inputs_total=0;

    // for 1-dimensional incoming weight vector   
    vector<double> input_weight_1d;    
    vector<double> delta_w_1d;      // calculated weight correction from backpropagation     
    vector<double> opt_v_1d;        // used for RMSprop, Adadelta and Adam
    vector<double> opt_w_1d;        // used for Adadelta and Adam

    // for 2-dimensional incoming weight vector
    vector<vector<double>> input_weight_2d;    
    vector<vector<double>> delta_w_2d;   
    vector<vector<double>> opt_v_2d;
    vector<vector<double>> opt_w_2d;

    // for 3-dimensional incoming weight vector
    vector<vector<vector<double>>> input_weight_3d;
    vector<vector<vector<double>>> delta_w_3d;      
    vector<vector<vector<double>>> opt_v_3d;
    vector<vector<vector<double>>> opt_w_3d; 

    // constructor
    NeuronObject(vector<int> weightinputs, OPTIMIZATION_METHOD opt_method, LAYER_TYPE type=standard);

    // destructor
    ~NeuronObject();
};

NeuronObject::NeuronObject(vector<int> weightinputs, OPTIMIZATION_METHOD opt_method, LAYER_TYPE type){
    // get dimensions
    input_dimensions=weightinputs.size();
    inputs_x=weightinputs[0];
    inputs_total=inputs_x;
    if (input_dimensions>1){
        inputs_y=weightinputs[1];
        inputs_total*=fmin(inputs_y,1);
    }
    if (input_dimensions>2){
        inputs_z=weightinputs[2];
        inputs_total*=fmin(inputs_z,1);
    }

    // return without any weight initializations if neuron has no incoming weights (=input layer)
    if (inputs_total==0){return;}

    // initialize weight coefficients for 1d input vector
    if (input_dimensions==1){
        for (int x=0;x<inputs_x;x++){
            double init = ((((double)rand())/RAND_MAX)) / (inputs_total+1);
            input_weight_1d.push_back(init);
            delta_w_1d.push_back(0);
        }
        if (opt_method==RMSprop || opt_method==ADADELTA || opt_method==ADAM || opt_method==AdaGrad){
            for (int x=0;x<inputs_x;x++){
                opt_v_1d.push_back(1);
            }
        }
        if (opt_method==ADADELTA || opt_method==ADAM){
            for (int x=0;x<inputs_x;x++){
                opt_w_1d.push_back(1);
            }
        }
    }   

    // resize weight coefficients for 2d input vector
    else if (input_dimensions==2){
        // initialize 2d vectors of incoming weights and deltas
        for (int x=0;x<inputs_x;x++){
            vector<double> weight_column;
            vector<double> delta_column;
            for (int y=0;y<inputs_y;y++){
                double init = ((((double)rand())/RAND_MAX)) / (inputs_total+1);
                weight_column.push_back(init);
                delta_column.push_back(1);
            }
            input_weight_2d.push_back(weight_column);
            delta_w_2d.push_back(delta_column);
        }
        // initialize 2d incoming opt_v vector
        if (opt_method==RMSprop || opt_method==ADADELTA || opt_method==ADAM || opt_method==AdaGrad){
            for (int x=0;x<inputs_x;x++){
                vector<double> column;
                for (int y=0;y<inputs_y;y++){
                    column.push_back(1);
                }
                opt_v_2d.push_back(column);
            }
        }
        // initialize 2d incoming opt_w vector
        if (opt_method==ADADELTA || opt_method==ADAM){
            for (int x=0;x<inputs_x;x++){
                vector<double> column;
                for (int y=0;y<inputs_y;y++){
                    column.push_back(1);
                }
                opt_w_2d.push_back(column);
            }  
        }
    }

    // resize weight coefficients for 3d input vector
    else if (input_dimensions==3){
        // initialize 3d vectors of incoming weights and deltas
        for (int x=0;x<inputs_x;x++){
            vector<vector<double>> weight_column;
            vector<vector<double>> delta_column;
            for (int y=0;y<inputs_y;y++){
                vector<double> weight_stack;
                vector<double> delta_stack;
                for (int z=0;z<inputs_z;z++){
                    double init = ((((double)rand())/RAND_MAX)) / (inputs_total+1);
                    weight_stack.push_back(init);
                    delta_stack.push_back(0);
                }
                weight_column.push_back(weight_stack);
                delta_column.push_back(delta_stack);
            }
            input_weight_3d.push_back(weight_column);
            delta_w_3d.push_back(delta_column);
        }
        // initialize 3d incoming opt_v vector
        if (opt_method==RMSprop || opt_method==ADADELTA || opt_method==ADAM || opt_method==AdaGrad){
            for (int x=0;x<inputs_x;x++){
                vector<vector<double>> yz_column;
                for (int y=0;y<inputs_y;y++){
                    vector<double> z_stack;
                    for (int z=0;z<inputs_z;z++){
                        z_stack.push_back(1);
                    }
                    yz_column.push_back(z_stack);
                }
                opt_v_3d.push_back(yz_column);
            }
        }
        // resize and initialize opt_w vector
        if (opt_method==ADADELTA || opt_method==ADAM){
            for (int x=0;x<inputs_x;x++){
                vector<vector<double>> yz_column;
                for (int y=0;y<inputs_y;y++){
                    vector<double> z_stack;
                    for (int z=0;z<inputs_z;z++){
                        z_stack.push_back(1);
                    }
                    yz_column.push_back(z_stack);
                }
                opt_w_3d.push_back(yz_column);
            }           
        }    
    }

    // initialize bias weight and recurrent weights
    bias_weight=((((double)rand())/RAND_MAX)) / (inputs_total+1);
    m1_weight=((((double)rand())/RAND_MAX)) / (inputs_total+1);
    m2_weight=((((double)rand())/RAND_MAX)) / (inputs_total+1);
    m3_weight=((((double)rand())/RAND_MAX)) / (inputs_total+1);
    m4_weight=((((double)rand())/RAND_MAX)) / (inputs_total+1);
};

NeuronObject::~NeuronObject(){
};
