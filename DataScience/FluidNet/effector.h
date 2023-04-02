#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include <cmath>
using namespace std;

class Effector{
    private:
        double raw_output;
        double rescaled_output;
        double raw_label; // label before scaling
        double scaled_label; // label after scaling (none / normalized / standardized / maxabs)
        SCALING scaling_method; // scaling method (none / normalized / standardized / maxabs)
        double label_min=__DBL_MAX__;   // used for minmax (=normalized) scaling
        double label_max=-__DBL_MAX__;   // used for minmax (=normalized) scaling
        double label_maxabs=__DBL_EPSILON__;// used for maxabs label scaling
        double label_rolling_average = 0; // rolling mean average, used for standardized label scaling
        double label_mdev2 = 0;         // squared mean deviation, used for standardized label scaling
        double label_variance = 1;      // used for standardized label scaling, calculated as approximation from rolling mdev2
        double label_stddev=1;          // standard deviation, used for standardized label scaling
        int iterations=0;
        double gradient=0;              // used for output error (derivative of loss function)
        double loss=0;                  // for MSE loss
        double loss_sum=0;              // used for average loss (after dividing by number of backprop iterations) 
        double loss_avg=0;              // average loss over all iterations        
    protected:
    public:
        double get_output(){return rescaled_output;}
        double get_gradient(){return gradient;}
        int from_index; // index of the associated neuron
        double get_loss_avg(){return loss_avg;}
        // set label (required before attempting backpropagation)
        void set_label(double value){
            iterations++;        
            raw_label = value;          
            switch (scaling_method){
                case none:
                    scaled_label = value;
                    break;
                case maxabs: // -1 to 1
                    label_maxabs = fmax(label_maxabs,abs(value));
                    scaled_label = value / label_maxabs;
                    if (scaled_label!=scaled_label){scaled_label =__DBL_EPSILON__;} // NaN protection
                    break;
                case normalized: // 0 to 1
                    label_min = fmin(value,label_min);
                    label_max = fmax(value,label_max);                    
                    scaled_label = value - label_min;
                    scaled_label = scaled_label / (label_max - label_min);
                    if (scaled_label!=scaled_label){scaled_label =__DBL_EPSILON__;} // NaN protection
                    break;
                default: // standardized (µ=0, sigma=1)
                    if (iterations<10){
                        label_rolling_average = label_rolling_average*0.5+value*0.5;
                        label_mdev2 = pow((value-label_rolling_average),2);
                        label_variance = label_variance*0.5+label_mdev2*0.5;     
                    }
                    else if (iterations<200){
                        label_rolling_average = label_rolling_average*0.95+value*0.05;
                        label_mdev2 = pow((value-label_rolling_average),2);
                        label_variance = label_variance*0.95+label_mdev2*0.05;     
                    }
                    else{
                        label_rolling_average = label_rolling_average*0.999+value*0.001;
                        label_mdev2 = pow((value-label_rolling_average),2);
                        label_variance = label_variance*0.999+label_mdev2*0.001;         
                    } 
                    label_stddev = sqrt(label_variance);                  
                    scaled_label = value - label_rolling_average; 
                    scaled_label /= label_stddev;
                    if (scaled_label!=scaled_label){scaled_label =__DBL_EPSILON__;} // NaN protection
                    break;
            }         
            // get global error: derivative of the 0.5err^2 loss function (MSE) = scaled_label-raw_output
            gradient = scaled_label - raw_output;

            // 0.5err^2 loss (current / cumulative / average)
            loss = 0.5 * gradient * gradient;
            loss_sum += loss;
            loss_avg = loss_sum / iterations;
        }

        // set output result from associated neuron and apply rescaling
        void set_result(double value){
            raw_output = value;          
            switch (scaling_method){
                case none:
                    rescaled_output = value;
                    break;
                case maxabs: // -1 to 1
                    rescaled_output = value * label_maxabs;
                    break;
                case normalized: // 0 to 1
                    rescaled_output = raw_output * (label_max - label_min) + label_min;
                    break;
                default: // standardized (µ=0, sigma=1)
                    rescaled_output = raw_output * label_stddev + label_rolling_average;
                    break;
            }  
        }

        // return rescaled result
        double get(){return rescaled_output;}

        // constructor
        Effector(SCALING scaling_method=normalized){
            this->scaling_method=scaling_method;
        }

        // destructor
        ~Effector(){}
};
