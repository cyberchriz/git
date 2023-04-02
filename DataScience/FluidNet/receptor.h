#include </home/christian/Documents/own_code/c++/DataScience/enums.h>
#include <cmath>
using namespace std;

class Receptor{
    private:
        double raw_input; // value before scaling
        double scaled_input; // value after scaling (none / normalized / standardized / maxabs)
        SCALING scaling_method; // scaling method (none / normalized / standardized / maxabs)
        double input_min=__DBL_MAX__;   // used for minmax (=normalized) scaling
        double input_max=-__DBL_MAX__;   // used for minmax (=normalized) scaling
        double input_maxabs=__DBL_EPSILON__;// used for maxabs input scaling
        double input_rolling_average = 0; // rolling mean average, used for standardized input scaling
        double input_mdev2 = 0;         // squared mean deviation, used for standardized input scaling
        double input_variance = 1;      // used for standardized input scaling, calculated as approximation from rolling mdev2
        double input_stddev=1;          // standard deviation, used for standardized input scaling
        int iterations=0;
    protected:
    public:
        int to_index; // index of the associated neuron
        double get_scaled_input(){return scaled_input;}
        double get_input(){return raw_input;}
        void set_input(double value){
            raw_input = value;          
            switch (scaling_method){
                case none:
                    scaled_input = value;
                    break;
                case maxabs: // -1 to 1
                    input_maxabs = fmax(input_maxabs,abs(value));
                    scaled_input = value / input_maxabs;
                    if (scaled_input!=scaled_input){scaled_input =__DBL_EPSILON__;} // NaN protection
                    break;
                case normalized: // 0 to 1
                    input_min = fmin(value,input_min);
                    input_max = fmax(value,input_max);                    
                    scaled_input = value - input_min;
                    scaled_input = scaled_input / (input_max - input_min);
                    if (scaled_input!=scaled_input){scaled_input =__DBL_EPSILON__;} // NaN protection
                    break;
                default: // standardized (µ=0, sigma=1)
                    iterations++;
                    if (iterations<10){
                        input_rolling_average = input_rolling_average*0.5+value*0.5;
                        input_mdev2 = pow((value-input_rolling_average),2);
                        input_variance = input_variance*0.5+input_mdev2*0.5;     
                    }
                    else if (iterations<200){
                        input_rolling_average = input_rolling_average*0.95+value*0.05;
                        input_mdev2 = pow((value-input_rolling_average),2);
                        input_variance = input_variance*0.95+input_mdev2*0.05;     
                    }
                    else{
                        input_rolling_average = input_rolling_average*0.999+value*0.001;
                        input_mdev2 = pow((value-input_rolling_average),2);
                        input_variance = input_variance*0.999+input_mdev2*0.001;         
                    } 
                    input_stddev = sqrt(input_variance);                  
                    scaled_input = value - input_rolling_average; 
                    scaled_input /= input_stddev;
                    if (scaled_input!=scaled_input){scaled_input =__DBL_EPSILON__;} // NaN protection
                    break;
            }            
        }
        // constructor
        Receptor(SCALING scaling_method=normalized){
            this->scaling_method=scaling_method;
        }
        // destructor
        ~Receptor(){}
};