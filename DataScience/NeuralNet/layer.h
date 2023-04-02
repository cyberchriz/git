#pragma once
#include <enums.h>
#include <activation_functions.h>
#include <neuron.h>
using namespace std;

class LayerObject{
    private:
    public:
        LAYER_TYPE type;
        ACTIVATION_FUNC activation;    
        double bias_error;
        double rSSG=0; // SSG = square root of sum of squared gradients
        double rSSG_average=0.5;
        double avg_sum_of_weights;
        int dimensions=0;
        int neurons_x=0;
        int neurons_y=0;
        int neurons_z=0;
        int neurons_total=0;
        vector<NeuronObject> neuron1d;
        vector<vector<NeuronObject>> neuron2d;
        vector<vector<vector<NeuronObject>>> neuron3d;        
        LayerObject(vector<int> neurons_per_dimension, vector<int> weightinputs_per_neuron, OPTIMIZATION_METHOD opt_method=Vanilla, LAYER_TYPE type=standard, ACTIVATION_FUNC activation=f_tanh){
            this->type = type;
            this->activation = activation;
            dimensions = neurons_per_dimension.size();
            if (dimensions>=1){neurons_x = neurons_per_dimension[0];neurons_total=neurons_x;}
            if (dimensions>=2){neurons_y = neurons_per_dimension[1];neurons_total*=neurons_y;}
            if (dimensions>=3){neurons_z = neurons_per_dimension[2];neurons_total*=neurons_z;}
            // create neurons
            if (dimensions==1){
                for (int x=0;x<neurons_x;x++){
                    neuron1d.push_back(NeuronObject(weightinputs_per_neuron, opt_method, type));
                }
            }
            if (dimensions==2){
                for (int x=0;x<neurons_x;x++){
                    vector<NeuronObject> y_column;
                    for (int y=0;y<neurons_y;y++){
                        y_column.push_back(NeuronObject(weightinputs_per_neuron, opt_method, type));
                    }
                    neuron2d.push_back(y_column);
                }
            }
            if (dimensions==3){
                for (int x=0;x<neurons_x;x++){
                    vector<vector<NeuronObject>> yz_column;
                    for (int y=0;y<neurons_y;y++){
                        vector<NeuronObject> z_stack;
                        for (int z=0;z<neurons_z;z++){
                            z_stack.push_back(NeuronObject(weightinputs_per_neuron, opt_method, type));
                        }
                        yz_column.push_back(z_stack);
                    }
                    neuron3d.push_back(yz_column);
                }
            }
        };
        ~LayerObject(){};
    };
