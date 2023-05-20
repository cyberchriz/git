#include "../headers/neuralnet.h"

// batch training
void NeuralNet::fit(const Array<Array<double>>& features, const Array<Array<double>>& labels, const int batch_size, const int epochs){
    int total_samples = features.get_elements();
    // get scaling parameters from entire dataset
    if (scaling_method == ScalingMethod::min_max_normalized || scaling_method == ScalingMethod::mean_normalized){
        features_min = features.nested_min();
        features_max = features.nested_max();
        labels_min = labels.nested_min();
        labels_max = labels.nested_max();
    }
    if (scaling_method == ScalingMethod::mean_normalized || scaling_method == ScalingMethod::standardized){
        features_mean = features.nested_mean();
        labels_mean = labels.nested_mean();
    }
    if (scaling_method == ScalingMethod::standardized){
        features_stddev = features.nested_stddev();
        labels_stddev = labels.nested_stddev();
    }
    // iterate over epochs
    for (int epoch = 0; epoch<epochs; epoch++){
        int batch = 0;
        // iterate over samples
        for (int sample = 0; sample < total_samples; sample++){
            // process samples for current batch batch
            if (batch_counter < batch_size){

                // scale features and run prediction
                if (scaling_method==ScalingMethod::min_max_normalized){
                    predict((features[sample]-features_min).Hadamard_division(features_max-features_min),false);
                }
                if (scaling_method==ScalingMethod::mean_normalized){
                    predict((features[sample]-features_mean).Hadamard_division(features_max-features_min),false);
                }
                if (scaling_method==ScalingMethod::standardized){
                    predict((features[sample]-features_mean).Hadamard_division(features_stddev),false);
                }

                // assign scaled labels
                if (scaling_method==ScalingMethod::min_max_normalized){
                    layer[layers-1].label = (labels[sample]-labels_min).Hadamard_division(labels_max-labels_min);
                }
                if (scaling_method==ScalingMethod::min_max_normalized){
                    layer[layers-1].label = (labels[sample]-labels_mean).Hadamard_division(labels_max-labels_min);
                }
                if (scaling_method==ScalingMethod::min_max_normalized){
                    layer[layers-1].label = (labels[sample]-labels_mean).Hadamard_division(labels_stddev);
                }                                

                calculate_loss();
                Log::log(LOG_LEVEL_DEBUG, "epoch ", epoch, ", batch ", batch, ", sample ", batch_counter, "/", batch_size, ", average loss across all outputs ", loss_avg);
            }
            // finalize batch
            else {
                Log::log(LOG_LEVEL_INFO, "epoch ", epoch, ", batch ", batch, "FINISHED, average loss (across all outputs) = ", loss_avg, "starting backprop (iteration ", backprop_iterations, ")");
                backpropagate();
                // reset loss_counter and loss_sum at end of mini-batch
                loss_counter = 0;
                layer[layers-1].loss_sum.fill_zeros();
                batch++;
            }
        }
    }
}

// single iteration online training;
// note: always perform batch training first,
// otherwise proper scaling isn't possible
void NeuralNet::fit(const Array<double>& features, const Array<double>& labels){
    if (backprop_iterations==0){return;}

        // scale features and run prediction
    if (scaling_method==ScalingMethod::min_max_normalized){
        predict((features-features_min).Hadamard_division(features_max-features_min),false);
    }
    if (scaling_method==ScalingMethod::mean_normalized){
        predict((features-features_mean).Hadamard_division(features_max-features_min),false);
    }
    if (scaling_method==ScalingMethod::standardized){
        predict((features-features_mean).Hadamard_division(features_stddev),false);
    }

    // assign scaled labels
    if (scaling_method==ScalingMethod::min_max_normalized){
        layer[layers-1].label = (labels-labels_min).Hadamard_division(labels_max-labels_min);
    }
    if (scaling_method==ScalingMethod::min_max_normalized){
        layer[layers-1].label = (labels-labels_mean).Hadamard_division(labels_max-labels_min);
    }
    if (scaling_method==ScalingMethod::min_max_normalized){
        layer[layers-1].label = (labels-labels_mean).Hadamard_division(labels_stddev);
    }

    calculate_loss();
    Log::log(LOG_LEVEL_DEBUG,
        "average loss (across all output neurons): ", loss_avg,
        "; starting backprop (iteration ", backprop_iterations, ")");
    backpropagate();
}

// feedforward, i.e. predict output from new feature input
Array<double> NeuralNet::predict(const Array<double>& features, bool rescale){
    // iterate over layers
    for (int l=1;l<layers;l++){
        switch (layer[l].type){

            case max_pooling_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] =
                            layer[l-1].feature_stack_h[n].pool_max(layer[l].pooling_slider_shape, layer[l].pooling_stride_shape);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.pool_max(layer[l].pooling_slider_shape, layer[l].pooling_stride_shape);
                }
            } break;

            case avg_pooling_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] =
                            layer[l-1].feature_stack_h[n].pool_avg(layer[l].pooling_slider_shape, layer[l].pooling_stride_shape);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }                    
                }
                else {
                    layer[l].h = layer[l-1].h.pool_avg(layer[l].pooling_slider_shape, layer[l].pooling_stride_shape);
                }
            } break;

            case dense_layer: case output_layer: {
                // iterate over elements
                for (int j=0;j<layer[l].neurons;j++){
                    // assign dotproduct of weight matrix * inputs, plus bias
                    std::vector<int> index = layer[l].h.get_index(j);
                    layer[l].h.set(j,layer[l-1].h.dotproduct(layer[l].W_x.get(index)) + layer[l].b[j]);
                }
            } break;

            case lstm_layer: {
                int t = layer[l].timesteps-1;
                // get current inputs x(t)
                layer[l].x_t.pop_first();
                layer[l].x_t.push_back(layer[l-1].h);            
                // shift c_t and h_t by 1 timestep
                layer[l].c_t.pop_first();
                layer[l].h_t.pop_first();
                // calculate gate inputs
                for (int j=0;j<layer[l].neurons;j++){
                    layer[l].f_gate.set(j,layer[l].W_f[j].dotproduct(layer[l].x_t[t]) + layer[l].U_f[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_f[j]);
                    layer[l].i_gate.set(j,layer[l].W_i[j].dotproduct(layer[l].x_t[t]) + layer[l].U_i[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_i[j]);
                    layer[l].o_gate.set(j,layer[l].W_o[j].dotproduct(layer[l].x_t[t]) + layer[l].U_o[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_o[j]);
                    layer[l].c_gate.set(j,layer[l].W_c[j].dotproduct(layer[l].x_t[t]) + layer[l].U_c[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_c[j]);
                }
                // activate gates
                layer[l].f_gate = layer[l].f_gate.activation(ActFunc::SIGMOID);
                layer[l].f_gate = layer[l].i_gate.activation(ActFunc::SIGMOID);
                layer[l].f_gate = layer[l].o_gate.activation(ActFunc::SIGMOID);
                layer[l].f_gate = layer[l].c_gate.activation(ActFunc::TANH);
                // add c_t and h_t results for current timestep
                layer[l].c_t.push_back(layer[l].f_gate.Hadamard_product(layer[l].c_t[t-1]) + layer[l].i_gate.Hadamard_product(layer[l].c_gate));
                layer[l].h_t.push_back(layer[l].o_gate.Hadamard_product(layer[l].c_t[t].activation(ActFunc::TANH)));
                layer[l].h = layer[l].h_t[t];
                } break;

            case recurrent_layer: {
                int t = layer[l].timesteps-1;
                layer[l].x_t.pop_first();
                layer[l].x_t.push_back(layer[l-1].h);
                layer[l].h_t.pop_first();
                layer[l].h_t.grow(1);
                for (int j=0;j<layer[l].neurons;j++){
                    // set h(t) = x(t)*W_x + h(t-1)*U + bias
                    layer[l].h_t[t][j] = layer[l].x_t[t].dotproduct(layer[l].W_x[j]) + layer[l].h_t[t-1].dotproduct(layer[l].U[j]) + layer[l].b[j];
                }
                layer[l].h = layer[l].h_t[t];
            } break;
                
            case convolutional_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h = layer[l-1].feature_stack_h.convolution(layer[l].filter_stack[n]);
                    }
                }
                else {
                    layer[l].feature_stack_x = layer[l-1].h.dissect(layer[l-1].dimensions-1);
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h = layer[l].feature_stack_x.convolution(layer[l].filter_stack[n]);
                    }
                }
            } break;
                
            case GRU_layer: {
                int t = layer[l].timesteps-1;
                // get current inputs x(t)
                layer[l].x_t.pop_first();
                layer[l].x_t.push_back(layer[l-1].h);            
                // shift c_t and h_t by 1 timestep
                layer[l].h_t.pop_first();
                // calculate gate inputs
                for (int j=0; j<layer[l].neurons; j++){
                    layer[l].z_gate.set(j,layer[l].W_z[j].dotproduct(layer[l].x_t[t]) + layer[l].U_z[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_z[j]);
                    layer[l].r_gate.set(j,layer[l].W_r[j].dotproduct(layer[l].x_t[t]) + layer[l].U_r[j].dotproduct(layer[l].h_t[t-1]) + layer[l].b_r[j]);
                    layer[l].c_gate.set(j,layer[l].W_c[j].dotproduct(layer[l].x_t[t]) + layer[l].U_c[j].dotproduct(layer[l].r_gate.Hadamard_product(layer[l].h_t[t-1])) + layer[l].b_c[j]);                    
                }
                // activate gates
                layer[l].z_gate = layer[l].z_gate.activation(ActFunc::SIGMOID);
                layer[l].r_gate = layer[l].r_gate.activation(ActFunc::SIGMOID);
                layer[l].c_gate = layer[l].c_gate.activation(ActFunc::TANH);
                layer[l].h_t[t] = ((layer[l].z_gate-1)*-1).Hadamard_product(layer[l].h_t[t-1]) + layer[l].z_gate.Hadamard_product(layer[l].c_gate);
                layer[l].h = layer[l].h_t[t];
            } break;
                
            case dropout_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n];
                        layer[l].feature_stack_h[n].fill_dropout(layer[l].dropout_ratio);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h;
                    layer[l].h.fill_dropout(layer[l].dropout_ratio);
                }
            } break;
                
            case ReLU_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n].activation(ActFunc::RELU);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.activation(ActFunc::RELU);
                }
            } break;
                
            case lReLU_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n].activation(ActFunc::LRELU);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.activation(ActFunc::LRELU);
                }
            } break;
                
            case ELU_layer: {
                                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n].activation(ActFunc::ELU);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.activation(ActFunc::ELU);
                }
            } break;
                
            case sigmoid_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n].activation(ActFunc::SIGMOID);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.activation(ActFunc::SIGMOID);
                }
            } break;
                
            case tanh_layer: {
                if (layer[l-1].stacked){
                    for (int n=0; n<feature_maps; n++){
                        layer[l].feature_stack_h[n] = layer[l-1].feature_stack_h[n].activation(ActFunc::TANH);
                    }
                    if (!layer[l+1].stacked){
                        layer[l].h = layer[l].feature_stack_h.stack();
                    }
                }
                else {
                    layer[l].h = layer[l-1].h.activation(ActFunc::TANH);
                }
            } break;
                
            case flatten_layer: {
                layer[l].h = layer[l-1].h.flatten();
            } break;
                
            default: {
                layer[l].h = layer[l-1].h;
            } break;
        }        
    }
    // return output
    if (!rescale || scaling_method==ScalingMethod::none){
        return layer[layers-1].h;
    }
    else {
        // rescale
        if (scaling_method == ScalingMethod::min_max_normalized){
            return layer[layers-1].h.Hadamard_product(labels_max-labels_min)+labels_min;
        }
        if (scaling_method == ScalingMethod::mean_normalized){
            return layer[layers-1].h.Hadamard_product(labels_max-labels_min)+labels_mean;
        }
        if (scaling_method == ScalingMethod::standardized){
            return layer[layers-1].h.Hadamard_product(labels_stddev)+labels_mean;
        }
    }
}

// save the model to a file
void NeuralNet::save(){
    // TODO
}

// load the model from a file
void NeuralNet::load(){
    // TODO
}

// prints a summary of the model architecture
void NeuralNet::summary(){
    // TODO
}   

// performs a single iteration of backpropagation
void NeuralNet::backpropagate(){
    // - delta rule for output neurons: delta w_ij= lr * loss_func_derivative * act'(net_inp_j) * out_i
    // - delta rule for hidden neurons: delta w_ij=lr * SUM_k[err_k*w_jk] * act'(net_inp_j) * out_i
    // - general rule: delta w_ij= lr * partial_error_k * layer_function_drv * out_i     
    backprop_iterations++;
    // get average gradients for this batch
    layer[layers-1].gradient/=batch_counter;
    // reset batch counter
    batch_counter = 0;
    // update weights and push partial derivative to preceding layer
    for (int l=layers-1; l>0; l--){
        switch (layer[l].type){

            case max_pooling_layer: {
                if (layer[l-1].stacked){
                    layer[l].gradient_stack = layer[l].gradient.dissect(layer[l].dimensions-1);
                    // initialize variables
                    std::vector<int> index_i(layer[l-1].dimensions);
                    std::vector<int> stride_shape_vec = initlist_to_vector(layer[l].pooling_stride_shape);
                    std::vector<int> combined_index(layer[l-1].dimensions);  
                    Array<int> slider_box(layer[l].pooling_slider_shape);                  
                    for (int map=0; map<feature_maps; map++){
                        layer[l-1].gradient_stack[map].fill_zeros();
                        // iterate over pooled elements
                        for (int j=0;j<layer[l].neurons;j++){
                            // get associated index
                            std::vector<int> index_j = layer[l].feature_stack_h.get_index(j);
                            // get corresponding source index
                            for (int n=0; n<layer[l-1].dimensions; n++){
                                index_i[n] = index_j[n] * stride_shape_vec[n];
                            }
                            for (int s=0; s<slider_box.get_elements(); s++) {
                                std::vector<int> box_element_index = slider_box.get_index(s);
                                for (int n=0; n<layer[l-1].dimensions;n++){
                                    combined_index[n] = index_i[n] + box_element_index[n];
                                }
                                if (layer[l-1].feature_stack_h[map].get(combined_index) == layer[l].feature_stack_h[map][j]){
                                    layer[l-1].gradient_stack[map].set(combined_index,layer[l].gradient_stack[map][j]);
                                }
                            }
                        }                        
                    }
                    layer[l-1].gradient=layer[l-1].gradient_stack.stack();
                }
                else {
                    layer[l-1].gradient.fill_zeros();
                    // initialize variables
                    std::vector<int> index_i(layer[l-1].dimensions);
                    std::vector<int> stride_shape_vec = initlist_to_vector(layer[l].pooling_stride_shape);
                    std::vector<int> combined_index(layer[l-1].dimensions);
                    Array<int> slider_box(layer[l].pooling_slider_shape);                    
                    // iterate over pooled elements
                    for (int j=0;j<layer[l].neurons;j++){
                        // get associated index
                        std::vector<int> index_j = layer[l].h.get_index(j);
                        // get corresponding source index
                        for (int n=0; n<layer[l-1].dimensions; n++){
                            index_i[n] = index_j[n] * stride_shape_vec[n];
                        }
                        for (int s=0; s<slider_box.get_elements(); s++) {
                            std::vector<int> box_element_index = slider_box.get_index(s);
                            for (int n=0; n<layer[l-1].dimensions;n++){
                                combined_index[n] = index_i[n] + box_element_index[n];
                            }
                            if (layer[l-1].h.get(combined_index) == layer[l].h[j]){
                                layer[l-1].gradient.set(combined_index,layer[l].gradient[j]);
                            }
                        }
                    }
                }
            } break;

            case avg_pooling_layer: {
                if (layer[l-1].stacked){
                    layer[l].gradient_stack = layer[l].gradient.dissect(layer[l].dimensions-1);
                    // initialize variables
                    Array<int> slider_box(layer[l].pooling_slider_shape);
                    int slider_box_elements = slider_box.get_elements();
                    std::vector<int> index_i(layer[l-1].dimensions);
                    std::vector<int> stride_shape_vec = initlist_to_vector(layer[l].pooling_stride_shape);
                    std::vector<int> combined_index(layer[l-1].dimensions);  
                    // iterate over feature maps                  
                    for (int map=0; map<feature_maps; map++){
                        layer[l-1].gradient_stack[map].fill_zeros();
                        // iterate over pooled elements
                        for (int j=0;j<layer[l].neurons;j++){
                            // get associated index
                            std::vector<int> index_j = layer[l].feature_stack_h.get_index(j);
                            // get corresponding source index
                            for (int n=0; n<layer[l-1].dimensions; n++){
                                index_i[n] = index_j[n] * stride_shape_vec[n];
                            }
                            for (int s=0; s<slider_box.get_elements(); s++) {
                                std::vector<int> box_element_index = slider_box.get_index(s);
                                for (int n=0; n<layer[l-1].dimensions;n++){
                                    combined_index[n] = index_i[n] + box_element_index[n];
                                }
                                layer[l-1].gradient_stack[map].set(combined_index,
                                    layer[l-1].gradient_stack[map].get(combined_index) + layer[l].gradient_stack[map][j]);
                            }
                        }
                        layer[l-1].gradient_stack[map] /= slider_box_elements;
                    }
                    layer[l-1].gradient=layer[l-1].gradient_stack.stack();
                }
                else {
                    layer[l-1].gradient.fill_zeros();
                    // initialize variables
                    std::vector<int> index_i(layer[l-1].dimensions);
                    std::vector<int> stride_shape_vec = initlist_to_vector(layer[l].pooling_stride_shape);
                    std::vector<int> combined_index(layer[l-1].dimensions);
                    Array<int> slider_box(layer[l].pooling_slider_shape);
                    int slider_box_elements = slider_box.get_elements();
                    // iterate over pooled elements
                    for (int j=0;j<layer[l].neurons;j++){
                        // get associated index
                        std::vector<int> index_j = layer[l].h.get_index(j);
                        // get corresponding source index
                        for (int n=0; n<layer[l-1].dimensions; n++){
                            index_i[n] = index_j[n] * stride_shape_vec[n];
                        }
                        for (int s=0; s<slider_box.get_elements(); s++) {
                            std::vector<int> box_element_index = slider_box.get_index(s);
                            for (int n=0; n<layer[l-1].dimensions;n++){
                                combined_index[n] = index_i[n] + box_element_index[n];
                            }
                            layer[l-1].gradient.set(combined_index,
                                layer[l-1].gradient.get(combined_index) + layer[l].gradient[j]);
                        }
                    }
                    layer[l-1].gradient /= slider_box_elements;
                }
            } break;

            case output_layer:
            case dense_layer: {
                // push gradients to preceding layer
                layer[l-1].gradient.fill_zeros();
                for (int i=0; i<layer[l-1].h.get_elements(); i++){
                    layer[l-1].gradient[i] = *layer[l-1].W_out[i].dotproduct(layer[l].h);
                }
                if (layer[l-1].stacked){
                    layer[l-1].gradient_stack = layer[l-1].gradient.dissect(layer[l-1].dimensions-1);
                }
                // update weights
                switch (opt_method){
                    case MOMENTUM: {
                        for (int j=0; j<layer[l].neurons; j++){
                            // general rule: delta w_ij= out_i * partial_error_k * layer_function_drv * lr
                            layer[l].delta_W_x[j] = (layer[l].delta_W_x[j] * momentum) + 
                                                    (layer[l-1].h * layer[l].gradient[j] * lr * (1-momentum));
                            layer[l].W_x[j] -= layer[l].delta_W_x[j];
                        }
                    } break; 

                    case NESTEROV: {
                        for (int j=0; j<layer[l].neurons; j++){
                            // lookahead step
                            Array<double> lookahead(layer[l].h.get_shape());
                            lookahead = (layer[l].delta_W_x[j] * momentum) +
                                        (layer[l-1].h * layer[l].gradient[j] * lr * (1-momentum));
                            // momentum step
                            layer[l].delta_W_x = (lookahead * momentum) +
                                                (layer[l-1].h * layer[l].gradient[j] * lr * (1-momentum));
                            // update step
                            layer[l].W_x[j] -= layer[l].delta_W_x[j];
                        }
                    } break;

                    case RMSPROP: {
                        for (int j=0; j<layer[l].neurons; j++){
                        // TODO
                        }
                    } break;

                    case ADAM: {
                        for (int j=0; j<layer[l].neurons; j++){
                        // TODO
                        }
                    } break;

                    case ADAGRAD: {
                        for (int j=0; j<layer[l].neurons; j++){
                        // TODO
                        }
                    } break;    

                    case ADADELTA: {
                        for (int j=0; j<layer[l].neurons; j++){
                        // TODO
                        }
                    } break; 

                    case VANILLA:
                    default: {
                        for (int j=0; j<layer[l].neurons; j++){
                            // general rule: delta w_ij= out_i * partial_error_k * layer_function_drv * lr
                            layer[l].delta_W_x[j] = layer[l-1].h * layer[l].gradient[j] * lr;
                            layer[l].W_x[j] -= layer[l].delta_W_x[j];
                        }
                    } break;                                                                                               
                }
            } break;

            case lstm_layer:
                // TODO
                /* STEPS:
                Iterate over the timesteps from last (current) to oldest (first); for each timestep:

                Compute the derivative of the loss function with respect to the output at the current time step:
                dL/dy(t) = the derivative of the loss function with respect to the output y(t) at time step t. This will depend on the specific loss function you are using.

                Compute the derivative of the loss function with respect to the LSTM output gate activation at the current time step:
                dL/do(t) = dL/dy(t) * tanh(c(t)) * sigmoid_derivative(o(t))

                Compute the derivative of the loss function with respect to the LSTM cell state at the current time step:
                dL/dc(t) = dL/dy(t) * o(t) * tanh_derivative(c(t)) + dL/dc(t+1) * f(t+1)

                Compute the derivative of the loss function with respect to the LSTM forget gate activation at the current time step:
                dL/df(t) = dL/dc(t) * c(t-1) * sigmoid_derivative(f(t))

                Compute the derivative of the loss function with respect to the LSTM input gate activation at the current time step:
                dL/di(t) = dL/dc(t) * g(t) * sigmoid_derivative(i(t))

                Compute the derivative of the loss function with respect to the LSTM candidate activation at the current time step:
                dL/dg(t) = dL/dc(t) * i(t) * tanh_derivative(g(t))

                Compute the gradients of the weight matrices and bias vectors for the current time step:
                dL/dW_i = dL/di(t) * x(t)^T
                dL/dW_f = dL/df(t) * x(t)^T
                dL/dW_o = dL/do(t) * x(t)^T
                dL/dW_c = dL/dg(t) * x(t)^T
                dL/dU_i = dL/di(t) * h(t-1)^T
                dL/dU_f = dL/df(t) * h(t-1)^T
                dL/dU_o = dL/do(t) * h(t-1)^T
                dL/dU_c = dL/dg(t) * h(t-1)^T
                dL/db_i = dL/di(t)
                dL/db_f = dL/df(t)
                dL/db_o = dL/do(t)
                dL/db_c = dL/dg(t)

                Update the weights and biases using the gradients and a suitable optimization algorithm, such as stochastic gradient descent (SGD) or Adam.
                
                Propagate the derivative of the loss function with respect to the hidden state h(t-1) and cell state c(t-1) to the previous time step, and repeat steps 1-8 for the previous time step, until you reach the first time step in the sequence.
                */                
                break;
            case recurrent_layer:
                // TODO
                break;
            case convolutional_layer:
                // TODO
                break;
            case GRU_layer:
                // TODO
                break;
            case dropout_layer:
                // TODO
                break;
            case ReLU_layer:
                // TODO
                break;
            case lReLU_layer:
                // TODO
                break;
            case ELU_layer:
                // TODO
                break;
            case sigmoid_layer:
                // TODO
                break;
            case tanh_layer:
                // TODO
                break;
            case flatten_layer:
                // TODO
                break;
            default:
                // do nothing
                break;
        }   
    }
    // reset gradients
    layer[layers-1].gradient.fill_zeros();
}

// calculates the loss according to the specified loss function;
// note: forward pass and label-assignments must be done first!
void NeuralNet::calculate_loss(){
    // get index of output layer
    int l=layers-1;
    batch_counter++;
    loss_counter++;
    switch (loss_function){
        // Mean Squared Error
        case MSE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).pow(2);
            layer[l].gradient += gradient_clipping ?
                                ((layer[l].h - layer[l].label) * 2).min(max_gradient) : 
                                (layer[l].h - layer[l].label) * 2;
        } break;

        // Mean Absolute Error
        case MAE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).abs();
            layer[l].gradient += gradient_clipping ? 
                                ((layer[l].h - layer[l].label).sign()).min(max_gradient) :
                                (layer[l].h - layer[l].label).sign();
        } break;

        // Mean Absolute Percentage Error
        case MAPE: {
            layer[l].loss_sum += (layer[l].label - layer[l].h).Hadamard_division(layer[l].label).abs();
            layer[l].gradient += gradient_clipping ? 
                                ((layer[l].h - layer[l].label).sign().Hadamard_product(layer[l].label).Hadamard_division((layer[l].h - layer[l].label)).abs()).min(max_gradient) :
                                (layer[l].h - layer[l].label).sign().Hadamard_product(layer[l].label).Hadamard_division((layer[l].h - layer[l].label)).abs();
        } break;

        // Mean Squared Logarithmic Error
        case MSLE: {              
            layer[l].loss_sum += ((layer[l].h + 1).log() - (layer[l].label + 1).log()).pow(2);
            layer[l].gradient += gradient_clipping ? 
                                ((((layer[l].h + 1).log() - (layer[l].label + 1).log()).Hadamard_division(layer[l].h + 1)) * 2).min(max_gradient) :
                                (((layer[l].h + 1).log() - (layer[l].label + 1).log()).Hadamard_division(layer[l].h + 1)) * 2;
        } break;

        // Categorical Crossentropy
        case CatCrossEntr: {
            // because the label is one-hot encoded, only the output that is labeled as true gets the update
            int true_output = 0;
            for (int j=0;j<layer[l].neurons;j++){
                if (layer[l].h[j]) {
                    true_output = j;
                }
            }
            layer[l].loss_sum[true_output] -= layer[l].label.Hadamard_product(layer[l].h.log()).sum();
            layer[l].gradient -= gradient_clipping ? 
                                (layer[l].label.Hadamard_division(layer[l].h)).min(max_gradient) :
                                layer[l].label.Hadamard_division(layer[l].h);
        } break;

        // Sparse Categorical Crossentropy
        case SparseCatCrossEntr: {
            layer[l].loss_sum -= layer[l].h.log();
            layer[l].gradient -= gradient_clipping ? 
                                (layer[l].label.Hadamard_division(layer[l].h)).min(max_gradient) : 
                                layer[l].label.Hadamard_division(layer[l].h);
        } break;

        // Binary Crossentropy
        case BinCrossEntr: {
            layer[l].loss_sum -= layer[l].label.Hadamard_product(layer[l].h.log()) + ((layer[l].label*-1)+1) * ((layer[l].h*-1)+1).log();
            layer[l].gradient -= gradient_clipping ? 
                                (layer[l].label.Hadamard_division(layer[l].h) - ((layer[l].label*-1)+1).Hadamard_division((layer[l].h*-1)+1)).min(max_gradient) : 
                                layer[l].label.Hadamard_division(layer[l].h) - ((layer[l].label*-1)+1).Hadamard_division((layer[l].h*-1)+1);
        } break;

        // Kullback-Leibler Divergence
        case KLD: {
            layer[l].loss_sum += layer[l].label.Hadamard_product(layer[l].label.Hadamard_division(layer[l].h).log());
            layer[l].gradient += gradient_clipping ?
                                (layer[l].label.Hadamard_product(layer[l].label.log() - layer[l].h.log() + 1)).min(max_gradient) :
                                layer[l].label.Hadamard_product(layer[l].label.log() - layer[l].h.log() + 1);
        } break;

        // Poisson
        case Poisson: {
            layer[l].loss_sum += layer[l].h - layer[l].label.Hadamard_product(layer[l].h.log());
            layer[l].gradient += gradient_clipping ? 
                                (layer[l].h - layer[l].label).min(max_gradient) : 
                                layer[l].h - layer[l].label;
        } break;

        // Hinge
        case Hinge: {
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1) + 1).max(0);
            layer[l].gradient -= gradient_clipping ? 
                                (layer[l].label.Hadamard_product((layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).sign())).min(max_gradient) :
                                layer[l].label.Hadamard_product((layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).sign());
        } break;

        // Squared Hinge
        case SquaredHinge: {
            layer[l].loss_sum += ((layer[l].label.Hadamard_product(layer[l].h) * -1 + 1).max(0)).pow(2);
            layer[l].gradient += gradient_clipping ? 
                                ((layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).max(0) * 2).min(max_gradient) : 
                                (layer[l].label.Hadamard_product(layer[l].h)*-1 + 1).max(0) * 2;
        } break;

        default: {
            // do nothing
        } break;
    }
    layer[l].loss = layer[l].loss_sum / loss_counter;
    loss_avg = layer[l].loss.sum() / layer[l].neurons;
}

// creates a new input layer or adds a new parallel input shape
// to a preexisting input layer
void NeuralNet::addlayer_input(std::initializer_list<int> shape){
    layer_init(input_layer, shape);  
}

// creates a new output layer or adds a new parallel output shape
// to a preexisting output layer
void NeuralNet::addlayer_output(std::initializer_list<int> shape, LossFunction loss_function){
    layer_init(output_layer, shape);
    int l = layers-1;
    layer[l].label = Array<double>(shape);
    layer[l].loss = Array<double>(shape);
    layer[l].loss_sum = Array<double>(shape);
    loss_function = loss_function;
    layer_make_dense_connections();
}

// creates an LSTM layer of the specified shape
void NeuralNet::addlayer_lstm(std::initializer_list<int> shape, const int timesteps){
    layer_init(lstm_layer, shape);
    int l = layers-1;
    layer[l].timesteps = timesteps;
    layer[l].c_t = Array<Array<double>>(timesteps);
    layer[l].h_t = Array<Array<double>>(timesteps);
    layer[l].x_t = Array<Array<double>>(timesteps);
    for (int t=0;t<timesteps;t++){
        // initialize vectors of timesteps for the cell state and hidden state
        layer[l].c_t[t] = Array<double>(shape); layer[l].c_t[t].fill_zeros();
        layer[l].h_t[t] = Array<double>(shape); layer[l].h_t[t].fill_zeros();
        layer[l].x_t[t] = Array<double>(shape); layer[l].h_t[t].fill_zeros();
    }
    // initialize gate value arrays
    layer[l].f_gate = Array<double>(shape);
    layer[l].i_gate = Array<double>(shape);
    layer[l].o_gate = Array<double>(shape);
    layer[l].c_gate = Array<double>(shape);    
    // initialize gate weights to h(t-1)
    layer[l].U_f = Array<Array<double>>(shape);
    layer[l].U_i = Array<Array<double>>(shape);
    layer[l].U_o = Array<Array<double>>(shape);
    layer[l].U_c = Array<Array<double>>(shape);
    // initialize gate weights to x(t)
    layer[l].W_f = Array<Array<double>>(layer[l-1].h.get_shape());
    layer[l].W_i = Array<Array<double>>(layer[l-1].h.get_shape());
    layer[l].W_o = Array<Array<double>>(layer[l-1].h.get_shape());
    layer[l].W_c = Array<Array<double>>(layer[l-1].h.get_shape());     
    // initialize weight matrices
    for (int j=0;j<layer[l].c_t.get_elements();j++){
        layer[l].U_f[j] = Array<double>(shape); layer[l].U_f[j].fill_He_ReLU(layer[l].neurons);
        layer[l].U_i[j] = Array<double>(shape); layer[l].U_i[j].fill_He_ReLU(layer[l].neurons);
        layer[l].U_o[j] = Array<double>(shape); layer[l].U_o[j].fill_He_ReLU(layer[l].neurons);
        layer[l].U_c[j] = Array<double>(shape); layer[l].U_c[j].fill_He_ReLU(layer[l].neurons);
        layer[l].W_f[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_f[j].fill_He_ReLU(layer[l-1].neurons);
        layer[l].W_i[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_i[j].fill_He_ReLU(layer[l-1].neurons);
        layer[l].W_o[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_o[j].fill_He_ReLU(layer[l-1].neurons);
        layer[l].W_c[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_c[j].fill_He_ReLU(layer[l-1].neurons);    
    }
    // initialize biases
    layer[l].b_f = Array<double>(shape); layer[l].b_f.fill_He_ReLU(layer[l].neurons);
    layer[l].b_i = Array<double>(shape); layer[l].b_f.fill_He_ReLU(layer[l].neurons);
    layer[l].b_o = Array<double>(shape); layer[l].b_f.fill_He_ReLU(layer[l].neurons);
    layer[l].b_c = Array<double>(shape); layer[l].b_f.fill_He_ReLU(layer[l].neurons);
}

// creates a GRU layer
void NeuralNet::addlayer_GRU(std::initializer_list<int> shape, const int timesteps){
    layer_init(GRU_layer, shape);
    int l = layers-1;
    layer[l].timesteps = timesteps;
    layer[l].c_t = Array<Array<double>>(timesteps);
    layer[l].h_t = Array<Array<double>>(timesteps);
    layer[l].x_t = Array<Array<double>>(timesteps);
    for (int t=0;t<timesteps;t++){
        // initialize vectors of timesteps for the cell state and hidden state
        layer[l].c_t[t] = Array<double>(shape); layer[l].c_t[t].fill_zeros();
        layer[l].h_t[t] = Array<double>(shape); layer[l].h_t[t].fill_zeros();
        layer[l].x_t[t] = Array<double>(shape); layer[l].h_t[t].fill_zeros();
    }
    // initialize gate value arrays  
    layer[l].z_gate = Array<double>(shape);
    layer[l].r_gate = Array<double>(shape);
    layer[l].c_gate = Array<double>(shape);
    // initialize gate weights to h(t-1) 
    layer[l].U_z = Array<Array<double>>(shape);
    layer[l].U_r = Array<Array<double>>(shape);
    layer[l].U_c = Array<Array<double>>(shape);     
    // initialize gate weights to x(t)
    layer[l].W_z = Array<Array<double>>(layer[l-1].h.get_shape());
    layer[l].W_r = Array<Array<double>>(layer[l-1].h.get_shape());
    layer[l].W_c = Array<Array<double>>(layer[l-1].h.get_shape());
    // initialize weight matrices
    for (int j=0;j<layer[l].neurons;j++){
        layer[l].U_z[j] = Array<double>(shape); layer[l].U_z[j].fill_He_ReLU(layer[l].neurons);
        layer[l].U_r[j] = Array<double>(shape); layer[l].U_r[j].fill_He_ReLU(layer[l].neurons);
        layer[l].U_c[j] = Array<double>(shape); layer[l].U_c[j].fill_He_ReLU(layer[l].neurons);
        layer[l].W_z[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_z[j].fill_He_ReLU(layer[l-1].neurons);
        layer[l].W_r[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_r[j].fill_He_ReLU(layer[l-1].neurons);
        layer[l].W_c[j] = Array<double>(layer[l-1].h.get_shape()); layer[l].W_c[j].fill_He_ReLU(layer[l-1].neurons);
    }
    // initialize biases
    layer[l].b_z = Array<double>(shape); layer[l].b_z.fill_He_ReLU(layer[l].neurons);
    layer[l].b_r = Array<double>(shape); layer[l].b_r.fill_He_ReLU(layer[l].neurons);
    layer[l].b_c = Array<double>(shape); layer[l].b_c.fill_He_ReLU(layer[l].neurons);   
}

// creates a recurrent layer of the specified shape
void NeuralNet::addlayer_recurrent(std::initializer_list<int> shape, int timesteps){
    layer_init(recurrent_layer, shape);
    int l = layers-1;
    layer[l].timesteps = timesteps;
    layer[l].x_t = Array<Array<double>>(timesteps);
    layer[l].h_t = Array<Array<double>>(timesteps);
    layer[l].U = Array<Array<double>>(shape);
    for (int j=0; j<layer[l].neurons; j++){
        layer[l].U[j] = Array<double>(shape); layer[l].U[j].fill_He_ReLU(layer[l].neurons);
    }
    layer_make_dense_connections();
}

// creates a fully connected layer
void NeuralNet::addlayer_dense(std::initializer_list<int> shape){
    layer_init(dense_layer, shape);
    layer_make_dense_connections();
}

// creates a convolutional layer
void NeuralNet::addlayer_convolutional(const int filter_radius, bool padding){
    // check valid layer type
    if (layers==0){
        Log::log(LOG_LEVEL_WARNING,
            "invalid usage of method 'void NeuralNet::addlayer_convolutional(const int filter_radius, bool padding)'",
            "the first layer always has to be of type 'input_layer'");
        return;
    }
    // initialize filters
    if (filter_radius<1){
        Log::log(LogLevel::LOG_LEVEL_INFO,
            "the filter radius for CNN layers should be >=1 but is ", filter_radius,
            ", -> will be set to 1");
    }
    std::vector<int> filter_shape(layer[layers-1].dimensions);
    for (int d=0;d<layer[layers-1].dimensions;d++){
        filter_shape[d] = d<=1 ? 1+2*std::max(1,filter_radius) : layer[layers-1].h.get_shape()[d];
    }
    // initialize layer
    if (layer[layers-1].stacked){
        layer_init(convolutional_layer, vector_to_initlist(layer[layers-1].feature_stack_h.get_convolution_shape(filter_shape, padding)));
    }
    else {
        layer_init(convolutional_layer, vector_to_initlist(layer[layers-1].h.get_convolution_shape(filter_shape, padding)));
    }
    layer[layers-1].filter_shape = filter_shape;
}

// creates a dropout layer
void NeuralNet::addlayer_dropout(const double ratio){
    int l = layers - 1;
    layer_init(dropout_layer, layer[l-1].shape);
    layer[l].dropout_ratio = ratio;
}

void NeuralNet::addlayer_sigmoid(){
    int l = layers - 1;
    layer_init(sigmoid_layer, layer[l-1].shape);
}

void NeuralNet::addlayer_ReLU(){
    int l = layers - 1;
    layer_init(ReLU_layer, layer[l-1].shape);;   
}

void NeuralNet::addlayer_lReLU(){
    int l = layers - 1;
    layer_init(lReLU_layer, layer[l-1].shape);  
}

void NeuralNet::addlayer_ELU(){
    int l = layers - 1;
    layer_init(ELU_layer, layer[l-1].shape);   
}

void NeuralNet::addlayer_tanh(){
    int l = layers - 1;
    layer_init(tanh_layer, layer[l-1].shape);   
}

void NeuralNet::addlayer_flatten(){
    std::initializer_list<int> shape = {layer[layers-1].neurons};
    layer_init(flatten_layer, shape);
}

void NeuralNet::addlayer_pool_avg(std::initializer_list<int> slider_shape, std::initializer_list<int> stride_shape){
    addlayer_pool(avg_pooling_layer, slider_shape, stride_shape);   
}

void NeuralNet::addlayer_pool_max(std::initializer_list<int> slider_shape, std::initializer_list<int> stride_shape){
    addlayer_pool(max_pooling_layer, slider_shape, stride_shape);
}

void NeuralNet::addlayer_pool(LayerType type, std::initializer_list<int> slider_shape, std::initializer_list<int> stride_shape){
    std::vector<int> layer_shape(layer[layers-1].dimensions);
    std::vector<int> slider_shape_vec = initlist_to_vector(slider_shape);
    std::vector<int> stride_shape_vec = initlist_to_vector(stride_shape);
    int l=layers;
    if (layer[l-1].stacked)
    {
        for (int d=0;d<layer[layers-1].dimensions;d++){
            layer_shape[d] = (layer[layers-1].feature_stack_h[0].get_shape()[d] - slider_shape_vec[d]) / stride_shape_vec[d];
        }
    }
    else {
        for (int d=0;d<layer[layers-1].dimensions;d++){
            layer_shape[d] = (layer[l-1].h.get_shape()[d] - slider_shape_vec[d]) / stride_shape_vec[d];
        }
    }    
    layer_init(type, vector_to_initlist(layer_shape));
    layer[l].pooling_slider_shape=slider_shape;
    layer[l].pooling_stride_shape=stride_shape;
}

// helper method to convert a std::vector<int> to std::initializer_list<int>
std::initializer_list<int> NeuralNet::vector_to_initlist(const std::vector<int>& vec) {
    std::initializer_list<int> init_list;
    for (auto& elem : vec) {
        init_list = {std::initializer_list<int>{elem}};
    }
}

// helper method to convert a std::initializer_list<int> to std::vector<int>
std::vector<int> initlist_to_vector(const std::initializer_list<int>& list){
    std::vector<int> vector(list.size());
    auto iterator = list.begin();
    for (int n=0;iterator!=list.end();n++, iterator++){
        vector[n] = *iterator;
    }
    return vector;
}

// helper method to make dense connections from
// preceding layer (l-1) to current layer (l)
// and initialize dense connection weights and biases
void NeuralNet::layer_make_dense_connections(){
    int l = layers-1;
    // initialize 'h' Array of preceding stacked layer
    if (layer[l-1].stacked){
        layer[l-1].h = Array<double>(layer[l-1].feature_stack_h.get_stacked_shape());
    }     
    // create incoming weights
    layer[l].W_x = Array<Array<double>>(layer[l].shape);
    int neurons_i = layer[l-1].h.get_elements();
    int neurons_j = layer[l].neurons;
    for (int j=0;j<neurons_j;j++){
        layer[l].W_x[j] = Array<double>(layer[l-1].h.get_shape());
        layer[l].W_x[j].fill_He_ReLU(neurons_i);
        layer[l].delta_W_x[j] = Array<double>(layer[l-1].h.get_shape());
        layer[l].delta_W_x[j].fill_zeros();
    }
    // attach outgoing weights of preceding layer
    layer[l-1].W_out = Array<Array<int>>(layer[l-1].h.get_shape());
    for (int i=0;i<neurons_i;i++){
        layer[l-1].W_out[i] = Array<int>(layer[l].shape);
        for (int j=0;j<neurons_j;j++){
            // store references to associated weights into <double *>
            layer[l-1].W_out[i][j] = &layer[l].W_x[j][i];
            // example of accessing the 'fan out' of weight values for any given neuron 'i' by dereferencing:
            // Array<double> dereferenced = *layer[l-1].W_out[i];
        }
    }
    // initialize bias weights
    layer[l].b = Array<double>(layer[l].shape);
    layer[l].b.fill_He_ReLU(neurons_i);
}

// basic layer setup
void NeuralNet::layer_init(LayerType type, std::initializer_list<int> shape){
    // check valid layer type
    if (layers==0 && type != input_layer){
        throw std::invalid_argument("the first layer always has to be of type 'input_layer'");
    }
    if (layers>0 && type == input_layer){
        throw std::invalid_argument("input layer already exists");
    }    
    if (layer[layers-1].type == output_layer){
        throw std::invalid_argument("an output layer has already been defined; can't add any new layers on top");
    }
    if (static_cast<int>(type) < 0 || static_cast<int>(type) >= static_cast<int>(LayerType::LAYER_TYPE_COUNT)){
        throw std::invalid_argument("layer type enum value must be a cardinal that's less than LAYER_TYPE_COUNT");
    }
    // create new layer
    layer.emplace_back(Layer());
    // setup layer parameters
    layers++;
    int l = layers - 1;
    layer[l].type = type;
    layer[l].shape = shape;
    // set 'stacked' parameter
    if (type==input_layer || type==dense_layer || type==output_layer || type==lstm_layer ||
        type==recurrent_layer || type==GRU_layer){
        layer[l].stacked=false;
    }
    else if (type==convolutional_layer){
        layer[l].stacked=true;
    }
    else {
        layer[l].stacked=layer[l-1].stacked;
    }
    // initialize 'x', 'h' and 'gradient' arrays
    if (layer[l].stacked){
        layer[l].feature_stack_h = Array<Array<double>>(feature_maps);
        layer[l].gradient_stack = Array<Array<double>>(feature_maps);
        if (!layer[l-1].stacked){
            layer[l].feature_stack_x = Array<Array<double>>(feature_maps);
        }
        for (int i=0; i<feature_maps; i++){
            layer[l].feature_stack_h[i] = Array<double>(shape);
            layer[l].gradient_stack[i] = Array<double>(shape);
            layer[l].gradient_stack[i].fill_zeros();
            if (!layer[l-1].stacked){
                layer[l].feature_stack_x[i] = Array<double>(layer[l-1].h.get_shape());
            }
        }
        layer[l].neurons = layer[l].feature_stack_h.get_elements();
    }
    else {
        layer[l].h = Array<double>(shape);
        layer[l].gradient = Array<double>(shape);
        layer[l].gradient.fill_zeros();
        layer[l].neurons = layer[l].h.get_elements();
    }
    layer[l].dimensions = shape.size();    
}