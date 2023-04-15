## Neural Network (type: Multilayer Perceptron)
## `class MLP`
- full flexibility for layers and neurons
- optimizers: Vanilla, Nesterov, ADAM, ADADELTA, Adagrad, RMSprop
- input & label scaling

Methods:
| method | description |
|--------|-------------|
| `MLP::add_layer(int neurons, OPTIMIZATION_METHOD _opt_method, ACTIVATION_FUNC _activation)`| 
| `MLP::reset_weights(uint start_layer, uint end_layer, double factor) | |
| `MLP::feedforward(uint start_layer, uint end_layer) | |
| `MLP::backpropagate() | |
| `MLP::set_input(uint index, double value) | |
| `MLP::get_input(uint index, uint l) | |
| `MLP::get_output(uint index) | |
| `MLP::get_hidden(uint index,uint layer_index) | |
| `MLP::set_label(uint index, double value) | |
| `MLP::autoencode() | |
| `MLP::save(std::string filename) | |
| `MLP::network_exists() | |
| `MLP::get_loss_avg() | |
| `MLP::set_training_mode(bool active) | |
| `MLP::set_learning_rate(double value) | |
| `MLP::set_learning_rate_decay(double value) | |
| `MLP::set_learning_momentum(double value) | |
| `MLP::set_learning_rate_auto(bool active=true) | |
| `MLP::set_scaling_method(SCALING method=normalized) | |
| `MLP::set_dropout(double value) | |
| `MLP::set_recurrent(bool confirm) | |
| `MLP::set_gradient_clipping(bool active,double threshold) | |
| `MLP::get_avg_h() | |
| `MLP::get_avg_output() | |     
| `MLP::get_lr() | |