## Neural Network (type: Multilayer Perceptron)
## `class MLP`
- a class for fully connected neural networks
- full freedom over the number of layers and neurons per layer
- automated scaling of inputs ("features"), outputs and labels ("targets")

Usage:
- `#include <mlp.h>`or include as part of `datascience.h`
  (make sure to update the compiler's include path or use relative paths instead)
- create a new instance of MLP, e.g.
```
MLP network;
```

Methods:
| method | description |
|--------|-------------|
| `MLP::add_layer(int neurons, OPTIMIZATION_METHOD _opt_method, ACTIVATION_FUNC _activation)`| adds a new layer to an MLP network object, with the specified number of neurons, [optimization method](#optimizers) and [activation function](../../general/docs/activation_functions.md) |
| `MLP::reset_weights(uint start_layer, uint end_layer, double factor)` | re-initializes the weights (the method is automatically optimized for the number of neurons ("fan in", "fan out") and activation function) |
| `MLP::feedforward(uint start_layer, uint end_layer)` | |
| `MLP::backpropagate()` | |
| `MLP::set_input(uint index, double value)` | |
| `MLP::get_input(uint index, uint l)` | |
| `MLP::get_output(uint index)` | |
| `MLP::get_hidden(uint index,uint layer_index)` | |
| `MLP::set_label(uint index, double value)` | |
| `MLP::autoencode()` | |
| `MLP::save(std::string filename)` | |
| `MLP::network_exists()` | |
| `MLP::get_loss_avg()` | |
| `MLP::set_training_mode(bool active)` | |
| `MLP::set_learning_rate(double value)` | |
| `MLP::set_learning_rate_decay(double value)` | |
| `MLP::set_learning_momentum(double value)` | |
| `MLP::set_learning_rate_auto(bool active=true)` | |
| `MLP::set_scaling_method(SCALING method=normalized)` | |
| `MLP::set_dropout(double value)` | |
| `MLP::set_recurrent(bool confirm)` | |
| `MLP::set_gradient_clipping(bool active,double threshold)` | |
| `MLP::get_avg_h()` | |
| `MLP::get_avg_output()` | |     
| `MLP::get_lr()` | |
___
### Optimizers
These are various algorithms that are designed to make the learning process (weight correction during backpropagation) more efficient and robust. The available optimizers are:
- Vanilla (optional: momentum, auto learning-rate)
- Nesterov
- ADAM
- AdaGrad
- ADADELTA
- RMSprop