author: cyberchriz (Christian Suer); language: C++
___
# git/DataScience: GENERAL OVERVIEW
This repository provides functionality for
- neural networks of various topologies (multilayer perceptron, RNN, autoencoder)
- sample analysis (regression, correlation, cointegration, stationary transformation, mean, median, histogram ...)
- probalistic distributions
___
# git/DataScience/distributions/headers/cumulative_distribution_functions.h

A cumulative distribution function returns the probability of a random variable taking on values less than or equal to a given value.
Classname: CdfObject, alias: cdf
Available methods:
```cpp
	- cdf<T>::gaussian(T x_val,T mu=0,T sigma=1);
	- cdf<T>::cauchy(T x_val,T x_peak=0,T gamma=1);
	- cdf<T>::laplace(T x_val,T mu=0,T sigma=1);
	- cdf<T>::pareto(T x_val,T alpha=1,T tail_index=1);
	- cdf<T>::lomax(T x_val,T alpha=1,T tail_index=1);
```
usage example:
```cpp
double x = 0.3;
double probability = cdf<double>::gaussian(x);
```
___
# git/DataScience/distributions/headers/probability_density_functions.h

A probability density function (PDF) is a statistical function that gives the probability
of a random variable taking on any particular value. It is the derivative of the cumulative distribution function.
Classname: PdfObject, alias: pdf
Available methods:
```cpp
	pdf<T>::gaussian(T x_val,T mu=0,T sigma=1);
	pdf<T>::cauchy(T x_val,T x_peak=0,T gamma=1);
	pdf<T>::laplace(T x_val,T mu=0,T sigma=1);
	pdf<T>::pareto(T x_val,T alpha=1,T tail_index=1);
	pdf<T>::lomax(T x_val,T alpha=1,T tail_index=1);
```
usage example:
```cpp
double x = 0.3;
double probability = pdf<double>::gaussian(x);
```
___
# git/DataScience/distributions/headers/random_distributions.h
Returns a random value from a specified distribution.
Classname: Random, alias: rnd
Available methods:
```cpp
	Random<T>::gaussian(T mu=0,T sigma=1);
	Random<T>::cauchy(T x_peak=0,T gamma=1);
	Random<T>::uniform(T min=0, T max=1);
	Random<T>::laplace(T mu=0,T sigma=1);
	Random<T>::pareto(T alpha=1,T tail_index=1);
	Random<T>::lomax(T alpha=1,T tail_index=1);
	Random<T>::sign();
	Random<T>::binary();
```
usage examples:
```cpp
double x = rnd<double>::gaussian(); // returns a random <double> value from a gaussian normal distribution with default µ=0 and sigma=1;
float y = rnd<float>::laplace();
double z = rnd<double>::uniform(0,10); // returns a random <double> from a uniform distribution between 0-10
char s = rnd<char>::sign(); // randomly returns either -1 or 1
bool b = rnd<bool>binary; // randomly returns either true or false
```
___
# git/DataScience/neuralnet/MLP/headers/mlp.h
Class for neural networks (multilayer perceptron, fully connected) with flexible topology.
The class allows multiple different activation functions, optimizers and options for automatic scaling of input features and labels/outputs.
The neurons can be recurrent.

usage example:
```cpp
#include <iostream>
#include <cmath>
#include "../headers/mlp.h"
#include "../../autoencoder.h"

int main(){
    // set up a test network
    MLP network = MLP("");
    
    // hyperparameters
		network.set_learning_rate(0.002);
		network.set_learning_momentum(0.9);
    network.set_learning_rate_auto(false);
		network.set_scaling_method(normalized);
    network.set_recurrent(false);
    network.set_dropout(0.0);
    network.set_training_mode(true);
    ACTIVATION_FUNC act_func=f_tanh;
    OPTIMIZATION_METHOD opt=Nesterov;
      
    // topology
    int input_neurons=5;
    int hidden_neurons=5;
    network.add_layer(input_neurons,opt,act_func);
    network.add_layer(hidden_neurons,opt,act_func);
		network.add_layer(hidden_neurons,opt,act_func);
    network.add_layer(input_neurons,opt,act_func);

    // test iterations
    for (int i=0;i<=1000000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            network.set_input(r,((double)rand())/RAND_MAX);
        }
        network.feedforward();
        network.autoencode();
        network.backpropagate();
        if (i<30 || i%10000==0){
            std::cout << "iteration: " << i << ", loss: " << network.get_loss_avg() << " ========================================================================================================\n";
            std::cout << "inputs:    " << network.get_input(0) << " | " << network.get_input(1) << " | " << network.get_input(2) << " | " << network.get_input(3) << " | " << network.get_input(4) << "\n";
            std::cout << "outputs:   " << network.get_output(0) << " | " << network.get_output(1) << " | " << network.get_output(2) << " | " << network.get_output(3) << " | " << network.get_output(4) << "\n";
        }
    }
    std::cout << "[...done]\n\n\n";
```
___
# git/DataScience/neuralnet/autoencoder.h
This is a child class of the Multilayer Perceptron (mlp.h) neural network class.
An autoencoder is a special type of neural network that first encodes from the input to a lower number of neurons ("dimensionalty reduction", "bottleneck layer"), then decodes back to a number of outputs that exactly matches the number of input. The network takes the inputs as targets (=labels) and propagates the errors of the predictions back in order to adjust the network's weights via stochastic gradient descent.
Classname: Autoencoder

Available methods:
```cpp
        double get_encoded(uint index); // returns the hidden state of a bottleneck neuron (with a given index)
        double get_decoded(uint index); // returns the output a neuron (with a given index) from the decoder part (=output layer)
        void set_encoded(uint index,double value); // fills a bottleneck neuron (with given index) with the specified value;
        void decode(); // decodes the bottleneck encoding (=feedforward from bottleneck layer to output layer)
        void encode(); // feedforward from input layer to bottleneck layer
        void sweep(); // feedforward entire network -> set inputs as labels -> backpropagate
        // parametric constructor
        Autoencoder(uint inputs, uint bottleneck_neurons, u_char encoder_hidden_layers=1, u_char decoder_hidden_layers=1, ACTIVATION_FUNC act_func=f_oblique_sigmoid, bool recurrent=true, double dropout=0)
```
usage example:
```cpp
#include <iostream>
#include <cmath>
#include "../headers/mlp.h"
#include "../../autoencoder.h"

int main(){
    // set up a test network
    Autoencoder ae = Autoencoder(5,5,1,1,f_LReLU,false,0);
 
    // test iterations 
    for (int i=0;i<=10000000; i++){
        // fill inputs with random numbers
        for (int r=0;r<5;r++){
            ae.set_input(r,((double)rand())/RAND_MAX);
        }
        ae.sweep();
        if (i<30 || i%100000==0){
            std::cout << "iteration: " << i << ", loss: " << ae.get_loss_avg() << " ========================================================================================================\n";
            std::cout << "inputs:    " << ae.get_input(0) << " | " << ae.get_input(1) << " | " << ae.get_input(2) << " | " << ae.get_input(3) << " | " << ae.get_input(4) << "\n";
            std::cout << "outputs:   " << ae.get_decoded(0) << " | " << ae.get_decoded(1) << " | " << ae.get_decoded(2) << " | " << ae.get_decoded(3) << " | " << ae.get_decoded(4) << "\n";
        }
    }
    std::cout << "[...done]\n\n\n"; 
}
```
___
# git/DataScience/neuralnet/activation_functions.h
This is a helper file containing activation functions for neural networks.

Available functions:
```cpp
// enumeration of available activation functions for neural networks
enum ACTIVATION_FUNC {
   f_ident,        // identity function
   f_sigmoid,      // sigmoid (logistic)
   f_ELU,          // exponential linear unit (ELU)
   f_ReLU,         // rectified linear unit (ReLU)
   f_LReLU,        // leaky ReLU
   f_tanh,         // hyperbolic tangent (tanh)
   f_oblique_tanh, // oblique tanh (custom)
   f_tanh_rectifier,// tanh rectifier
   f_arctan,       // arcus tangent (arctan)
   f_arsinh,       // area sin. hyperbolicus (inv. hyperbol. sine)
   f_softsign,     // softsign (Elliot)
   f_ISRU,         // inverse square root unit (ISRU)
   f_ISRLU,        // inv.squ.root linear unit (ISRLU)
   f_softplus,     // softplus
   f_bentident,    // bent identity
   f_sinusoid,     // sinusoid
   f_sinc,         // cardinal sine (sinc)
   f_gaussian,     // gaussian
   f_differentiable_hardstep, // differentiable hardstep
   f_leaky_diff_hardstep, // leaky differentiable hardstep
   f_softmax,      // normalized exponential (softmax)
   f_oblique_sigmoid, // oblique sigmoid
   f_log_rectifier, // log rectifier
   f_leaky_log_rectifier, // leaky log rectifier
   f_ramp          // ramp
                   // note: the softmax function can't be part of this library because it has no single input but needs the other neurons of the layer, too, so it
  };               //       needs to be defined from within the neural network code
```
Usage example:
```cpp
// normal usage
double h = activate(x, f_ReLU);
// get derivative of activation function
double d = deactivate(h, f_ReLU);
```
___
# git/DataScience/sample.h
The Sample class returns an abundance of statistical data from a given sample.

Available methods:
```cpp
double mean(); // returns the arithmetic mean of a sample that has been provided with the parametric constructor
double median(); // returns the median of a sample that has been provided with the parametric constructor
double weighted_average(bool as_series=false); // returns the weighted average 
std::vector<int> ranking(bool low_to_high=true); // returns a ranking (as std::vector<int>) of a sample that has been provided with the parametric constructor
std::vector<T> exponential_smoothing(bool as_series=false); // provides exponential smoothing (for time series) of a sample that has been provided with the parametric constructor
double variance(); // returns the variance of a sample that has been provided with the parametric constructor
double stddev(std::vector<T>& sample); // returns the standard deviation from any sample vector of typename T
double stddev(); // returns the standard deviation from sample as received from parametric object constructor     
unsigned int find(T value,int index_from=0,int index_to=__INT_MAX__); // count the number of appearances of a given value within a a sample that has been provided with the parametric constructor
double Dickey_Fuller(); // performs an augmented Dickey-Fuller test (unit root test for stationarity) on a sample that has been provided with the parametric constructor
void correlation(); // performs a Pearson correlation on two samples that have been provided with the parametric constructor
std::vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2); // performs a stationary transformation (e.g. for time series) on a sample that has been provided with the parametric contructor
std::vector<T> sort(bool ascending=true); // returns a sorted copy of the values of a sample that has been provided with the parametric constructor
std::vector<T> shuffle(); // returns a randomly shuffled copy of the values of a sample that has been provided with the parametric constructor
std::vector<T> log_transform(); // returns a log transformation of a sample that has been provided with the parametric constructor
double Engle_Granger(); // performs an Engle-Granger test (for cointegration) on a sample that has been provided with the parametric constructor
double polynomial_predict(T x); // predict y-values given new x-values, assuming polynomial dependence
double polynomial_MSE(); // calculate mean squared error, assuming polynomial dependence
bool isGoodFit(double threshold=0.95); // check goodness of fit (based on r_squared)
void polynomial_regression(int power=5); // performs polynomial regression on a sample that has been provided with the parametric constructor
void linear_regression(); // performs linear regression on a sample that has been provided with the parametric constructor
double get_Pearson_R(); // returns Pearson R from a correlation testing of two samples
double get_slope(); // returns the slope from linear regression of a sample provided with the parametric constructor
double get_y_intercept(); // returns the intersection of linear regression with the y axis
double get_r_squared(); // returns the r2 from regression
double get_z_score(); // returns the z-score from the correlation of two samples that have been provided with the parametric constructor
double get_t_score(); // returns the t-score from the correlation of two samples that have been provided with the parametric constructor
double get_Spearman_Rho(); // returns the Spearman Rho from the correlation of two samples that have been provided with the parametric constructor
double get_x_mean(); // returns the arrithmetic mean of the first of two samples that have been provided with the parametric constructor
double get_y_mean(); // returns the arrithmetic mean of the second of two samples that have been provided with the parametric constructor
double get_covariance(); returns the covariance of the correlation of two samples that have been provided with the parametric constructor
Histogram<T> histogram(unsigned int bars); returns a histogram (with specified number of bars) of the values of a sample that has been provided with the parametric constructor
        
// constructors
Sample(std::vector<T> data_vect) // parametric constructor for single std::vector<T> sample
Sample(std::vector<T> x_vect, std::vector<T> y_vect); // parametric constructor for two samples of type std::vector<T>
Sample(T *data_vect); // parametric constructor for single array-type sample
Sample(T *x_vect, T *y_vect) // parametric constructor for two array-type samples
```
