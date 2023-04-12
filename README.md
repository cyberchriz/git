author: cyberchriz (Christian Suer)
language: C++
___
## OVERVIEW
This repository provides functionality for
- Data Science
    - neural networks of various topologies
        - multilayer perceptron
            - full flexibility for layers and neurons
            - optimizers: Vanilla, Nesterov, ADAM, ADADELTA, Adagrad, RMSprop
            - input & label scaling
        - recurrent neural networks (RNN)
        - autoencoder
        - general:
            - activation functions (+corresponding derivatives):
                sigmoid, ELU, ReLU, leaky ReLU, tanh, oblique tanh, tanh rectifier, arctan, arsinh, softsign (Elliot), ISRU, ISRLU, softsign, softplus, ident, bent ident, sinusoid, sinc, gaussian, differentiable hardstep, leaky diff. hardstep, log rectifier, leaky log rectifier, ramp
            - weight initializers
    - sample analysis
        - single sample
            - linear regression
                - slope + y-axis intercept
                - r squared + goodness of fit
                - prediction
            - polynomial regression
                - r squared + goodness of fit
                - mean squared error (MSE)
                - prediction
            - stationary transformation
            - Dickey-Fuller test (for stationarity)
            - logarithmic transformation
            - exponential smoothing
            - mean, median, weighted average
            - variance, standard deviation
            - histogram
            - ranking
            - find (=number of occurences of a given value)
            - shuffle
            - sort
        - two samples
            - correlation
                - Pearson
                - Spearman
                - Student's t-score
                - z-score
            - covariance
            - Engle-Granger test (for cointegration)
            - x_mean, y_mean
    - distributions
        - random numbers (gaussian, uniform, Laplace, Cauchy, Lomax, Pareto, binary, sign)
        - cumulative distribution functions (gaussian/normal, Laplace, Cauchy, Lomax, Pareto)
        - probability density functions (gaussian/normal, Laplace, Cauchy, Lomax, Pareto)
    - custom n-dimensional arrays (stack-allocated)
        - fill (zeros, value, identity, random gaussian, random uniform)
        - mean, median, variance, standard deviation
        - sum, product
        - find, replace
        - elementwise add / substract / multiply / divide / power / square root
        - elementwise apply custum function
        - 1d vector
        - 2d matrix
            - dotproduct
            - transpose
- utilities
    - debugging & performance
        - heap allocation logger
        - timer
    - general purpose
        - number validation (NaN/InF check)
___
THIS REPOSITORY IS 'WORK IN PROGRESS'. MOST OF THE CODE WILL WORK WITHOUT ANY ISSUES, BUT I'LL BE CONSTANTLY IMPROVING IT AND ADDING TO IT. YOU MAY STILL FEEL FREE TO USE IT AS IT IS, BUT KEEP IN MIND THAT NOTHING's PERFECT.
I'M SELF-TAUGHT AND NOT A PROFESSIONAL PROGRAMMER.
THIS REPOSITORY IS PUBLIC FOR A REASON - COLLABORATION, IMPLEMENTATION IN YOUR OWN CODE, CRITIQUE AND SUGGESTIONS ARE WELCOME.
