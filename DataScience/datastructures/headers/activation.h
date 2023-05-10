// +=================================+   
// | Activation Functions            |
// +=================================+
#pragma once

// forward declaration
template<typename T> class Array;

enum ActFunc {
    RELU,       // rectified linear unit (ReLU)
    LRELU,      // leaky rectified linear unit (LReLU)
    ELU,        // exponential linar unit (ELU)
    SIGMOID,    // sigmoid (=logistic)
    TANH,       // hyperbolic tangent (tanh)
    SOFTMAX,    // softmax (=normalized exponential)
    IDENT       // identity function
};
// class declaration
template<typename T>
class Activation {
    private:
        // private members
        Array<T>* arr;
        static constexpr double alpha = 0.01; // slope constant for lReLU and lELU
        std::unique_ptr<Array<T>> result;        

        // private methods
        struct Function {
            public:
                void ReLU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i] * arr->data[i]>0;
                    }  
                }
                void lReLU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i]>0 ? arr->data[i] : arr->data[i]*alpha;
                    }
                }
                void ELU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i]>0 ? arr->data[i] : alpha*(std::exp(arr->data[i])-1); 
                    } 
                }
                void sigmoid(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = 1/(1+std::exp(-arr->data[i])); 
                    } 
                }
                void tanh(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = std::tanh(arr->data[i]);
                    } 
                }
                void softmax(){
                    // TODO
                }     
                void ident(){
                    // do nothing
                    return;
                }   
            };
        struct Derivative {
            public:
                void ReLU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i]>0 ? 1 : 0;
                    }  
                }
                void lReLU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i]>0 ? 1 : alpha;
                    }
                }
                void ELU(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = arr->data[i]>0 ? 1 : alpha*std::exp(arr->data[i]);
                    }
                }
                void sigmoid(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = std::exp(arr->data[i])/std::pow(std::exp(arr->data[i])+1,2); 
                    }
                }
                void tanh(){
                    for (int i=0;i<arr->get_elements();i++){
                        result->data[i] = 1-std::pow(std::tanh(arr->data[i]),2);
                    }
                }
                void softmax(){
                    // TODO !!
                }
                void ident(){
                    result->fill.values(1);
                }
            };
    public:
        Array<T> function(ActFunc method){
            switch (method){
                case RELU: Function::ReLU(); break;
                case LRELU: Function::lReLU(); break;
                case ELU: Function::ELU(); break;
                case SIGMOID: Function::sigmoid(); break;   
                case TANH: Function::tanh(); break;
                case SOFTMAX: Function::softmax(); break;
                case IDENT: Function::ident(); break;
                default: /* do nothing */ break;
            }
            return std::move(*result);
        }
        Array<T> derivative(ActFunc method){
            switch (method){
                case RELU: Derivative::ReLU(); break;
                case LRELU: Derivative::lReLU(); break;
                case ELU: Derivative::ELU(); break;
                case SIGMOID: Derivative::sigmoid(); break;   
                case TANH: Derivative::tanh(); break;
                case SOFTMAX: Derivative::softmax(); break;
                case IDENT: Derivative::ident(); break;
                default: /* do nothing */ break;
            }
            return std::move(*result);
        }
        Function _function;
        Derivative _derivative;
        // constructor
        Activation() :
            arr(nullptr){} 
        Activation(Array<T>* arr) : arr(arr){
            result = std::make_unique<Array<T>>(arr->get_shape());
        } 
};