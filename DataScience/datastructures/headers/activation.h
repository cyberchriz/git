// +=================================+   
// | Activation Functions            |
// +=================================+
#pragma once

// forward declaration
template<typename T> class Array;

enum ActFunc {
    ReLU,       // rectified linear unit (ReLU)
    lReLU,      // leaky rectified linear unit (LReLU)
    ELU,        // exponential linar unit (ELU)
    sigmoid,    // sigmoid (=logistic)
    tanh,       // hyperbolic tangent (tanh)
    softmax,    // softmax (=normalized exponential)
    ident       // identity function
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
                case ReLU: Function::ReLU(); break;
                case lReLU: Function::lReLU(); break;
                case ELU: Function::ELU(); break;
                case sigmoid: Function::sigmoid(); break;   
                case tanh: Function::tanh(); break;
                case softmax: Function::softmax(); break;
                case ident: Function::ident(); break;
                default: /* do nothing */ break;
            }
            return std::move(*result);
        }
        Array<T> derivative(ActFunc method){
            switch (method){
                case ReLU: Derivative::ReLU(); break;
                case lReLU: Derivative::lReLU(); break;
                case ELU: Derivative::ELU(); break;
                case sigmoid: Derivative::sigmoid(); break;   
                case tanh: Derivative::tanh(); break;
                case softmax: Derivative::softmax(); break;
                case ident: Derivative::ident(); break;
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
            result = std::make_unique<Array<T>>(arr.get_shape());
        } 
};