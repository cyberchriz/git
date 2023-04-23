#pragma once
#include <cmath> 
#include <memory>
#include <cstdarg>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "../../distributions/headers/random_distributions.h"
//#define MEMLOG
#include "../../../utilities/headers/memlog.h"

// forward declarations
template<typename T> class Array;
template<typename T> class Matrix;
template<typename T> class Vector;

// list of time series differencing methods for stationarity transformation
enum DIFFERENCING
  {
   integer=1,
   logreturn=2,
   fractional=3,
   deltamean=4,
   original=5
  };

// data structure for sample histograms
// histo_x: right(=upper) border of given bar 
template<typename T>
struct Histogram_bar{
    T lower_boundary;
    T upper_boundary;
    int abs_count;
    double rel_count;
};

// histogram structure
template<typename T>
struct Histogram{
    T min, max;
    T bar_width;
    T width;
    std::unique_ptr<Histogram_bar<T>[]> bar;
    Histogram(int bars){
        bar = std::make_unique<Histogram_bar<T>[]>(bars);
    }
    Histogram() = delete;
    ~Histogram(){};
};

// Linear Regression struct
template<typename T>
struct LinReg{
    double x_mean, y_mean=0;
    double y_intercept, slope;
    double r_squared;
    std::unique_ptr<double[]> y_regression;
    std::unique_ptr<double[]> residuals;
    double SST=0;
    double SSR=0;
    T predict(const T x){return y_intercept + slope * x;}
    bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
    // constructor & destructor
    LinReg(const int elements){
        y_regression = std::make_unique<double[]>(elements);
        residuals = std::make_unique<double[]>(elements);
    }
    LinReg() = delete;
    ~LinReg(){};
};

// Polynomial Regression struct
template<typename T>
struct PolyReg{
    public:
        double SS_res=0;
        double SS_tot=0;
        double RSS=0;
        double MSE;      
        double RSE;
        double y_mean=0;
        double x_mean=0;
        double r_squared;
        std::unique_ptr<double[]> coefficient;  
        bool is_good_fit(double threshold=0.95){return r_squared>threshold;}
        T predict(const T x){
            double y_pred = 0;
            for (int p = 0; p<=power;p++) {
                y_pred += coefficient[p] * std::pow(x, p);
            }
            return y_pred;
        };  
        // constructor & destructor
        PolyReg(const int elements, const int power){
            this->elements=elements;
            this->power=power;
            coefficient = std::make_unique<double[]>(power+1);
        }
        ~PolyReg(){}
    private:
        int elements;
        int power;
};

// Correlation struct
template<typename T>
struct Correlation{ 
    double x_mean, y_mean;  
    double x_stddev, y_stddev;     
    double y_intercept, slope;
    double covariance;
    double Pearson_R, Spearman_Rho;   
    double r_squared;    
    double RSS, SST, SSE, SSR = 0; 
    double z_score, t_score;          
    std::vector<T> y_predict;
    void print(){
        std::cout
        << "Correlation Results (this vs. other):"
        << "\n   - x_mean = " << x_mean
        << "\n   - y_mean = " << y_mean
        << "\n   - x_stddev = " << x_stddev
        << "\n   - y_stddev = " << y_stddev
        << "\n   - y_intercept = " << y_intercept
        << "\n   - slope = " << slope
        << "\n   - covariance = " << covariance
        << "\n   - Pearson_R = " << Pearson_R
        << "\n   - Spearman_Rho = " << Spearman_Rho
        << "\n   - r_squared = " << r_squared
        << "\n   - RSS = " << RSS
        << "\n   - SST = " << SST
        << "\n   - SSE = " << SSE
        << "\n   - SSR = " << SSR
        << "\n   - z_score = " << z_score
        << "\n   - t_score = " << t_score << std::endl;
    }
    // constructor
    Correlation(int elements){
        y_predict.resize(elements);
    }  
};

// class for multidimensional arrays
template<typename T>
class Array{
    public:     
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value);
        void set(const std::vector<int>& index, const T value);
        T get(const std::initializer_list<int>& index) const;
        T get(const std::vector<int>& index) const;
        int get_dimensions() const;
        int get_size(int dimension) const;
        int get_elements() const;
        int get_subspace(int dimension) const;        

        // nested Struct for Filling / Intializing
        struct Fill {
            void values(const T value);
            void zeros();
            void identity();
            void random_gaussian(const T mu=0, const T sigma=1);
            void random_uniform(const T min=0, const T max=1.0);
            virtual void range(const T start=0, const T step=1);
            Fill(Array<T>& arr):arr(arr){}
            Array<T>& arr;
        };
        friend class Array<T>::Fill;

        // basic distribution properties
        T min() const;
        T max() const;
        double mean() const;
        double median() const;
        T mode() const;
        double variance() const;
        double stddev() const;
        double skewness() const;
        double kurtosis() const;

        // addition
        T sum() const;
        std::unique_ptr<Array<T>> operator+(const T value) const;
        std::unique_ptr<Array<T>> operator+(const Array<T>& other) const;
        std::unique_ptr<Array<T>> operator++(int) const; //=postfix increment
        Array<T>& operator++(); //=prefix increment
        void operator+=(const T value);
        void operator+=(const Array<T>& other);

        // substraction
        void substract(const T value);
        std::unique_ptr<Array<T>> operator-(const T value) const;
        std::unique_ptr<Array<T>> operator-(const Array<T>& other) const;
        std::unique_ptr<Array<T>> operator--(int) const; //=postfix decrement
        Array<T>& operator--(); //=prefix decrement
        void operator-=(const T value);
        void operator-=(const Array<T>& other);

        // multiplication
        T product() const; // product reduction (=multiplying all elements with each other)
        virtual std::unique_ptr<Array<T>> operator*(const T factor) const; // elementwise multiplication
        virtual std::unique_ptr<Array<T>> tensordot(const Array<T>& other, const std::vector<int>& axes) const; //=tensor reduction
        virtual T dotproduct(const Array<T>& other) const; //=scalar product
        virtual T operator*(const Array<T>& other) const; //=alias for the dotproduct (=scalar product)
        void operator*=(const T factor);
        std::unique_ptr<Array<T>> Hadamard(const Array<T>& other) const;
        
        // division
        std::unique_ptr<Array<T>> operator/(const T quotient) const;
        void operator/=(const T quotient);

        // modulo
        void operator%=(const double num);
        std::unique_ptr<Array<double>> operator%(const double num) const;

        // exponentiation & logarithm
        void pow(const T exponent);
        void pow(const Array<T>& other);
        void sqrt();
        void log();
        void log10();

        // rounding
        void round();
        void floor();
        void ceil();

        // find, replace
        void replace(const T old_value, const T new_value);
        int find(const T value) const;

        // custom functions
        void function(const T (*pointer_to_function)(T));

        // assignment
        virtual Array<T>& operator=(const Array<T>& other);

        // elementwise comparison by single value
        std::unique_ptr<Array<bool>> operator>(const T value) const;
        std::unique_ptr<Array<bool>> operator>=(const T value) const;
        std::unique_ptr<Array<bool>> operator==(const T value) const;
        std::unique_ptr<Array<bool>> operator!=(const T value) const;
        std::unique_ptr<Array<bool>> operator<(const T value) const;
        std::unique_ptr<Array<bool>> operator<=(const T value) const;

        // elementwise comparison with second array
        std::unique_ptr<Array<bool>> operator>(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator>=(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator==(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator!=(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator<(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator<=(const Array<T>& other) const;

        // elementwise logical operations
        std::unique_ptr<Array<bool>> operator&&(const bool value) const;
        std::unique_ptr<Array<bool>> operator||(const bool value) const;
        std::unique_ptr<Array<bool>> operator!() const;
        std::unique_ptr<Array<bool>> operator&&(const Array<T>& other) const;
        std::unique_ptr<Array<bool>> operator||(const Array<T>& other) const;

        // type casting
        template<typename C> operator Array<C>();
        
        // conversion
        std::unique_ptr<Vector<T>> flatten() const;
        virtual std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0) const;
        virtual std::unique_ptr<Matrix<T>> asMatrix() const;
        std::unique_ptr<Array<T>> asArray(const std::initializer_list<int>& initlist, T init_value=0) const;

        // output
        void print(std::string comment="", std::string delimiter=", ", std::string line_break="\n", bool with_indices=false) const;

        // nested Struct for neural network activations
        struct Activation {
            private:
                static constexpr double alpha = 0.01; // slope constant for lReLU and lELU
                struct Function {
                    void ReLU();
                    void lReLU();
                    void ELU();
                    void sigmoid();
                    void tanh();
                    void softmax();     
                    void ident();   
                };
                struct Derivative {
                    void ReLU();
                    void lReLU();
                    void ELU();
                    void sigmoid();
                    void tanh();
                    void softmax();
                    void ident();
                };
            public:
                enum Method {
                    ReLU,       // rectified linear unit (ReLU)
                    lReLU,      // leaky rectified linear unit (LReLU)
                    ELU,        // exponential linar unit (ELU)
                    sigmoid,    // sigmoid (=logistic)
                    tanh,       // hyperbolic tangent (tanh)
                    softmax,    // softmax (=normalized exponential)
                    ident       // identity function
                };
                void function(Method method);
                void derivative(Method method);
                Function _function;
                Derivative _derivative;
                Activation(Array<T>& arr):arr(arr){}
                Array<T>& arr;
        };
        friend class Array<T>::Activation;

        // nested Struct for scaling methods
        struct Scaling {  
            void minmax(T min=0,T max=1){
                T data_min = this->_data.min();
                T data_max = this->_data.max();
                double factor = (max-min) / (data_max-data_min);
                for (int i=0; i<this->get_elements(); i++){
                    this->_data[i] = (this->_data[i] - data_min) * factor + min;
                }
            };
            void mean(){
                T data_min = this->_data.min();
                T data_max = this->_data.max();
                T range = data_max - data_min;
                double mean = this->mean();
                for (int i=0; i<this->get_elements(); i++){
                    this->_data[i] = (this->_data[i] - mean) / range;
                }
            };
            void standardized(){
                double mean = this->mean();
                double stddev = this->stddev();
                for (int i=0; i<this->get_elements(); i++){
                    this->_data[i] = (this->_data[i] - mean) / stddev;
                }                    
            };
            void unit_length(){
                // calculate the Euclidean norm of the data array
                T norm = 0;
                int elements = this->get_elements();
                for (int i = 0; i < elements; i++) {
                    norm += std::pow(this->_data[i], 2);
                }
                if (norm==0){return;}
                norm = std::sqrt(norm);
                // scale the data array to unit length
                for (int i = 0; i < elements; i++) {
                    this->_data[i] /= norm;
                }                    
            };
            Scaling(Array<T>& arr):arr(arr){}
            Array<T>& arr;
        };
        friend class Array<T>::Scaling;

        // constructor & destructor declarations
        Array(){};
        Array(const std::initializer_list<int>& init_list);
        Array(const std::vector<int>& dimensions);
        virtual ~Array();

    protected:
    
        // protected member variables
        bool equal_size(const Array<T>& other) const;
        int _elements=0; // total number of _elements in all _dimensions
        int _dimensions=0;
        std::vector<int> _size; // holds the size (number of _elements) per individual dimension 
        std::vector<int> _subspace_size;
        
        // protected methods
        int get_element(const std::initializer_list<int>& index) const;
        int get_element(const std::vector<int>& index) const;
        void resizeArray(std::unique_ptr<T[]>& arr, const int newSize);
        std::initializer_list<int> array_to_initlist(int* arr, int size) const;
        std::unique_ptr<int[]> initlist_to_array(const std::initializer_list<int>& lst) const;            

    public:
        // public member variables
        std::unique_ptr<T[]> _data = nullptr; // 1dimensional array of source _data
        std::unique_ptr<Fill> fill;
        std::unique_ptr<Activation> activation;
        std::unique_ptr<Scaling> scale;
};

// derived class from Array<T>, for 2d matrix
template<typename T>
class Matrix : public Array<T>{
    public:
        // getters & setters
        void set(const int row, const int col, T value);
        T get(const int row, const int col) const;

        // special matrix operations
        T dotproduct(const Matrix<T>& other) const; //=scalar product
        T operator*(const Matrix<T>& other) const; //=Alias for the dotproduct (=scalar product)
        std::unique_ptr<Matrix<T>> tensordot(const Matrix<T>& other) const; //=tensor reduction, alias for operator*
        std::unique_ptr<Matrix<T>> transpose() const;

        // assignment
        Matrix<T>& operator=(const Matrix<T>& other);

        // conversion
        std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0) const override;

        // constructor declarations
        Matrix(){};
        Matrix(const int rows, const int cols);
        ~Matrix() override;
};

// derived class from Array<T>, for 1d vectors
template<typename T>
class Vector : public Array<T>{
    public:
        // getters & setters
        void set(const std::initializer_list<int>& index, const T value) = delete;
        T get(const std::initializer_list<int>& index) = delete;
        void set(const int index, const T value);
        T get(const int index) const;

        // dynamic handling
        int push_back(T value);
        T pop();
        T erase(const int index);
        int grow(const int additional_elements,T value=0);
        int shrink(const int remove_amount);        
        void resize(const int new_size);        
        int get_capacity() const;
        int size() const;
        std::unique_ptr<Vector<T>> flatten()=delete;

        // Multiplication
        T dotproduct(const Vector<T>& other) const; //=scalar product, alias for operator*
        T operator*(const Vector<T>& other) const; //=scalar product, alias for dotproduct()

        // sample analysis
        std::unique_ptr<Vector<int>> ranking() const;
        std::unique_ptr<Vector<T>> exponential_smoothing(bool as_series=false) const;
        double weighted_average(bool as_series=true) const;
        double Dickey_Fuller(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        double Engle_Granger(const Vector<T>& other) const;
        std::unique_ptr<Vector<T>> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2) const;
        std::unique_ptr<Vector<T>> sort(bool ascending=true) const;
        std::unique_ptr<Vector<T>> shuffle() const;
        std::unique_ptr<LinReg<T>> linear_regression(const Vector<T>& other) const;
        std::unique_ptr<PolyReg<T>> polynomial_regression(const Vector<T>& other, const int power=5) const;
        std::unique_ptr<Correlation<T>> correlation(const Vector<T>& other) const;
        double covariance(const Vector<T>& other) const;
        std::unique_ptr<Histogram<T>> histogram(uint bars) const;

        // assignment
        Vector<T>& operator=(const Vector<T>& other);

        // conversion
        std::unique_ptr<Matrix<T>> asMatrix(const int rows, const int cols, T init_value=0)  const override;
        std::unique_ptr<Matrix<T>> asMatrix() const override;
        std::unique_ptr<Matrix<T>> transpose() const;
        std::unique_ptr<Vector<T>> reverse() const;

        // constructor & destructor declarations
        Vector(){};
        Vector(const int elements);
        ~Vector() override;
    private:
        const float _reserve = 0.5;
        int _capacity;
};



// the corresponding file with the definitions must be included
// because this is the template class
// (in order to avoid 'undefined reference' compiler errors)
#include "../sources/array.cpp"