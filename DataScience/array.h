#pragma once
#include <cmath> 
#include </home/christian/Documents/own_code/c++/DataScience/sample.h>
using namespace std;

// class for multidimensional arrays

class Array{
    private:
        double *data;
        int elements;
        int dimensions;
        int *size;
    public:
        int get_element(int *index);
        void set(int *index, double value=0);
        void set(int index, double value=0){data[index]=value;};
        double get(int *index);
        double get(int index){return data[index];};
        int get_dimensions(){return dimensions;};
        int get_size(int dimension){return size[dimension];};
        int items(){return elements;};
        void fill(double value=0);
        void identity_fill();
        void random_normal(double mu=0, double sigma=1);
        void random_uniform(double x_mean=0.5,double range=0.5);
        double mean(){return Sample(data).mean();};
        double median(){return Sample(data).median();};
        double variance(){return Sample(data).variance();};
        double stddev(){return Sample(data).stddev();};
        double sum();
        void add(double value);
        void add(Array* array);
        double multiply();
        void multiply(double factor);
        void multiply(Array* arrayB);
        void substract(double value);
        void substract(Array* arrayB);
        void divide(double quotient);
        void divide(Array* arrayB);
        void power(double exponent);
        void root();
        void replace(double old_value, double new_value);
        int find(double value);
        void map(double (*pointer_to_function)(double));
        Array* copy();
        Array* dotproduct(Array* arrayB);
        Array* transpose();
        // constructor declaration
        Array(int *dim_size);
        // destructor declaration
        ~Array();
};

// constructor definition
Array::Array(int *dim_size){
    dimensions = sizeof(dim_size)/__SIZEOF_POINTER__;
    elements=1;
    size = new int[dimensions];
    for (int d=0;d<dimensions;d++){
        elements*=dim_size[d];
        size[d]=dim_size[d];
    }
    data = new double[elements];
};

// destructor definition
Array::~Array(){
    delete size;
    delete data;
};

// set a value by index
void Array::set(int *index, double value){
    data[get_element(index)] = value;
};

// get a array value from index
double Array::get(int *index){
    int element=get_element(index);
    if (element==-1 || sizeof(index)==0){return NAN;}
    return data[element];
};

// get 1d element index from multidimensional index
int Array::get_element(int *index){
    if (sizeof(index)/__SIZEOF_POINTER__ > dimensions){
        return -1;
    }
    int result=index[0];
    // int result=index[0] + index[1]*size[0] + index[2]*size[0]*size[1] + index[3]*size[0]*size[1]*size[2] ...;
    for (int d=1;d<dimensions;d++){
        int add=index[d];
        for(int s=d;s>0;s--){
            add*=size[s-1];
        }
        result+=add;
    }
    return result;
};

// fill entire array with given value
void Array::fill(double value){
    for (int i=0;i<elements;i++){
        data[i]=value;
    }
};

// fill array with identity matrix
// (works with array of up to 4 dimensions)
void Array::identity_fill(){
    if(dimensions==1){this->fill(1);}
    else if (dimensions==2){
        for (int x=0;x<this->size[0];x++){
            for (int y=0;y<this->size[1];y++){
                int index[]={x,y};
                data[this->get_element(index)] = x==y;
            }
        }
    }
    else if (dimensions==3){
        for (int x=0;x<this->size[0];x++){
            for (int y=0;y<this->size[1];y++){
                for (int z=0;z<this->size[2];z++){
                    int index[]={x,y,z};
                    data[this->get_element(index)] = x==y && y==z;
                }
            }
        }
    }
    else if (dimensions==4){
        for (int x=0;x<this->size[0];x++){
            for (int y=0;y<this->size[1];y++){
                for (int z=0;z<this->size[2];z++){
                    for (int a=0;a<this->size[3];a++){
                        int index[]={x,y,z,a};
                        data[this->get_element(index)] = x==y && y==z && z==a;
                    }
                }
            }
        }
    }    
}

// fill with random normal distribution
void Array::random_normal(double mu,double sigma){
    for (int i=0;i<elements;i++){
        double random=(double)rand() / RAND_MAX;                    // random value within range 0-1
        random/=sqrt(2*M_PI*pow(sigma,2));                          // reduce to the top of the distribution (f(x_val=mu))
        double algsign=1;if (rand()>(0.5*RAND_MAX)){algsign=-1;}    // get random algebraic sign
        data[i] = algsign * (mu + sigma * sqrt (-2 * log (random / (1/sqrt(2*M_PI*pow(sigma,2))))));
    }
};

// fill with random uniform distribution
void Array::random_uniform(double x_mean,double range){
    for (int i=0;i<elements;i++){
        double random=(double)rand() / (0.5*RAND_MAX) - 1;          // random value within range +/- 1
        random*=range;
        random+=x_mean;
        data[i] = random;
    }
};

// modifies the array by applying the referred function (should take a single double as argument)
// template: Array::map(&func);
void Array::map(double (*pointer_to_function)(double)){
    for (int i=0;i<elements;i++){
        data[i]=pointer_to_function(data[i]);
    }
};

// returns the sum of all array elements
double Array::sum(){
    double result=0;
    for (int i=0;i<elements;i++){
        result+=data[i];
    }
    return result;
}

// returns the product of all array elements
double Array::multiply(){
    double result = data[0];
    for (int i=1;i<elements;i++){
        result*=data[i];
    }
    return result;
}

// elementwise (scalar) multiplication by given factor
void Array::multiply(double factor){
    for (int i=0;i<elements;i++){
        data[i]*=factor;
    }
}

// elementwise (scalar) division by given quotient
void Array::divide(double quotient){
    if (quotient==0){return;}
    for (int i=0;i<elements;i++){
        data[i]/=quotient;
    }
}

// elementwise (scalar) addition of specified value
void Array::add(double value){
    for (int i=0;i<elements;i++){
        data[i]+=value;
    }
}

// elementwise (scalar) substraction of specified value
void Array::substract(double value){
    for (int i=0;i<elements;i++){
        data[i]-=value;
    }
}

// elementwise (scalar) multiplication with second array (with matching dimensions)
void Array::multiply(Array* arrayB){
    int n=fmin(arrayB->elements,this->elements);
    for (int i=0;i<n;i++){
        this->data[i]*=arrayB->data[i];
    }
}

// elementwise (scalar) addition of second array (with matching dimensions)
void Array::add(Array* arrayB){
    int n=fmin(arrayB->elements,this->elements);
    for (int i=0;i<n;i++){
        this->data[i]+=arrayB->data[i];
    }
}

// elementwise (scalar) substraction of second array (with matching dimensions)
void Array::substract(Array* arrayB){
    int n=fmin(arrayB->elements,this->elements);
    for (int i=0;i<n;i++){
        this->data[i]-=arrayB->data[i];
    }
}

// elementwise (scalar) division by second array (with matching dimensions)
void Array::divide(Array* arrayB){
    int n=fmin(arrayB->elements,this->elements);
    for (int i=0;i<n;i++){
        this->data[i]/=arrayB->data[i];
    }
}

// replace all findings of given value by specified new value
void Array::replace(double old_value, double new_value){
    for (int i=0;i<elements;i++){
        if (data[i]==old_value){
            data[i]=new_value;
        }
    }
}

// returns the number of occurrences of the specified value
int Array::find(double value){
    int counter=0;
    for (int i=0;i<elements;i++){
        counter+=(data[i]==value);
    }
    return counter;
}

// elementwise (scalar) exponentiation by given power
void Array::power(double exponent){
    for (int i=0;i<elements;i++){
        data[i]=pow(data[i],exponent);
    }
}

// elementwise square root
void Array::root(){
    for (int i=0;i<elements;i++){
        data[i]=sqrt(data[i]);
    }
}

// returns an identical copy
Array* Array::copy(){
    Array* arr_copy = new Array(this->size);
    for (int i=0;i<this->elements;i++){
        arr_copy->data[i] = this->data[i];
    }
    return arr_copy;
}

// returns the dotproduct of two 2d matrices
Array* Array::dotproduct(Array* arrayB){
    Array* result=nullptr;
    if (this->size[0] == arrayB->size[1]){
        int new_dimension[]={arrayB->size[1],this->size[0]};
        result = new Array(new_dimension);
        int result_index[2];
        for (int result_y=0;result_y<this->size[0];result_y++){
            for (int result_x=0;result_x<arrayB->size[1];result_x++){
                int result_index[]={result_x,result_y};
                result->data[result->get_element(result_index)] = 0;
                for (int i=0;i<this->size[0];i++){
                    int A_index[]={i,result_y};
                    int B_index[]={result_x,i};
                    result->data[result->get_element(result_index)] += this->data[this->get_element(A_index)] * arrayB->data[arrayB->get_element(B_index)];
                }
            }
        }
    }
    return result;
}

// returns the matrix transpose
// note: only for 1d or 2d array!!
Array* Array::transpose(){
    Array* result=nullptr;
    if (this->dimensions==1){
        int new_dimensions[] = {1,this->size[0]};
        result = new Array(new_dimensions);
        for (int x=0;x<this->size[0];x++){
            int index[] = {0,x};
            result->data[result->get_element(index)] = this->data[x];
        }
    }
    else if (this->dimensions==2){
        int new_dimensions[] = {this->size[1],this->size[0]};
        result = new Array(new_dimensions);
        for (int x=0;x<this->size[0];x++){
            for (int y=0;y<this->size[1];y++){
                int source_index[] = {x,y};
                int target_index[] = {y,x};
                result->data[result->get_element(target_index)] = this->data[this->get_element(source_index)];
            }
        }        
    }
    return result;
}