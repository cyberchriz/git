#include "../headers/array.h"

// constructor for multi-dimensional array;
// pass dimensions (elements per dimension) as array
template<typename T>
Array<T>::Array(u_int32_t &dim_size){
    this->size = dim_size;
    dimensions = sizeof(dim_size) / sizeof(u_int32_t);
    elements=1;
    for (u_int32_t d=0;d<dimensions;d++){
        elements*=dim_size[d];
    }
    data.resize(elements)
};

// constructor for 1d vector
template<typename T>
Vector<T>::Vector(u_int32_t elements){
    size.resize(1);
    this->elements = elements;
    size[0] = elements;
    dimensions = 1;
    data.resize(elements);
}

// constructor for 2d matrix
template<typename T>
Matrix<T>::Matrix(u_int32_t elements_x, u_int32_t elements_y){
    size.resize(2);
    elements = elements_x * elements_y;
    size[0] = elements_x;
    size[1] = elements_y;
    dimensions = 2;
    data.resize(elements);
}

// set a value by array index
// (for multi-dimensional arrays)
template<typename T>
void Array<T>::set(u_int32_t &index, T value){
    data[get_element(index)] = value;
};

// set a value by index for 1d vector
template<typename T>
void Vector<T>::set(u_int32_t index, T value){
    data[index] = value;
};

// set a value by index for 2d matrix
template<typename T>
void Matrix<T>::set(u_int32_t index_x, u_int32_t index_y, T value){
    data[get_element({index_x,index_y})] = value;
}

// get value from array index
// (pass array index as array)
template<typename T>
T Array<T>::get(u_int32_t &index){
    u_int32_t element=get_element(index);
    if (isnan(element) || element>elements){return NAN;}
    return data[element];
};

// get 1d element index from multidimensional index
template<typename T>
u_int32_t Array<T>::get_element(u_int32_t &index){
    // confirm valid number of dimensions
    if (sizeof(index)/sizeof(u_int32_t) > dimensions){
        return NAN;
    }
    // principle: result=index[0] + index[1]*size[0] + index[2]*size[0]*size[1] + index[3]*size[0]*size[1]*size[2] + ...
    static u_int32_t result;
    static u_int32_t add;
    result=index[0];
    for (u_int32_t i=1;i<dimensions;i++){
        add=index[i];
        for(u_int32_t s=0;s<i;s++){
            add*=size[s];
        }
        result+=add;
    }
    return result;
};

// fill entire array with given value
template<typename T>
void Array<T>::fill_values(T value){
    for (u_int32_t i=0;i<elements;i++){
        data[i]=value;
    }
};

// fill array with identity matrix
template<typename T>
void Array<T>::fill_identity(){
    // get size of smallest dimension
    u_int32_t max_index=__UINT32_MAX__;
    for (int i=0;i<dimensions;i++){
        max_index=std::fmin(max_index,size[i]);
    }
    u_int32_t index[dimensions];
    for (u_int32_t i=0;i<max_index;i++){
        for (u_int32_t d=0;d<dimensions;d++){
            index[d]=i;
        }
        set(index,1);
    }
}

// fill with random normal distribution
template<typename T>
void Array<T>::fill_random_gaussian(T mu, T sigma){
    for (u_int32_t i=0;i<elements;i++){
        data[i] = Random::gaussian(mu,sigma);
    }
};

// fill with random uniform distribution
template<typename T>
void Array<T>::fill_random_uniform(T min,T max){
    for (u_int32_t i=0;i<elements;i++){
        data[i] = Random::uniform(x_mean,range);
    }
};

// modifies the given vector, matrix or array by applying
// the referred function to all its values
// (this function should take a single type <T> as argument)
template<typename T>
void Array<T>::function(T (*pointer_to_function)(T)){
    for (u_int32_t i=0;i<elements;i++){
        data[i]=pointer_to_function(data[i]);
    }
};

// returns the sum of all array elements
template<typename T>
T Array<T>::sum(){
    if (elements==0){return NAN;}
    T result=0;
    for (u_int32_t i=0;i<elements;i++){
        result+=data[i];
    }
    return result;
}

// returns the product of all array elements
template<typename T>
T Array<T>::product(){
    if (elements==0){return NAN;}
    T result = data[0];
    for (u_int32_t i=1;i<elements;i++){
        result*=data[i];
    }
    return result;
}

// elementwise (scalar) multiplication by given factor
template<typename T>
void Array<T>::multiply(T factor){
    for (u_int32_t i=0;i<elements;i++){
        data[i]*=factor;
    }
}

// elementwise (scalar) division by given quotient
template<typename T>
void Array<T>::divide(T quotient){
    if (quotient==0){return;}
    for (u_int32_t i=0;i<elements;i++){
        data[i]/=quotient;
    }
}

// elementwise (scalar) addition of specified value
template<typename T>
void Array<T>::add(T value){
    for (u_int32_t i=0;i<elements;i++){
        data[i]+=value;
    }
}

// elementwise (scalar) substraction of specified value
template<typename T>
void Array<T>::substract(T value){
    for (u_int32_t i=0;i<elements;i++){
        data[i]-=value;
    }
}

// elementwise (scalar) multiplication with second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::multiply(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    u_int32_t n=fmin(other.get_elements(),this->elements);
    for (u_int32_t i=0;i<n;i++){
        this->data[i]*=other.data[i];
    }
}

// elementwise (scalar) addition of second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::add(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    u_int32_t n=fmin(other.get_elements(),this->elements);
    for (u_int32_t i=0;i<n;i++){
        this->data[i]+=other.data[i];
    }
}

// elementwise (scalar) substraction of second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::substract(const Array &other){
    if (this->dimensions!=other.dimensions){return;}
    u_int32_t n=fmin(other.get_elements(),this->elements);
    for (u_int32_t i=0;i<n;i++){
        this->data[i]-=other->data[i];
    }
}

// elementwise (scalar) division by second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::divide(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    u_int32_t n=fmin(other.get_elements(),this->elements);
    for (u_int32_t i=0;i<n;i++){
        this->data[i]/=other.data[i];
    }
}

// replace all findings of given value by specified new value
template<typename T>
void Array<T>::replace(T old_value, T new_value){
    for (u_int32_t i=0;i<elements;i++){
        if (data[i]==old_value){
            data[i]=new_value;
        }
    }
}

// returns the number of occurrences of the specified value
template<typename T>
u_int32_t Array<T>::find(T value){
    u_int32_t counter=0;
    for (u_int32_t i=0;i<elements;i++){
        counter+=(data[i]==value);
    }
    return counter;
}

// elementwise (scalar) exponentiation by given power
template<typename T>
void Array<T>::pow(T exponent){
    for (u_int32_t i=0;i<elements;i++){
        data[i]=pow(data[i],exponent);
    }
}

// elementwise square root
template<typename T>
void Array<T>::sqrt(){
    for (u_int32_t i=0;i<elements;i++){
        data[i]=sqrt(data[i]);
    }
}

// assignment operator:
// copies the values from a second vector, matrix or array
// into the values of the current vector, matrix or array;
// the dimensions of target and source should match!
template<typename T>
void Array<T>::operator=(const Array<T>& other){
    u_int32_t max_dim = std::fmin(this->dimensions,other->get_dimensions());
    u_int32_t max_index;
    u_int32_t index[max_dim];
    u_int32_t element;
    for (int d=0;d<max_dim;d++){
        max_index=std::fmin(this->size[d],other->get_size(d));
        for (int i=0;i<max_index;i++){
            index[d]=i;
            this->set(index,other->data.get(index));
        }
    }
}

// returns the dotproduct of two 2d matrices
template<typename T>
Matrix<T> Matrix<T>::dotproduct(const Matrix& other){
    Matrix<T> result({this->size[0],this->size[1]});
    if (this->size[0] == other->size[1]){
        u_int32_t result_index[2];
        for (u_int32_t result_y=0;result_y<this->size[0];result_y++){
            for (u_int32_t result_x=0;result_x<other->size[1];result_x++){
                result_index={result_x,result_y};
                result.set(result_index,0);
                for (u_int32_t i=0;i<this->size[0];i++){
                    u_int32_t A_index[]={i,result_y};
                    u_int32_t B_index[]={result_x,i};
                    result.set(result_index,result.get(result_index) + this->data[this->get_element(A_index)] * other->data[other->get_element(B_index)]);
                }
            }
        }
    }
    return result;
}

// returns the matrix transpose
template<typename T>
Matrix<T> Matrix<T>::transpose(){
    Matrix<T> result({this->size[1],this->size[0]});
    for (u_int32_t x=0;x<this->size[0];x++){
        for (u_int32_t y=0;y<this->size[1];y++){
            u_int32_t source_index[] = {x,y};
            u_int32_t target_index[] = {y,x};
            result.set(target_index,this->get(source_index));
        }
    }        
    return result;
}