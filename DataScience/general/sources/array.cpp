#include "../headers/array.h"

// constructor for multi-dimensional array;
// pass dimensions (elements per dimension) as array
template<typename T>
Array<T>::Array(std::initializer_list<int> dim_size){
    init_list=dim_size;
    this->dimensions = (int)dim_size.size();
    size.resize(dimensions);
    for (int n=0, auto iterator=dim_size.begin();iterator!=dim_size.end();n++, iterator++){
        this->size[n]=*iterator;
    }
    this->elements=1;
    for (int d=0;d<dimensions;d++){
        this->elements*=std::fmax(1,size[d]);
    }
    this->data.resize(elements);
};

// constructor for 1d vector
template<typename T>
Vector<T>::Vector(const int elements){
    init_list={elements};
    this->size.resize(1);
    this->elements = elements;
    this->size[0] = elements;
    this->dimensions = 1;
    this->data.resize(elements);
}

// constructor for 2d matrix
template<typename T>
Matrix<T>::Matrix(const int elements_x, const int elements_y){
    init_list = {elements_x, elements_y};
    this->size.resize(2);
    this->elements = elements_x * elements_y;
    this->size[0] = elements_x;
    this->size[1] = elements_y;
    this->dimensions = 2;
    this->data.resize(this->elements);
}

// set a value by array index
// (for multi-dimensional arrays)
template<typename T>
void Array<T>::set(std::initializer_list<int> index, const T value){
    data[get_element(index)] = value;
};

// set a value by index for 1d vector
template<typename T>
void Vector<T>::set(const int index, const T value){
    this->data[index] = value;
};

// set a value by index for 2d matrix
template<typename T>
void Matrix<T>::set(const int index_x, const int index_y, const T value){
    this->data[this->get_element({index_x,index_y})] = value;
}

// get value from array index
// (pass array index as array)
template<typename T>
T Array<T>::get(std::initializer_list<int> index){
    int element=get_element(index);
    if (std::isnan(element) || element>elements){return NAN;}
    return data[element];
};

// get 1d element index from multidimensional index
template<typename T>
int Array<T>::get_element(std::initializer_list<int> index){
    // confirm valid number of dimensions
    if (index.size() > dimensions){
        return NAN;
    }
    // principle: result=index[0] + index[1]*size[0] + index[2]*size[0]*size[1] + index[3]*size[0]*size[1]*size[2] + ...
    static int result;
    static int add;
    result = *index.begin();
    for (int i=1, auto iterator=index.begin()+1; iterator!=index.end(); i++, iterator++){
        add = *iterator;
        for(int s=0;s<i;s++){
            add*=size[s];
        }
        result+=add;
    }
    return result;
};

// fill entire array with given value
template<typename T>
void Array<T>::fill_values(const T value){
    for (int i=0;i<elements;i++){
        data[i]=value;
    }
};

// fill array with identity matrix
template<typename T>
void Array<T>::fill_identity(){
    // get size of smallest dimension
    int max_index=__INT_MAX__;
    for (int i=0;i<dimensions;i++){
        max_index=std::fmin(max_index,size[i]);
    }
    int index[dimensions];
    for (int i=0;i<max_index;i++){
        for (int d=0;d<dimensions;d++){
            index[d]=i;
        }
        set(index,1);
    }
}

// fill with random normal distribution
template<typename T>
void Array<T>::fill_random_gaussian(const T mu, const T sigma){
    for (int i=0;i<elements;i++){
        data[i] = Random<T>::gaussian(mu,sigma);
    }
};

// fill with random uniform distribution
template<typename T>
void Array<T>::fill_random_uniform(const T min, const T max){
    for (int i=0;i<elements;i++){
        data[i] = Random<T>::uniform(min,max);
    }
};

// modifies the given vector, matrix or array by applying
// the referred function to all its values
// (this function should take a single type <T> as argument)
template<typename T>
void Array<T>::function(const T (*pointer_to_function)(T)){
    for (int i=0;i<elements;i++){
        data[i]=pointer_to_function(data[i]);
    }
};

// returns the sum of all array elements
template<typename T>
T Array<T>::sum(){
    if (elements==0){return NAN;}
    T result=0;
    for (int i=0;i<elements;i++){
        result+=data[i];
    }
    return result;
}

// returns the product of all array elements
template<typename T>
T Array<T>::product(){
    if (elements==0){return NAN;}
    T result = data[0];
    for (int i=1;i<elements;i++){
        result*=data[i];
    }
    return result;
}

// elementwise (scalar) multiplication by given factor
template<typename T>
void Array<T>::multiply(const T factor){
    for (int i=0;i<elements;i++){
        data[i]*=factor;
    }
}

// elementwise (scalar) division by given quotient
template<typename T>
void Array<T>::divide(const T quotient){
    if (quotient==0){return;}
    for (int i=0;i<elements;i++){
        data[i]/=quotient;
    }
}

// elementwise (scalar) addition of specified value
template<typename T>
void Array<T>::add(const T value){
    for (int i=0;i<elements;i++){
        data[i]+=value;
    }
}

// elementwise (scalar) substraction of specified value
template<typename T>
void Array<T>::substract(const T value){
    for (int i=0;i<elements;i++){
        data[i]-=value;
    }
}

// elementwise (scalar) multiplication with second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::multiply(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    int n=fmin(other.get_elements(),this->elements);
    for (int i=0;i<n;i++){
        this->data[i]*=other.data[i];
    }
}

// elementwise (scalar) addition of second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::add(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    int n=fmin(other.get_elements(),this->elements);
    for (int i=0;i<n;i++){
        this->data[i]+=other.data[i];
    }
}

// elementwise (scalar) substraction of second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::substract(const Array &other){
    if (this->dimensions!=other.dimensions){return;}
    int n=fmin(other.get_elements(),this->elements);
    for (int i=0;i<n;i++){
        this->data[i]-=other->data[i];
    }
}

// elementwise (scalar) division by second vector, matrix or array;
// the number of dimensions must match!
template<typename T>
void Array<T>::divide(const Array& other){
    if (this->dimensions!=other.dimensions){return;}
    int n=fmin(other.get_elements(),this->elements);
    for (int i=0;i<n;i++){
        this->data[i]/=other.data[i];
    }
}

// replace all findings of given value by specified new value
template<typename T>
void Array<T>::replace(const T old_value, const T new_value){
    for (int i=0;i<elements;i++){
        if (data[i]==old_value){
            data[i]=new_value;
        }
    }
}

// returns the number of occurrences of the specified value
template<typename T>
int Array<T>::find(const T value){
    int counter=0;
    for (int i=0;i<elements;i++){
        counter+=(data[i]==value);
    }
    return counter;
}

// elementwise (scalar) exponentiation by given power
template<typename T>
void Array<T>::pow(const T exponent){
    for (int i=0;i<elements;i++){
        data[i]=pow(data[i],exponent);
    }
}

// elementwise square root
template<typename T>
void Array<T>::sqrt(){
    for (int i=0;i<elements;i++){
        data[i]=sqrt(data[i]);
    }
}

// assignment operator:
// copies the values from a second vector, matrix or array
// into the values of the current vector, matrix or array;
// the dimensions of target and source should match!
template<typename T>
void Array<T>::operator=(const Array<T>& other){
    int max_dim = std::fmin(this->dimensions,other->get_dimensions());
    int max_index;
    int index[max_dim];
    int element;
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
        int result_index[2];
        for (int result_y=0;result_y<this->size[0];result_y++){
            for (int result_x=0;result_x<other->size[1];result_x++){
                result_index={result_x,result_y};
                result.set(result_index,0);
                for (int i=0;i<this->size[0];i++){
                    int A_index[]={i,result_y};
                    int B_index[]={result_x,i};
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
    for (int x=0;x<this->size[0];x++){
        for (int y=0;y<this->size[1];y++){
            int source_index[] = {x,y};
            int target_index[] = {y,x};
            result.set(target_index,this->get(source_index));
        }
    }        
    return result;
}
