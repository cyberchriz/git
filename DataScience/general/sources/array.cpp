#include "../headers/array.h"

// +=================================+   
// | getters & setters               |
// +=================================+

// assigns a value to an array element via its index
template<typename T>
void Array<T>::set(std::initializer_list<int> index, const T value){
    this->_data[get_element(index)] = value;
}

// assigns a value to a Vector element via its index
template<typename T>
void Vector<T>::set(const int index, const T value){
    this->_data[index] = value;
}

// assigns a value to a 2d matrix element via its index
template<typename T>
void Matrix<T>::set(const int row, const int col, const T value){
    this->_data[get_element({row,col})] = value;
}

// returns the value of an array element via its index
template<typename T>
T Array<T>::get(std::initializer_list<int> index){
    int element=get_element(index);
    if (std::isnan(element) || element>elements){return NAN;}
    return _data[element];
}

// returns the value of a vector element via its index
template<typename T>
T Vector<T>::get(const int index){
    if (index>this->_elements){return NAN;}
    return this->_data[index];
}

// returns the value of a 2d matrix element via its index
template<typename T>
T Matrix<T>::get(const int row, const int col){
    int element=get_element({row,col});
    if (std::isnan(element) || element>elements){return NAN;}
    return _data[element];
}

// returns the number of dimensions of the array
template<typename T>
int Array<T>::get_dimensions(){
    return _dimensions;
}

// returns the number of elements of the specified array dimension
template<typename T>
int Array<T>::get_size(int dimension){
    return _size[dimension];
}

// returns the total number of elements across the array
// for all dimensions
template<typename T>
int Array<T>::get_elements(){
    return _elements;
}

// assigns a value to an element of a one-dimensional vector via its index
template<typename T>
void Vector<T>::set(const int index, const T value){
    this->_data[index] = value;
};

// assigns a value to an element of a two-dimensional matrix via its index
template<typename T>
void Matrix<T>::set(const int row, const int col, const T value){
    this->_data[this->get_element({row,col})] = value;
}

// converts a multidimensional index to 1d
template<typename T>
int Array<T>::get_element(std::initializer_list<int> index){
    // confirm valid number of _dimensions
    if (index._size() > _dimensions){
        return NAN;
    }
    // principle: result=index[0] + index[1]*size[0] + index[2]*size[0]*size[1] + index[3]*size[0]*size[1]*size[2] + ...
    static int result;
    static int add;
    result = *index.begin();
    for (int i=1, auto iterator=index.begin()+1; iterator!=index.end(); i++, iterator++){
        add = *iterator;
        for(int s=0;s<i;s++){
            add*=_size[s];
        }
        result+=add;
    }
    return result;
};

// +=================================+   
// | fill, initialize                |
// +=================================+

// fill entire array with given value
template<typename T>
void Array<T>::fill_values(const T value){
    for (int i=0;i<this->_elements;i++){
        _data[i]=value;
    }
};

// fill array with identity matrix
template<typename T>
void Array<T>::fill_identity(){
    // set matrix to zeros
    this->fill_values(0);
    // get size of smallest dimension
    int max_index=__INT_MAX__;
    for (int i=0; i<this->_dimensions; i++){
        max_index=std::fmin(max_index,this->_size[i]);
    }
    int index[this->_dimensions];
    // add 'ones' of identity matrix
    for (int i=0;i<max_index;i++){
        for (int d=0;d<this->_dimensions;d++){
            index[d]=i;
        }
        set(index,1);
    }
}

// fill with random normal distribution
template<typename T>
void Array<T>::fill_random_gaussian(const T mu, const T sigma){
    for (int i=0; i<this->_elements; i++){
        this->_data[i] = Random<T>::gaussian(mu,sigma);
    }
};

// fill with random uniform distribution
template<typename T>
void Array<T>::fill_random_uniform(const T min, const T max){
    for (int i=0; i<this->_elements;i++){
        this->_data[i] = Random<T>::uniform(min,max);
    }
};

// fill with the array with zeros
template<typename T>
void Array<T>::fill_zeros(){
    this->fill_values(0);
}

// +=================================+   
// | Distribution Properties         |
// +=================================+

// returns the arrithmetic mean of all values of the array
template<typename T>
double Array<T>::mean(){
    return Sample<T>(this->_data).mean();
}

// returns the median of all values of the array
template<typename T>
double Array<T>::median(){
    return Sample<T>(this->_data).median();
}

// returns the variance of all values of the array
template<typename T>
double Array<T>::variance(){
    return Sample<T>(this->_data).variance();
}

// returns the standard deviation of all values of the array
template<typename T>
double Array<T>::stddev(){
    return Sample<T>(this->_data).stddev();
}

// +=================================+   
// | Addition                        |
// +=================================+

// returns the sum of all array elements
template<typename T>
T Array<T>::sum(){
    T result=0;
    for (int i=0; i<this->_elements; i++){
        result+=this->_data[i];
    }
    return result;
}

// elementwise addition of the specified value to all values of the array
template<typename T>
Array<T> Array<T>::operator+(const T value){
    Array<T> result(this->_init_list);
    for (int i=0;i <this->_elements; i++){
        result._data[i]=this->_data[i]+value;
    }
    return result;
}

// returns the resulting array of the elementwise addition of
// two array of equal dimensions;
// will return a NAN array if the dimensions don't match!
template<typename T>
Array<T> Array<T>::operator+(const Array& other){
    Array<T> result(this->_init_list);
    if (!equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]+other._data[i];
        }
    }
    return result;
}

// increments the values of the array by +1
template<typename T>
void Array<T>::operator++(){
    this->+=1;
}

// elementwise addition of the specified
// value to the elements of the array
template<typename T>
void Array<T>::operator+=(const T value){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=value;
    }
}

// elementwise addition of the values of the second
// array to the corresponding values of the current array;
// the dimensions of the arrays must match!
// the function will otherwise turn the source array into
// a NAN array!
template<typename T>
void Array<T>::operator+=(const Array& other){
    if (!equal_size(other)){
        this->fill_values(NAN);
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=other._data[i];
    }
}

// +=================================+   
// | Substraction                    |
// +=================================+

// elementwise substraction of the specified value from all values of the array
template<typename T>
Array<T> Array<T>::operator-(const T value){
    Array<T> result(this->_init_list);
    for (int i=0;i <this->_elements; i++){
        result._data[i]=this->_data[i]-value;
    }
    return result;
}

// returns the resulting array of the elementwise substraction of
// two array of equal dimensions (this minus other);
// will return a NAN array if the dimensions don't match!
template<typename T>
Array<T> Array<T>::operator-(const Array& other){
    Array<T> result(this->_init_list);
    if (!equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]-other._data[i];
        }
    }
    return result;
}

// decrements the values of the array by -1
template<typename T>
void Array<T>::operator--(){
    this->-=1;
}

// elementwise substraction of the specified
// value from the elements of the array
template<typename T>
void Array<T>::operator-=(const T value){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=value;
    }
}

// elementwise substraction of the values of the second
// array from the corresponding values of the current array,
// i.e. this minus other;
// the dimensions of the arrays must match!
// the function will otherwise turn the source array into
// a NAN array!
template<typename T>
void Array<T>::operator-=(const Array& other){
    if (!equal_size(other)){
        this->fill_values(NAN);
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=other._data[i];
    }
}

// +=================================+   
// | Multiplication                  |
// +=================================+

// returns the product reduction, i.e. the result
// of all individual elements of the array
template<typename T>
T Array<T>::product(){
    if (this->_elements==0){
        return NAN;
    }
    T result = this->_data[0];
    for (int i=1; i<this->_elements; i++){
        result*=this->_data[i];
    }
    return result;
}

// elementwise multiplication with a scalar
template<typename T>
Array<T> Array<T>::operator*(const T factor){
    Array<T> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]*factor;
    }
    return result;
}

// elementwise multiplication (*=) with a scalar
template<typename T>
void Array<T>::operator*=(const T factor){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]*=factor;
    }
}

// elementwise multiplication of the values of the current
// array with the corresponding values of a second array,
// resulting in the 'Hadamard product';
// the dimensions of the two arrays must match!
// the function will otherwise return a NAN array!
template<typename T>
Array<T> Array<T>::Hadamard(const Array& other){
    Array<T> result(this->_init_list);
    if(!equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]*other._data[i];
        }
    }
    return result;
}

// vector dotproduct ("scalar product")
template<typename T>
T Vector<T>::dotproduct(const Vector& other){
    if (this->_elements != other._elements){
        return NAN;
    }
    T result = 0;
    for (int i = 0; i < this->_elements; i++){
        result += _data[i] * other._data[i];
    }
    
    return result;
}

// operator* as alias for vector dotproduct
template<typename T>
T Vector<T>::operator*(const Vector& other){
    return this->dotproduct(other);
}

// returns the dotproduct of two 2d matrices
template<typename T>
Matrix<T> Matrix<T>::dotproduct(const Matrix& other){
    // Create the resulting matrix
    Matrix<T> result(_size[0], other._size[1]);
    // Check if the matrices can be multiplied
    if (this->_size[1] != other._size[0]){
        result.fill_values(NAN);
        return result;
    }
    // Compute the dot product
    for (int i = 0; i < this->_size[0]; i++) {
        for (int j = 0; j < other._size[1]; j++) {
            T sum = 0;
            for (int k = 0; k < this->_size[1]; k++) {
                sum += this->get(i, k) * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

// operator* as alias vor matrix dotproduct
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other){
    return this->dotproduct(other);
}

// +=================================+   
// | Division                        |
// +=================================+

// elementwise division by a scalar
template<typename T>
Array<T> Array<T>::operator/(const T quotient){
    Array<T> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]/quotient;
    }
    return result;
}

// elementwise division (/=) by a scalar
template<typename T>
void Array<T>::operator/=(const T quotient){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]/=quotient;
    }
}

// +=================================+   
// | Modulo                          |
// +=================================+

// elementwise modulo operation, converting the array values
// to the remainders of their division by the specified number
template<typename T>
void Array<T>::operator%=(const double num){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]%=num;
    }
}

// elementwise modulo operation, resulting in an array that
// contains the remainders of the division of the values of
// the original array by the specified number
template<typename T>
Array<double> Array<T>::operator%(const double num){
    Array<T> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]%num;
    }
    return result;
}

// +=================================+   
// | Exponentiation                  |
// +=================================+

// elementwise exponentiation to the power of
// the specified exponent
template<typename T>
void Array<T>::pow(const T exponent){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::pow(this->_data[i], exponent);
    }
}

// elementwise exponentiation to the power of
// the corresponding values of the second array;
// the dimensions of the two array must match!
// the function will otherwise return a NAN array!
template<typename T>
void Array<T>::pow(const Array& other){
    if (!equal_size(other)){
        this->fill_values(NAN);
        return;
    }
    else {
        for (int i=0; i<this->_elements; i++){
            this->_data[i]=std::pow(this->_data[i], other._data[i]);
        }
    }
}

// converts the individual values of the array
// elementwise to their square root
template<typename T>
void Array<T>::sqrt(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::sqrt(this->_data[i], exponent);
    }
}

// +=================================+   
// | Rounding                        |
// +=================================+

// rounds the values of the array elementwise
// to their nearest integers
template<typename T>
void Array<T>::round(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::round(this->_data[i]);
    }
}

// rounds the values of the array elementwise
// to their next lower integers
template<typename T>
void Array<T>::floor(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::floor(this->_data[i]);
    }
}

// rounds the values of the array elementwise
// to their next higher integers
template<typename T>
void Array<T>::ceil(){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=std::ceil(this->_data[i]);
    }
}

// +=================================+   
// | Find, Replace                   |
// +=================================+

// returns the number of occurrences of the specified value
template<typename T>
int Array<T>::find(const T value){
    int counter=0;
    for (int i=0; i<this->_elements; i++){
        counter+=(this->_data[i]==value);
    }
    return counter;
}

// replace all findings of given value by specified new value
template<typename T>
void Array<T>::replace(const T old_value, const T new_value){
    for (int i=0; i<this->_elements; i++){
        if (this->_data[i]==old_value){
            this->_data[i]=new_value;
        }
    }
}

// +=================================+   
// | Custom Functions                |
// +=================================+

// modifies the given vector, matrix or array by applying
// the referred function to all its values
// (the referred function should take a single argument of type <T>)
template<typename T>
void Array<T>::function(const T (*pointer_to_function)(T)){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=pointer_to_function(this->_data[i]);
    }
}

// +=================================+   
// | Assignment                      |
// +=================================+

// assignment operator:
// copies the values from a second vector, matrix or array
// into the values of the current vector, matrix or array;
// the _dimensions of target and source should match!
// the operation will otherwise result in a NAN array!
template<typename T>
void Array<T>::operator=(const Array<T>& other){
    if (!equal_size(other)){
        this->fill_values(NAN);
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i] = other._data[i];
    }
}

// returns an identical copy of the current array
template<typename T>
Array<T> Array<T>::copy(){
    Array<T> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i];
    }
    return result;
}

// +=================================+   
// | Elementwise Comparison by       |
// | Single Value                    |
// +=================================+

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than
// the specified value
template<typename T>
Array<bool> Array<T>::operator>(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]>value;
    }
    return result
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than or equal
// to the specified argument value
template<typename T>
Array<bool> Array<T>::operator>=(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]>=value;
    }
    return result
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are equal to the specified
// argument value
template<typename T>
Array<bool> Array<T>::operator==(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]==value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are unequal to the specified
// argument value
template<typename T>
Array<bool> Array<T>::operator!=(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]!=value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than the
// specified argument value
template<typename T>
Array<bool> Array<T>::operator<(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]<value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than or equal to
// the specified argument value
template<typename T>
Array<bool> Array<T>::operator<=(const T value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]<=value;
    }
    return result;
}

// +=================================+   
// | Elementwise Comparison by       |
// | Second Array                    |
// +=================================+

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are greater
// than the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator>(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]>other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are greater
// than or equal to the corresponding values of the
// second array, i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator>=(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]>=other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are equal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator==(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]==other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are unequal
// to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator!=(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]!=other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less
// than the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator<(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]<other._data[i];
        }
    }
    return result;
}

// returns a boolean array of equal dimensions
// to the source array, with its values indicating
// whether the values of the source array are less than
// or equal to the corresponding values of the second array,
// i.e. by elementwise comparison;
// make sure the the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator<=(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]<=other._data[i];
        }
    }
    return result;
}
// +=================================+   
// | Elementwise Logical Operations  |
// +=================================+

// returns a boolean array as the result of the
// logical AND of the source array and the specified
// boolean argument value
template<typename T>
Array<bool> Array<T>::operator&&(const bool value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]&&value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical OR of the source array and the specified
// boolean argument value
template<typename T>
Array<bool> Array<T>::operator||(const bool value){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=this->_data[i]||value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical NOT of the source array
template<typename T>
Array<bool> Array<T>::operator!(){
    Array<bool> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=!this->_data[i];
    }
    return result;
}

// returns a boolean array as the result of the
// elementwise logical AND operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator&&(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]&&other->_data[i];
        }
    }
    return result;
}

// returns a boolean array as the result of the
// elementwise logical OR operation between the
// source array and the corresponding values of the
// second array;
// make sure that the dimensions of both arrays match!
// the function will otherwise return a NAN array!
template<typename T>
Array<bool> Array<T>::operator||(const Array& other){
    Array<bool> result(this->_init_list);
    if (!this->equal_size(other)){
        result.fill_values(NAN);
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result._data[i]=this->_data[i]||other->_data[i];
        }
    }
    return result;
}

// +=================================+   
// | Type Casting                    |
// +=================================+

// type casting (explicit by default)
template<typename T>
template<typename C>
Array<T>::operator Array<C>(){
    Array<C> result(this->_init_list);
    for (int i=0; i<this->_elements; i++){
        result._data[i]=C(this->_data[i]);
    }
    return result;
}

// +=================================+   
// | Constructors & Destructors      |
// +=================================+

// constructor for multi-dimensional array;
// pass dimension size (elements per dimension)
// as an initializer_list, e.g. {3,4,4}
template<typename T>
Array<T>::Array(std::initializer_list<int> dim_size){
    init_list=dim_size;
    this->_dimensions = (int)dim_size.size();
    this->_size = new int(_dimensions);
    for (int n=0, auto iterator=dim_size.begin();iterator!=dim_size.end();n++, iterator++){
        this->_size[n]=*iterator;
    }
    this->elements=1;
    for (int d=0;d<_dimensions;d++){
        this->elements*=std::fmax(1,_size[d]);
    }
    this->_data=new T(this->_elements);
};

// destructor for parent class
template<typename T>
Array<T>::~Array(){
    delete _data;
    delete _size;
}

// constructor for a one-dimensional vector
template<typename T>
Vector<T>::Vector(const int elements){
    init_list={elements};
    this->_size = new int(1);
    this->_elements = elements;
    this->_size[0] = elements;
    this->_capacity = int(elements * (1.0+_reserve));
    this->_dimensions = 1;
    this->_data=new T(this->_capacity);
}

// constructor for 2d matrix
template<typename T>
Matrix<T>::Matrix(const int rows, const int cols){
    init_list = {rows, colss};
    this->_size = new int(2);
    this->_elements = rows * cols;
    this->_size[0] = rows;
    this->_size[1] = cols;
    this->_dimensions = 2;
    this->_data = new T(this->_elements);
}

// +=================================+   
// | Private Member Functions        |
// +=================================+

// check whether this array and a second array
// match with regard to their number of dimensions
// and their size per individual dimensions
template<typename T>
bool Array<T>::equal_size(const Array& other){
    if (this->_dimensions!=other->get_dimensions()){
        return false
    }
    for (int n=0; n<this->_dimensions; n++){
        if (this->_size[n]!=other.get_size(n)){
            return false;
        }
    }
    return true;
}    

// change the size of a simple C++ array
// by allocating new memory and copying the previous
// data to the new location
template<typename T>
void Array<T>::resizeArray(T*& arr, int newSize) {
    // Create a new array with the desired size
    T* newArr = new T[newSize];
    // Copy the elements from the old array to the new array
    for (int i = 0; i < newSize; i++) {
        if (i < sizeof(arr)/sizeof(T)) {
            newArr[i] = arr[i];
        } else {
            newArr[i] = 0;
        }
    }
    // Delete the old array
    delete[] arr;
    // Assign the new array to the old array variable
    arr = newArr;
}


// +=================================+   
// | Dynamic Vector Handling         |
// +=================================+

// push back 1 element into the Vector
template<typename T>
int Vector<T>::push_back(const T value){
    this->_elements++;
    if (this->_elements>this->_capacity){
        this->_capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(&this_data, this->_capacity);
    }
    this->_data[_elements-1]=value;
    return this->_elements;
}

// pop 1 element from the end the Vector
template<typename T>
T Vector<T>::pop(){
    this->_elements--;
    return this->_data[elements];
}

// returns the available total capacity of a vector
// without re-allocating memory
template<typename T>
int Vector<T>::get_capacity(){
    return this->_capacity;
}

// returns the current size (=number of elements)
// of the vector; equivalent to .get_elements()
// or .get_size(0);
template<typename T>
int Vector<T>::size(){
    return this->_elements;
}

// +=================================+   
// | Vector as Matrix                |
// +=================================+

// returns the vector as a single column matrix, 
// i.e. as transposition with data in rows (single column)
template<typename T>
Matrix<T> Vector<T>::transpose(){
    Matrix<T> result(this->_elements,1);
    for (int i=0; i<this->_elements; i++){
        result.set(i,0,this->_data[i]);
    }
    return result;
}

// returns a copy of the vector as a single row matrix
template<typename T>
Matrix<T> Vector<T>::asMatrix(){
    Matrix<T> result(1, this->_elements);
    for (int col=0; col<this->_elements; col++){
        result.set(0,col,this->_data[col]);
    }
    return result;
}

// +=================================+   
// | Vector Sample Analysis          |
// +=================================+

// returns a vector of integers that represent
// a ranking of the source vector
template<typename T>
Vector<int> Vector<T>::ranking(bool ascending=true){
    Vector<int> result(this->_elements);
    result._data=Sample(this->_data)::ranking(ascending);
    return result;
}

// returns an exponentially smoothed copy of the source vector
template<typename T>
Vector<T> Vector<T>::exponential_smoothing(bool as_series=false){
    Vector<T> result(this->_elements);
    result._data=Sample(this->_data)::exponential_smoothing(as_series);
    return result;
} 

// performs an augmented Dickey-Fuller unit root test
// for stationarity; a result <0.05 typically implies
// that the Null hypothesis can be rejected, i.e. the
// data of the vector are stationary
template<typename T>
double Vector<T>::Dickey_Fuller(){
    return Sample(this->_data)::Dickey_Fuller();
}

// returns a stationary transformation of the vector data;
// differencing methods:
//    integer=1,
//    logreturn=2,
//    fractional=3,
//    deltamean=4,
//    original=5
template<typename T>
Vector<T> Vector<T>::stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2){
    Vector<T> result(_elements);
    result._data=Sample(this->_data)::stationary(method,degree,fract_exponent);
    return result;
}

// returns a sorted copy of the vector
template<typename T>
Vector<T> Vector<T>::sort(bool ascending=true){
    Vector<T> result(_elements);
    result._data=Sample(this->_data)::sort(ascending);
    return result;
}

// returns a shuffled copy of the vector
template<typename T>
Vector<T> Vector<T>::shuffle(){
    Vector<T> result(_elements);
    result._data=Sample(this->_data)::shuffle();
    return result;
}

// returns a logarithmically transformed copy
// of the source vector
template<typename T>
Vector<T> Vector<T>::log_transform(){
    Vector<T> result(_elements);
    result._data=Sample(this->_data)::log_transform();
    return result;
}

// performs polynomial regression (to the specified
// power) on the vector and predicts a new value
// for a hypothetical new index value
template<typename T>
T Vector<T>::polynomial_predict(T x,int power=5){
    Sample temp(this->_data);
    temp.polynomial_regression();
    return temp.polynomial_predict(x);
}

// returns the Mean Squared Error (MSE) of polynomial
// regression to the specified power
template<typename T>
double Vector<T>::polynomial_MSE(int power=5){
    Sample temp(this->_data);
    temp.polynomial_regression(power);
    return temp.polynomial_MSE();
}

// returns whether linear regression is a good fit
// with respect to the given confidence interval
template<typename T>
bool Vector<T>::isGoodFit_linear(double threshold=0.95){
    Sample temp(this->_data);
    temp.linear_regression();
    return temp.isGoodFit(threshold);
}

// returns whether polynomial regression (to the specified
// power) is a good fit with respect to the given confidence
// interval
template<typename T>
bool Vector<T>::isGoodFit_polynomial(int power=5,double threshold=0.95){
    Sample temp(this->_data);
    temp.polynomial_regression(power);
    return temp.isGoodFit(threshold);
}

// performs linear regression and predict a new value
// for a hypothetical new index x
template<typename T>
T Vector<T>::linear_predict(T x){
    return Sample(this->_data)::linear_predict(x);
}

// returns the slope of linear regression of the vector data
template<typename T>
double Vector<T>::get_slope(){
    return Sample(this->_data)::get_slope();
} 

// returns the y-axis intercept of linear regression of the vector data
template<typename T>
double Vector<T>::get_y_intercept(){
    return Sample(this->_data)::get_y_intercept();
}

// returns the coefficient of determination (r2) of
// linear regression of the vector data
template<typename T>
double Vector<T>::get_r_squared_linear(){Sample temp(this->_data);temp.linear_regression();return temp.get_r_squared()};

// returns the coefficient of determination (r2) of
// polynomial regression of the vector data
// (to the specified power)
template<typename T>
double Vector<T>::get_r_squared_polynomial(int power=5){Sample temp(this->_data);temp.polynomial_regression(power);return temp.get_r_squared();}

// Matrix transpose
template<typename T>
Matrix<T> Matrix<T>::transpose(){
    // create a new matrix with swapped dimensions
    Matrix<T> result(this->_size[1], this->_size[0]);

    for(int i = 0; i < _size[0]; i++){
        for(int j = 0; j < _size[1]; j++){
            // swap indices and copy element to result
            result.set(j, i, get(i, j));
        }
    }
    return result;
}
