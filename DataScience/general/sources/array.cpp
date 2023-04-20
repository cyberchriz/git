#include "../headers/array.h"
#ifndef ARRAY_CPP
#define ARRAY_CPP

// +=================================+   
// | getters & setters               |
// +=================================+

// assigns a value to an array element via
// a std::initializer_list<int> index
template<typename T>
void Array<T>::set(const std::initializer_list<int>& index, const T value){
    this->_data[get_element(index)] = value;
}

// assigns a value to an array element, with index parameter
// as const std::vector<int>&
template<typename T>
void Array<T>::set(const std::vector<int>& index, const T value){
    this->_data[get_element(index)] = value;
}

// assigns a value to an element of a one-dimensional vector
// via its index (with index parameter as type const int)
template<typename T>
void Vector<T>::set(const int index, const T value){
    this->_data[index] = value;
};

// assigns a value to an element of a two-dimensional
// matrix via its index (index parameters as const int)
template<typename T>
void Matrix<T>::set(const int row, const int col, const T value){
    std::initializer_list<int> index={row,col};
    this->_data[this->get_element(index)] = value;
}

// returns the value of an array element via its index
template<typename T>
T Array<T>::get(const std::initializer_list<int>& index){
    int element=get_element(index);
    if (std::isnan(element) || element>this->_elements){return T(NAN);}
    return _data[element];
}

// returns the value of an array element via
// its index (as type const std::vector<int>&)
template<typename T>
T Array<T>::get(const std::vector<int>& index){
    int element=get_element(index);
    if (std::isnan(element) || element>this->_elements){return T(NAN);}
    return _data[element];
}

// returns the value of a vector element via its index
// (as a const int value)
template<typename T>
T Vector<T>::get(const int index){
    if (index>this->_elements){return T(NAN);}
    return this->_data[index];
}

// returns the value of a 2d matrix element via its index
template<typename T>
T Matrix<T>::get(const int row, const int col){
    std::initializer_list<int> index={row,col};
    int element=this->get_element(index);
    if (element<0 || element>this->_elements){return T(NAN);}
    return this->_data[element];
}

// returns the number of dimensions of the array
template<typename T>
int Array<T>::get_dimensions(){
    return this->_dimensions;
}

// returns the number of elements of the specified array dimension
template<typename T>
int Array<T>::get_size(int dimension){
    return this->_size[dimension];
}

// returns the total number of elements across the array
// for all dimensions
template<typename T>
int Array<T>::get_elements(){
    return this->_elements;
}

// converts a multidimensional index (represented by the values of
// a std::initializer_list<int>) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::initializer_list<int>& index) {
    // confirm valid number of _dimensions
    if (index.size() > _dimensions){
        return -1;
    }
    // check if one-dimensional
    // (note: the end pointer points to directly after(!) the last element)
    if (index.begin()+1==index.end()){
        return *index.begin();
    }
    // initialize result to number of elements belonging to last dimension
    int result = *(index.end()-1);
    // initialize iterator to counter of second last dimension
    auto iterator = index.end()-2;
    // initialize dimension index to second last dimension
    int i = this->_dimensions-2;
    // decrement iterator down to first dimension
    for (; iterator >= index.begin(); i--, iterator--){
        // initialize amount to add to count in dimension i
        int add = *iterator;
        // multiply by product of sizes of dimensions higher than i
        int s=this->_dimensions-1;
        for(; s >i; s--){
            add *= this->_size[s];
        }
        // add product to result 
        result += add;
    }
    return result;
}

// converts a multidimensional index (represented by C-style
// integer array) to a one-dimensional index (=as a scalar)
template<typename T>
int Array<T>::get_element(const std::vector<int>& index) {
    // initialize result to number of elements belonging to last dimension
    int result = index[this->_dimensions-1];
    // initialize dimension index to second last dimension
    int i = this->_dimensions-2;
    for (; i>=0;i--){
        // initialize amount to add to count in dimension i
        int add = index[i];
        // multiply by product of sizes of dimensions higher than i
        int s=this->_dimensions-1;
        for(; s > i; s--){
            add *= this->_size[s];
        }
        // add product to result;
        result += add;
    }
    return result;
}

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

// fill the array with zeros
template<typename T>
void Array<T>::fill_zeros(){
    this->fill_values(0);
}

// fills a multidimensional array with a continuous
// range of numbers (with specified start parameter
// referring to the zero position and a step parameter)
// in all dimensions
template<typename T>
void Array<T>::fill_range(const T start, const T step){
    std::vector<int> index(this->_dimensions);
    std::fill(index.begin(),index.end(),0);
    for (int d=0;d<this->_dimensions;d++){
        for (int i=0;i<this->_size[d];i++){
            index[d]=i;
            this->set(index,start+i*step);
        }
    }
}

// fills a 2d Matrix with a continuous range of numbers
// in both dimensions, from a starting point at the zero
// index, a start value and a step parameter
template<typename T>
void Matrix<T>::fill_range(const T start, const T step) {
    for (int row=0;row<this->_size[0];row++){
        for (int col=0;col<this->_size[1];col++){
            this->set(row,col,start+step*std::fmax(row,col));
        }
    }
}

// fills a vector with a continuous range of numbers
// with a start value at index 0 and a step parameter
template<typename T>
void Vector<T>::fill_range(const T start, const T step){
    for (int i=0;i<this->_elements;i++){
        this->_data[i]=start+i*step;
    }
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
std::unique_ptr<Array<T>> Array<T>::operator+(const T value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0;i <this->_elements; i++){
        result->_data[i]=this->_data[i]+value;
    }
    return result;
}

// returns the resulting array of the elementwise addition of
// two array of equal dimensions;
// will return a NAN array if the dimensions don't match!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator+(const Array& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if (!equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]+other->_data[i];
        }
    }
    return result;
}

// prefix increment operator;
// increments the values of the array by +1
template<typename T>
Array<T>& Array<T>::operator++(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=1;
    }
    return *this;
}

// postfix increment operator;
// makes an internal copy of the array,
// then increments all values of the array by +1,
// then returns the temporary copy;
// note: more overhead then with the prefix increment
// because of extra copy!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator++(int){
    auto temp = this->copy();
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=1;
    }
    return temp;
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
        this->fill_values(T(NAN));
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]+=other->_data[i];
    }
}

// +=================================+   
// | Substraction                    |
// +=================================+

// elementwise substraction of the specified value from all values of the array
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator-(const T value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0;i <this->_elements; i++){
        result->_data[i]=this->_data[i]-value;
    }
    return result;
}

// returns the resulting array of the elementwise substraction of
// two array of equal dimensions (this minus other);
// will return a NAN array if the dimensions don't match!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator-(const Array& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if (!equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]-other->_data[i];
        }
    }
    return result;
}

// prefix decrement operator;
// first decrements the values of the array by -1,
// then returns the modified array
template<typename T>
Array<T>& Array<T>::operator--(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=1;
    }
    return *this;
}

// postfix decrement operator;
// makes an internal copy of the array,
// then decrements all values of the array by -1,
// then returns the temporary copy;
// note: more overhead then with the prefix decrement
// because of extra copy!
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator--(int){
    auto temp = this->copy();
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=1;
    }
    return temp;
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
        this->fill_values(T(NAN));
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i]-=other->_data[i];
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
        return T(NAN);
    }
    T result = this->_data[0];
    for (int i=1; i<this->_elements; i++){
        result*=this->_data[i];
    }
    return result;
}

// elementwise multiplication with a scalar
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator*(const T factor){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]*factor;
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
std::unique_ptr<Array<T>> Array<T>::Hadamard(const Array& other){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    if(!equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]*other->_data[i];
        }
    }
    return result;
}

// vector dotproduct ("scalar product")
template<typename T>
T Vector<T>::dotproduct(const Vector& other){
    if (this->_elements != other->_elements){
        return T(NAN);
    }
    T result = 0;
    for (int i = 0; i < this->_elements; i++){
        result += this->_data[i] * other->_data[i];
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
std::unique_ptr<Matrix<T>> Matrix<T>::dotproduct(const Matrix& other){
    // Create the resulting matrix
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_size[0], other->_size[1]);
    // Check if the matrices can be multiplied
    if (this->_size[1] != other->_size[0]){
        result->fill_values(T(NAN));
        return result;
    }
    // Compute the dot product
    for (int i = 0; i < this->_size[0]; i++) {
        for (int j = 0; j < other->_size[1]; j++) {
            T sum = 0;
            for (int k = 0; k < this->_size[1]; k++) {
                sum += this->get(i, k) * other->get(k, j);
            }
            result->set(i, j, sum);
        }
    }
    return result;
}

// operator* as alias vor matrix dotproduct
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::operator*(const Matrix& other){
    return this->dotproduct(other);
}

// +=================================+   
// | Division                        |
// +=================================+

// elementwise division by a scalar
template<typename T>
std::unique_ptr<Array<T>> Array<T>::operator/(const T quotient){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]/quotient;
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
std::unique_ptr<Array<double>> Array<T>::operator%(const double num){
    std::unique_ptr<Array<double>> result = std::make_unique<Array<double>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]%num;
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
        this->fill_values(T(NAN));
        return;
    }
    else {
        for (int i=0; i<this->_elements; i++){
            this->_data[i]=std::pow(this->_data[i], other->_data[i]);
        }
    }
}

// converts the individual values of the array
// elementwise to their square root
template<typename T>
void Array<T>::sqrt(){
    for (int i=0; i<this->_elements; i++){
        this->_data[i]=std::sqrt(this->_data[i]);
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

// assignment operator with second Array as argument:
// copies the values from a second vector, matrix or array
// into the values of the current vector, matrix or array;
// the _dimensions of target and source should match!
// the operation will otherwise result in a NAN array!
template<typename T>
void Array<T>::operator=(const Array<T>& other){
    if (!equal_size(other)){
        this->fill_values(T(NAN));
        return;
    }
    for (int i=0; i<this->_elements; i++){
        this->_data[i] = other->_data[i];
    }
}

// assignment operator with single numeric argument:
// invokes the fill_values(const T value) method
template<typename T>
void Array<T>::operator=(T){
    this->fill_values(value);
}

// returns an identical copy of the current array
template<typename T>
std::unique_ptr<Array<T>> Array<T>::copy(){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator>(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]>value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are greater than or equal
// to the specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator>=(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]>=value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are equal to the specified
// argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator==(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]==value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are unequal to the specified
// argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator!=(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]!=value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than the
// specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]<value;
    }
    return result;
}

// returns a boolean array of equal dimensions,
// with its values indicating whether the values
// of the source array are less than or equal to
// the specified argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator<=(const T value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]<=value;
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
std::unique_ptr<Array<bool>> Array<T>::operator>(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]>other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator>=(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]>=other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator==(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]==other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator!=(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]!=other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator<(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]<other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator<=(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]<=other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator&&(const bool value){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]&&value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical OR of the source array and the specified
// boolean argument value
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator||(const bool value){
    std::unique_ptr<Array<bool>> result(new Array(this->_size));
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=this->_data[i]||value;
    }
    return result;
}

// returns a boolean array as the result of the
// logical NOT of the source array
template<typename T>
std::unique_ptr<Array<bool>> Array<T>::operator!(){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=!this->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator&&(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]&&other->_data[i];
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
std::unique_ptr<Array<bool>> Array<T>::operator||(const Array& other){
    std::unique_ptr<Array<bool>> result = std::make_unique<Array<bool>>(this->_size);
    if (!this->equal_size(other)){
        result->fill_values(T(NAN));
    }
    else {
        for (int i=0; i<this->_elements; i++){
            result->_data[i]=this->_data[i]||other->_data[i];
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
    Array<C> result = std::make_unique<Array<C>>(this->_size);
    for (int i=0; i<this->_elements; i++){
        result->_data[i]=C(this->_data[i]);
    }
    return result;
}

// +=================================+   
// | Class Conversion                |
// +=================================+

// flattens an array or matrix into a one-dimensional vector
template<typename T>
std::unique_ptr<Vector<T>> Array<T>::flatten(){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=this->_data;
    return result;
}

// converts an array into a 2d matrix of the specified size;
// if the new matrix has less elements in any of the dimensions,
// the surplus elements of the source array will be ignored;
// if the new matrix has more elements than the source array, these
// additional elements will be initialized with the specified value
template<typename T>
std::unique_ptr<Matrix<T>> Array<T>::asMatrix(const int rows, const int cols, T init_value){
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill_values(init_value);
    std::vector<int> index(this->_dimensions);
    // reset the indices of higher dimensions to all zeros
    for (int d=0;d<this->_dimensions;d++){
        index[d]=0;
    }
    // iterate over first and second dimension, i.e. keeping
    // the higher dimensions at index zero
    for (int row=0;row<this->_size[0];row++){
        index[0]=row;
        for (int col=0;col<this->_size[1];col++){
            index[1]=col;
            result->set(row,col,this->get(index));
        }
    }
    return result;
}

// converts a matrix into another 2d matrix of different size;
// if the new matrix has less elements in any of the two dimensions,
// the surplus elements of the source will be ignored; if the new matrix
// has more elements, these additional elements will be initialized with
// the specified value
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::asMatrix(const int rows, const int cols, T init_value){
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill_values(init_value);
    for (int r=0;r<std::fmin(rows,this->_size[0]);r++){
        for (int c=0;c<std::fmin(cols,this->_size[1]);c++){
            result->set(r,c,this->get(r,c));
        }
    }
    return result;
}

// converts a vector into a 2d matrix of the specified size;
// the extra rows will be initialized with zeros
// if the new matrix has less columns than the source vector
// has elements, the surplus elements (for row 0) of the source
// will be ignored; if the new matrix has more columns than the
// source vector has elements, these additional elements will
// be initialized with the specified init value;
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::asMatrix(const int rows, const int cols, T init_value){
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(rows,cols);
    result->fill_values(init_value);
    for (int i=0;i<std::fmin(this->_elements,cols);i++){
        result->set(0,i, this->_data[i]);
    }
    return result;
}

// converts an array, matrix or vector into a 2d matrix;
// the exact behavior will depend on the source dimensions:
// 1. if the source array already is 2-dimensional, the total size
// and the size per dimension will remain unchanged, only the
// datatype of the returned object is now 'Matrix<T>'
// 2. if the source has more than 2 dimensions, only values from
// index 0 of the higher dimensions will be copied into the
// returned result
template<typename T>
std::unique_ptr<Matrix<T>> Array<T>::asMatrix(){
    std::unique_ptr<Matrix<T>> result;;
    if (this->_dimensions==2){
        result=std::make_unique<Matrix<T>>(this->_size[0],this->_size[1]);
        result->_data=this->_data;
    }
    else {
        result=std::make_unique<Matrix<T>>(this->_size[0],this->_size[1]);
        std::vector<int> index(this->_dimensions);
        // reset dimension indices to all zeros
        std::fill(index.begin(),index.end(),0);
        // iterate over first and second dimension, i.e. keeping
        // the higher dimensions at index zero
        for (int row=0;row<this->_size[0];row++){
            index[0]=row;
            for (int col=0;col<this->_size[1];col++){
                index[1]=col;
                // assign matching elements with values from source
                result->set(row,col,this->get(index));
            }
        }
    }
    return result;
}

// converts a vector into a single row matrix with
// unchanged number of elements
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::asMatrix(){
    std::unique_ptr<Matrix<T>> result;;
    result=std::make_unique<Matrix<T>>(1,this->_elements);
    result->_data=this->_data;
    return result;
}

// converts a vector or matrix into an array
// or converts a preexisting array into an array of
// the specified new size;
// surplus elements of the source that go beyond the
// limits of the target will be cut off; if the target
// is bigger, the surplus target elements that have no
// corresponding index at the source will be initialized
// with the specified init_value
template<typename T>
std::unique_ptr<Array<T>> Array<T>::asArray(const std::initializer_list<int>& init_list, T init_value){
    std::unique_ptr<Array<T>> result = std::make_unique<Array<T>>(init_list);
    result->fill_values(init_value);
    // reset result index to all zeros
    std::vector<int> result_index(result->get_dimensions());
    std::fill(result_index.begin(),result_index.end(),0);
    // reset source index to all zeros
    std::vector<int> source_index(this->_dimensions);
    std::fill(source_index.begin(),source_index.end(),0);
    // iterate over source dimensions
    for (int d=0;d<std::fmin(this->_dimensions,result->get_dimensions());d++){
        // iterate over elements of given dimension
        for (int i=0;i<std::fmin(this->_size[d],result->get_size(d));i++){
            result_index[d]=i;
            source_index[d]=i;
            result->set(result_index,this->_data[get_element(source_index)]);
        }
    }
    return result;
}

// +=================================+   
// | Constructors & Destructors      |
// +=================================+

// constructor for multi-dimensional array:
// pass dimension size (elements per dimension)
// as an initializer_list, e.g. {3,4,4}
template<typename T>
Array<T>::Array(const std::initializer_list<int>& init_list) {
    // set dimensions + check if init_list empty
    this->_dimensions = (int)init_list.size();
    if (this->_dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> _size member variable
    this->_size.reserve(_dimensions);
    auto iterator=init_list.begin();
    int n=0;
    for (; iterator!=init_list.end();n++, iterator++){
        this->_size.push_back(*iterator);
    }
    // count total number of elements
    this->_elements=1;
    for (int d=0;d<_dimensions;d++){
        this->_elements*=this->_size[d];
    }
    // create data buffer
    this->_data=new T[this->_elements];
};

// constructor for multidimensional array:
// pass dimension size (elements per dimension)
// as type std::vector<int>
template<typename T>
Array<T>::Array(const std::vector<int>& dimensions){
    // set dimensions + check if init_list empty
    this->_dimensions = dimensions.size();
    if (this->_dimensions==0){
        return;
    }
    // store size of individual dimensions in std::vector<int> _size member variable
    // and count total number of elements
    this->_size.reserve(this->_dimensions);
    this->_elements=1;
    for (int i=0;i<this->_dimensions;i++){
        this->_size.push_back(dimensions[i]);
        this->_elements*=dimensions[i];
    }
    // create data buffer
    this->_data=new T[this->_elements];    
}

// destructor for parent class
template<typename T>
Array<T>::~Array(){
    // note: the following condition tries to avoid deleting
    // memory that has never been allocated (could happen if the
    // constructor came with an empty of or invalid initializer list)
    if (this->_dimensions>0){
        delete[] _data;
    }
}

// constructor for a one-dimensional vector
template<typename T>
Vector<T>::Vector(const int elements) {
    this->_size.push_back(elements);
    this->_elements = elements;
    this->_capacity = (1.0f+this->_reserve)*elements;
    this->_dimensions = 1;
    this->_data=new T[this->_capacity];
}

// constructor for 2d matrix
template<typename T>
Matrix<T>::Matrix(const int rows, const int cols) {
    this->_size.reserve(2);
    this->_elements = rows * cols;
    this->_size.push_back(rows);
    this->_size.push_back(cols);
    this->_dimensions = 2;
    this->_data = new T[this->_elements];
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
        return false;
    }
    for (int n=0; n<this->_dimensions; n++){
        if (this->_size[n]!=other->get_size(n)){
            return false;
        }
    }
    return true;
}    

// change the size of a simple C++ array
// by allocating new memory and copying the previous
// data to the new location
template<typename T>
void Array<T>::resizeArray(T*& arr, const int newSize) {
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
        resizeArray(&this->_data, this->_capacity);
    }
    this->_data[this->_elements-1]=value;
    return this->_elements;
}

// resize the vector to a new number of elements
template<typename T>
void Vector<T>::resize(const int new_size){
    this->_elements=new_size;
    if (this->_elements>this->_capacity){
        this->_capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(&this->_data, this->_capacity);
    }    
}
// grows the vector size by the specified number of
// additional elements and initializes these new elements
// to the specified value (default=0);
// will only re-allocate memory if the new size exceeds
// the capacity; returns the new total number of elements
template<typename T>
int Vector<T>::grow(const int additional_elements,T value){
    if (additional_elements<1){return 0;}
    int new_size=this->_elements+additional_elements;
    // re-allocate memory if the new size exceeds the capacity
    if (new_size>this->_capacity){
        this->capacity=int(this->_elements*(1.0+this->_reserve));
        resizeArray(&this->_data, this->_capacity);
    }
    // initialize the new elements
    for (int i=this->_elements;i<new_size;i++){
        this->_data[i]=value;
    }
    this->_elements=new_size;
    return new_size;
}

// shrinks the vector size by the specified number of
// elements and returns the resulting new number of
// remaining total elements
template<typename T>
int Vector<T>::shrink(const int remove_amount){
    int new_size=std::fmax(0,this->_elements-remove_amount);
    this->_elements=new_size;
    return new_size;
}

// pop 1 element from the end the Vector
template<typename T>
T Vector<T>::pop(){
    this->_elements--;
    return this->_data[this->_elements];
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
// | Vector Transpose                |
// +=================================+

// returns the vector as a single column matrix, 
// i.e. as transposition with data in rows (single column)
template<typename T>
std::unique_ptr<Matrix<T>> Vector<T>::transpose(){
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_elements,1);
    for (int i=0; i<this->_elements; i++){
        result->set(i,0,this->_data[i]);
    }
    return result;
}

// +=================================+   
// | Vector Sample Analysis          |
// +=================================+

// returns a vector of integers that represent
// a ranking of the source vector
template<typename T>
std::unique_ptr<Vector<int>> Vector<T>::ranking(bool ascending){
    std::unique_ptr<Vector<int>> result = std::make_unique<Vector<int>>(this->_elements);
    result->_data=Sample<T>(this->_data).ranking(ascending);
    return result;
}

// returns an exponentially smoothed copy of the source vector
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::exponential_smoothing(bool as_series){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=Sample<T>(this->_data).exponential_smoothing(as_series);
    return result;
} 

// performs an augmented Dickey-Fuller unit root test
// for stationarity; a result <0.05 typically implies
// that the Null hypothesis can be rejected, i.e. the
// data of the vector are stationary
template<typename T>
double Vector<T>::Dickey_Fuller(){
    return Sample<T>(this->_data).Dickey_Fuller();
}

// returns a stationary transformation of the vector data;
// differencing methods:
//    integer=1,
//    logreturn=2,
//    fractional=3,
//    deltamean=4,
//    original=5
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::stationary(DIFFERENCING method,double degree,double fract_exponent){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=Sample<T>(this->_data).stationary(method,degree,fract_exponent);
    return result;
}

// returns a sorted copy of the vector
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::sort(bool ascending){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=Sample<T>(this->_data).sort(ascending);
    return result;
}

// returns a shuffled copy of the vector
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::shuffle(){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=Sample<T>(this->_data).shuffle();
    return result;
}

// returns a logarithmically transformed copy
// of the source vector
template<typename T>
std::unique_ptr<Vector<T>> Vector<T>::log_transform(){
    std::unique_ptr<Vector<T>> result = std::make_unique<Vector<T>>(this->_elements);
    result->_data=Sample<T>(this->_data).log_transform();
    return result;
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then performs polynomial regression (to the specified
// power) predicts a new value for a hypothetical new index value
template<typename T>
T Vector<T>::polynomial_predict(T x,int power){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }
    Sample2d<T> temp(x_indices, this->_data);
    temp.polynomial_regression();
    return temp.polynomial_predict(x);
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns the Mean Squared Error (MSE) of polynomial
// regression to the specified power
template<typename T>
double Vector<T>::polynomial_MSE(int power){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }    
    Sample2d<T> temp(x_indices, this->_data);
    temp.polynomial_regression(power);
    return temp.polynomial_MSE();
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns whether linear regression is a good fit
// with respect to the given confidence interval
template<typename T>
bool Vector<T>::isGoodFit_linear(double threshold){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    Sample2d<T> temp(x_indices, this->_data);
    temp.linear_regression();
    return temp.isGoodFit(threshold);
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values, then
// returns whether polynomial regression (to the specified
// power) is a good fit with respect to the given confidence
// interval
template<typename T>
bool Vector<T>::isGoodFit_polynomial(int power,double threshold){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    Sample2d<T> temp(x_indices, this->_data);
    temp.polynomial_regression(power);
    return temp.isGoodFit(threshold);
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values, then
// performs linear regression and predicts a new value
// for a hypothetical new index x
template<typename T>
T Vector<T>::linear_predict(T x){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    return Sample2d<T>(x_indices, this->_data).linear_predict(x);
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns the slope of linear regression
template<typename T>
double Vector<T>::get_slope(){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    return Sample2d<T>(x_indices, this->_data).get_slope();
} 

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns the y-axis intercept of linear regression
template<typename T>
double Vector<T>::get_y_intercept(){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    return Sample2d<T>(x_indices, this->_data).get_y_intercept();
}

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns the coefficient of determination (r2) of
// linear regression
template<typename T>
double Vector<T>::get_r_squared_linear(){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    Sample2d<T> temp(x_indices, this->_data);
    temp.linear_regression();
    return temp.get_r_squared();
};

// interprets the vector data as series data
// with evenly spaced integer scaling of the x-axis
// and the vector data as the corresponding y-values,
// then returns the coefficient of determination (r2)
// of polynomial regression to the specified power
template<typename T>
double Vector<T>::get_r_squared_polynomial(int power){
    T x_indices[this->_elements];
    for (int i=0;i<this->_elements;i++){
        x_indices[i]=i;
    }        
    Sample2d<T> temp(x_indices, this->_data);
    temp.polynomial_regression(power);
    return temp.get_r_squared();
}

// Matrix transpose
template<typename T>
std::unique_ptr<Matrix<T>> Matrix<T>::transpose(){
    // create a new matrix with swapped dimensions
    std::unique_ptr<Matrix<T>> result = std::make_unique<Matrix<T>>(this->_size[1], this->_size[0]);

    for(int i = 0; i < this->_size[0]; i++){
        for(int j = 0; j < this->_size[1]; j++){
            // swap indices and copy element to result
            result->set(j, i, get(i, j));
        }
    }
    return result;
}

// +=================================+   
// | Output                          |
// +=================================+

// prints the Vector/Matrix/Array to the console
template<typename T>
void Array<T>::print(std::string comment, std::string delimiter, std::string line_break, bool with_indices){
    if (comment!=""){
        std::cout << comment << std::endl;
    }

    if (this->_dimensions==1){
        // iterate over elements
        for (int i=0;i<this->_elements;i++){
            // add indices
            if (with_indices){
                std::cout << "[" << i << "]=";
            }
            // add value
            std::cout << this->_data[i];
            // add delimiter between elements (except after last value in row)
            if (i != this->_elements-1) {
                std::cout << delimiter;
            }         
        }
        // add line break character(s) to end of the row
        std::cout << line_break;        
        return;       
    }

    // create a vector for temporary storage of the current index (needed for indexing dimensions >=2);
    std::vector<int> index(this->_dimensions,0);
    std::fill(index.begin(),index.end(),0);   

    if (this->_dimensions==2){
        // iterate over rows
        for (int row=0; row < (this->_dimensions==1 ? 1 : this->_size[0]); row++) {
            // iterate over columns
            for (int col=0; col < (this->_dimensions==1 ? this->_size[0] : this->_size[1]); col++) {                
                // add indices
                if (with_indices) {
                    std::cout << "[" << row << "]" << "[" << col << "]=";
                }
                // add value
                index[0]=row; index[1]=col;
                std::cout << this->get(index);
                // add delimiter between columns (except after last value in row)
                if (col != this->_size[1]-1) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;            
        }
    }

    else { //=dimensions >=2
        // iterate over rows
        for (int row = 0; row < this->_size[0]; row++) {
            index[0] = row;
            // iterate over columns
            for (int col = 0; col < this->_size[1]; col++) {
                index[1] = col;
                // add opening brace for column
                std::cout << "{";
                // iterate over higher dimensions
                for (int d = 2; d < this->_dimensions; d++) {
                    // add opening brace for dimension
                    std::cout << "{";
                    // iterate over entries in the current dimension
                    for (int i = 0; i < this->_size[d]; i++) {
                        // update index
                        index[d] = i;
                        // add indices
                        if (with_indices) {
                            for (int dd = 0; dd < this->_dimensions; dd++) {
                                std::cout << "[" << index[dd] << "]";
                            }
                            std::cout << "=";
                        }
                        // add value
                        std::cout << this->get(index);
                        // add delimiter between values
                        if (i != this->_size[d] - 1) {
                            std::cout << delimiter;
                        }
                    }
                    // add closing brace for the current dimension
                    std::cout << "}";
                    // add delimiter between dimensions
                    if (d != this->_dimensions - 1) {
                        std::cout << delimiter;
                    }
                }
                // add closing brace for column
                std::cout << "}";
                // add delimiter between columns
                if (col != this->_size[1] - 1) {
                    std::cout << delimiter;
                }
            }
            // add line break character(s) to end of current row
            std::cout << line_break;
        }
    }
}

// helper function to convert an array to
// a std::initializer_list<int>
template<typename T>
std::initializer_list<int> Array<T>::array_to_initlist(int* arr, int size) {
    return {arr, arr + size};
}  

// helper function to convert a std::initializer_list<int>
// to a one-dimensional integer array
template<typename T>
std::unique_ptr<int[]> Array<T>::initlist_to_array(const std::initializer_list<int>& lst) {
    std::unique_ptr<int[]> arr(new int[lst.size()]);
    int i = 0;
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        arr[i++] = *it;
    }
    return arr;
}


#endif