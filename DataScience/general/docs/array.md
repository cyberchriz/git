[[return to main page]](../../../README.md)
# Custom Vectors, 2d Matrices and n-dimensional Arrays
usage: `#include <array.h>` or include as part of `<datascience.h>`

description:
- this class implements numeric vectors, matrices and arrays
- 'under the hood' these are simple one-dimensional arrays
- a multidimensional index will be internally converted into a 1d index
- the main benefit of the class comes with the wide variety of mathematic functions that can be applied to these arrays

# `class Array`
An instance of the class can be created by passing its dimensions as type `std::initializer_list<int>` to the constructor.
For example, for a 3x3x4 array of type `<double>` we can write:
```
Array<double> myArr{3,3,4};
```
Public Methods:


### Getters & Setters

|method|description|
|------|-----------|
| `Array<T>::set(std::initializer_list<int> index, T value)` | assigns the value at the given index (with the index in curly braces) |
| `Array<T>::get(std::intializer_list<int> index)` | returns the value at the given index (with the index in curly braces) |
| `Array<T>::get_dimensions()` | returns an integer array that holds the array dimensions |
| `Array<T>::get_size(int dimension)` | returns the size of the specified dimension as an integer |
| `Array<T>::get_elements()` | returns the total number of elements of the entire array (all dimensions) |


### Fill & Initialize

|method|description|
|------|-----------|
| `Array<T>::fill_values(T value)` | fills the entire array with the specified value (of datatype T) |
| `Array<T>::fill_zeros()` | fills the entire array with zeros |
| `Array<T>::fill_identity()` | applies the identity matrix to the array |
| `Array<T>::fill_random_gaussian(T mu=0, T sigma=1)` | fills the array with random values from a gaussian normal distribution with the given mean and sigma |
| `Array<T>::fill_random_uniform(T min=0,T max=1.0)` | fills the array with values from a random uniform distribution within the given range |


### Distribution Properties

|method|description|
|------|-----------|
| `Array<T>::mean()` | returns the arrithmetic mean of all values of the array (all dimensions) |
| `Array<T>::median()` | returns the Median of all values of the array (all dimensions) |
| `Array<T>::variance()` | returns the variance of all values of the array (all dimensions) |
| `Array<T>::stddev()` | returns the standard deviation of all values of the array (all dimensions) |


### Addition

|method|description|
|------|-----------|
| `Array<T>::sum()` | returns the sum of all values of the array (all dimensions) |
| `Array<T> operator+(const T value)` | elementwise (scalar) addition of the specified value to all values of the array |
| `Array<T>::operator+(const Array& other)` | elementwise (scalar) addition of a second array of equal dimensions to the current array |
| `operator++()` | increment all values (elementwise) by +1 |
| `operator+=(const T value)`| elementwise (scalar) addition of the specified value to all values of the array|
| `operator+=(const Array& other)` | elementwise (scalar) addition of a second array of equal dimensions to the current array|


### Substraction

|method|description|
|------|-----------|
|`Array<T> operator-(const T value)` | elementwise (scalar) substraction of the specified value from all values of the array |
| `Array<T>::operator-(const Array& other)` | elementwise substraction of a second array of equal dimensions from the current array |
|`operator--()`| decrement all values (elementwise) by -1 |
|`operator-=(const T value)` | elementwise (scalar) substraction of the specified value from all values of the array|
|`operator-=(const Array& other)`| elementwise (scalar) substraction of a second array of equal dimensions from the current array|


### Multiplication

|method|description|
|------|-----------|
| `Array<T>::product()` | returns the product of all values of the array |
| `Array<T>::operator*(const T factor)` | elementwise multiplication of all values of the array with the given factor |
| `operator*=(const T factor)` | elementwise (scalar) multiplication of the individual values of the array by the specified factor|
| `Array<T>::Hadamard(const Array& other)` | elementwise (scalar) multiplication with a second array of equal dimensions, i.e. the Hadamard product |


### Division

|method|description|
|------|-----------|
| `Array<T>::operator/(T quotient)` | divides all values of the array elementwise by the given quotient |
| `operator/=(const T quotient)` | elementwise (scalar) division of the individual values of the array by the specified quotient|


### Modulo

|method|description|
|------|-----------|
|`operator%=(const double num)`| sets all values of the array to the remainder of their division by the given number |
| `Array<double> operator%(const double num)`| returns an Array of type `<double>`that holds the remainders of the division of the original array by the given number |


### Exponentiation

|method|description|
|------|-----------|
| `Array<T>::pow(const T exponent)` | exponentiates all values of the array by the specified power |
| `Array<T> pow(const Array& other)` | elementwise exponentiation of the values of the original array to the power of the corresponding values of the second array |
| `Array<T>::sqrt()` | applies the square roots to all values of the array |


### Rounding

|method|description|
|------|-----------|
| `Array<T>::round()`| rounds the values of the array elementwise to their nearest integer|
| `ArrayyT>::floor()`| rounds the values of the array elementwise to their next lower integers|
| `ArrayyT>::ceil()`| rounds the values of the array elementwise to their next higher integers|

### Find, Replace

|method|description|
|------|-----------|
| `Array<T>::replace(const T old_value, const T new_value)` | replaces all findings of the specified old value by the specified new value |
| `int Array<T>::find(const T value)` | returns the number of findings of the specified value across the entire array (all dimensions) |


### Custom Functions

|method|description|
|------|-----------|
| `Array<T>::function(const T (*pointer_to_function)(T))` | applies the specified function to all values of the array (elementwise) |


### Assignment

|method|description|
|------|-----------|
| `Array<T>::operator=(const Array& other)` | copies the values of the specified second array (of equal dimensions) into the current array |
| `Array<T> copy()`| returns an identical copy of the original array |


### Elementwise Comparison By Single Value

|method|description|
|------|-----------|
| `Array<bool> operator>(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than the given value |
| `Array<bool> operator>=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than or equal to the given value |
| `Array<bool> operator==(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is equal to the given value |
| `Array<bool> operator!=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is unequal to the given value |
| `Array<bool> operator<(const T value)`|returns a boolean array with weach value representing whether the corresponding value in the original array is less than the given value |
| `Array<bool> operator<=(const T value)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than or equal to the given value |


### Elementwise Comparison With Second Array

|method|description|
|------|-----------|
// elementwise comparison with second array
| `Array<bool> operator>(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than the corresponding value in the second array |
| `Array<bool> operator>=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is greater than or equal to the corresponding value in the second array |
| `Array<bool> operator==(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is equal to the corresponding value in the second array |
| `Array<bool> operator!=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is unequal to the corresponding value in the second array |
| `Array<bool> operator<(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than the corresponding value in the second array |
| `Array<bool> operator<=(const Array& other)`|returns a boolean array with each value representing whether the corresponding value in the original array is less than or equal to the corresponding value in the second array |


### Elementwise Logical Operations

|method|description|
|------|-----------|
| `Array<bool> operator&&(const bool value)`|returns a boolean array with each value representing the logical `AND` between the corresponding values of the original array and the given value |
| `Array<bool> operator||(const bool value)`|returns a boolean array with each value representing the logical `OR` between the corresponding values of the original array and the given value |
| `Array<bool> operator!()`|returns a boolean array with each value representing the logical `NOT` of the corresponding values of the original array |
| `Array<bool> operator&&(const Array& other)`|returns a boolean array with each value representing the logical `AND` between the corresponding values of the original array and a second array |
| `Array<bool> operator||(const Array& other)`|returns a boolean array with each value representing the logical `OR` between the corresponding values of the original array and a second array |


### Type Casting

|method|description|
|------|-----------|
| `template<typename C> operator Array<C>()`| returns an explicit type cast of the original array |

___

# `class Matrix`
This is a derived class of `class Array<T>`. It inherits all its methods, but deals with the specific case of 2d arrays.
The constructor is slightly different: because the dimensions are already known to be 2d, they don't need to be passed in as a `std::initializer_list`, but simply by passing the size of the x and y dimensions instead.
The `.get()` method equally only needs two integers for the x and y indices it refers to.
The `.set()` method needs two integers for the x and y indices, followed by the assigned value as a third argument.

For example a 2d matrix of size 4x4 and data type `<float>` can be instantiated like this:
```
Matrix<float> myMatrix(4,4);
```
On top of all the methods given above, inherited from `class Array<T>`, the Matrix class additionally implements:


|method|description|
|------|-----------|
|`Matrix<T>::dotproduct(const Matrix& other)`|returns the resulting new `Matrix<T>` given by the dotproduct of the current matrix and a second matrix|
|`Matrix<T>::operator*(const Matrix& other)`| alias method for the dotproduct|
|`Matrix<T>::transpose()`|returns the transpose of the current matrix as the resulting new `Matrix<T>`|

___

# `class Vector`
This is also a derived class of `class Array<T>` and it also inherits all its methods, but deals with the specific case of 1d arrays.
The constructor is slightly different: because the dimensions are already known to be 1d, only the number of elements is required.
The `.get()` method equally only needs a single integer for the index it refers to.
The `.set()` methods needs an integer for the index it refers to, followed by the assigned value as a second argument.

For example a 1d `Vector` with 100 elements of type `<double>` can be instantiated like this:
```
Vector<double> myVec(100);
```

### Vector Getters & Setters

|method|description|
|------|-----------|
| `set(const int index, const T value)` | assigns the given value to the element at the specified index |
| `T get(const int index)`| returns the value of the element at the specified index |


### Dynamic Vector Handling

|method|description|
|------|-----------|
| `int Vector<T>::push_back(T value)` | adds 1 element and assigns its value; returns the resulting number of elements |
| `int Vector<T>::grow(const int additional_elements,T value=0)`|grows the vector size by the specified number of additional elements and initializes these new elements to the specified value (default=0); will only re-allocate memory if the new size exceeds the capacity; returns the resulting new total number of elements|
| `int Vector<T>::shrink(const int remove_amount)`|shrinks the vector size by the specified number of elements and returns the resulting new number of remaining total elements|
| `T Vector<T>::pop()` | removes the last element and returns its value |
| `int Vector<T>::get_capacity()`| returns the available capacity without memory re-allocations |
| `int Vector<T>::size()`| returns the number of elements; equivalent to `int Vector<T>::get_elements()`|
| `Vector<T>::resize(const int new_size)`| changes the vector size to the new number of elements; re-allocation only takes place if the new size exceeds the capacity|


### Special Vector Operations

|method|description|
|------|-----------|
| `T Vector<T>::dotproduct(const Vector& other)`| returns the dotproduct |
| `T Vector<T>::operator*(const Vector& other)`| alias method for the dotproduct |
| `Matrix<T> Vector<T>::asMatrix()` | converts the vector as a single row 'horizontal' matrix (all data as columns) |
| `Matrix<T> Vector<T>::transpose()`| converts the vector to a single column 'vertical' matrix (all data as rows)|


### Vector Sample Analysis Methods (=Implementations from `<sample.h>`)

|method|description|
|------|-----------|
| `Vector<int> ranking(bool ascending=true)`| returns an integer Vector that holds a ranking of the corresponding values contained in the original Vector |
| `Vector<T> exponential_smoothing(bool as_series=false)`| returns an exponentially smoothed copy of the original Vector |
| `double Dickey_Fuller()`| returns the result of an augmented Dickey-Fuller unit root test for stationarity; a value of <0.05 usually implies that the Null hypothesis can be rejected, i.e. the sample is stationary |
| `Vector<T> stationary(DIFFERENCING method=integer,double degree=1,double fract_exponent=2)`| returns a stationary copy (e.g. for time series data) of the original Vector|
| `Vector<T> sort(bool ascending=true)`| returns a sorted copy of the original Vector |
| `Vector<T> shuffle()`| returns a randomly shuffled copy of the original Vector|
| `Vector<T> log_transform()`| returns a logarithmically transformed copy of the original Vector|
| `T polynomial_predict(T x,int power=5)`|performs polynomial regression to the specified power on the Vector data and predicts the fitting value for a hypothetical new index x|
| `double polynomial_MSE(int power=5)`| returns the Mean Squared Error (MSE) of polynomial regression to the specified power|
| `bool isGoodFit_linear(double threshold=0.95)`| returns whether linear regression is a good fit with respect to the specified confidence interval|
| `bool isGoodFit_polynomial(int power=5,double threshold=0.95)`| returns whether polynomial regression (to the specified power) is a good fit with respect to the specified confidence interval|
| `T linear_predict(T x)`| predicts the result of a hypothetical new index x based on linear regression of the Vector data |
| `double get_slope()`|returns the slope of linear regression of the vector data |
| `double get_y_intercept()`| returns the y-axis intercept of linear regression of the vector data |
| `double get_r_squared_linear()`| returns the coefficient of determination (r2) of linear regression of the Vector data|| `double get_r_squared_polynomial(int power=5)`| returns the coefficient of determination (r2) of polynomial regression (to the specified power) of the Vector data|

[[return to main page]](../../../README.md)