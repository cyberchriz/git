[[return to main page]](../../../README.md)
## Custom Vectors, 2d Matrices and n-dimensional Arrays
usage: `#include <array.h>` or include as part of `<datascience.h>`

description:
- this class implements fast static stack-allocated numeric vectors, matrices and arrays
- 'under the hood' these are simple one-dimensional arrays
- a multidimensional index will be internally converted into a 1d index
- the main benefit of the class comes with the wide variety of mathematic functions that can be applied to these arrays

## `class Array`
An instance of the class can be created by passing an array of the array dimensions as a an argument to the constructor.
For example, for a 3x3x4 array of type <double> we can write:
```
Array<double> myArr({3,3,4});
```
Methods:
|method|description|
|------|-----------|
| `Array<T>::set(int* index, T value)` | assigns the value at the given index |
| `Array<T>::get(int* index)` | returns the value at the given index |
| `Array<T>::get_dimensions()` | returns an integer array that holds the array dimensions |
| `Array<T>::get_size(int dimension)` | returns the size of the specified dimension as an integer |
| `Array<T>::get_elements()` | returns the total number of elements of the entire array (all dimensions) |
| `Array<T>::fill_values(T value)` | fills the entire array with the specified value (of datatype T) |
| `Array<T>::fill_zeros()` | fills the entire array with zeros |
| `Array<T>::fill_identity()` | applies the identity matrix to the array |
| `Array<T>::fill_random_gaussian(T mu=0, T sigma=1)` | fills the array with random values from a gaussian normal distribution with the given mean and sigma |
| `Array<T>::fill_random_uniform(T min=0,T max=1.0)` | fills the array with values from a random uniform distribution within the given range |
| `Array<T>::mean()` | returns the arrithmetic mean of all values of the array (all dimensions) |
| `Array<T>::median()` | returns the Median of all values of the array (all dimensions) |
| `Array<T>::variance()` | returns the variance of all values of the array (all dimensions) |
| `Array<T>::stddev()` | returns the standard deviation of all values of the array (all dimensions) |
| `Array<T>::sum()` | returns the sum of all values of the array (all dimensions) |
| `Array<T>::add(T value)` | elementwise addition of the specified value to all values of the array |
| `Array<T>::add(const Array& other)` | elementwise (scalar) addition of a second array of equal dimensions to the current array |
| `Array<T>::product()` | returns the product of all values of the array |
| `Array<T>::multiply(T factor)` | elementwise multiplication of all values of the array with the given factor |
| `Array<T>::multiply(const Array& other)` | elementwise (scalar) multiplication with a second array of equal dimensions |
| `Array<T>::substract(T value)` | elementwise substraction of the specified value from all values of the array |
| `Array<T>::substract(const Array& other)` | elementwise substraction of a second array of equal dimensions from the current array |
| `Array<T>::divide(T quotient)` | divides all values of the array by the given quotient |
| `Array<T>::divide(const Array& other)` | elementwise (scalar) division of the current array by a second array of equal dimensions |
| `Array<T>::pow(T exponent)` | exponentiates all values of the array by the specified power |
| `Array<T>::sqrt()` | applies the square roots to all values of the array |
| `Array<T>::replace(T old_value, T new_value)` | replaces all findings of the specified old value by the specified new value |
| `Array<T>::find(T value)` | returns the number of findings of the specified value across the entire array (all dimensions) |
| `Array<T>::function(T (*pointer_to_function)(T))` | applies the specified function to all values of the array (elementwise) |
| `Array<T>::operator=(const Array& other)` | copies the values of the specified second array (of equal dimensions) into the current array |

## `class Matrix`
This is a derived class of `class Array<T>`. It inherits all its methods, but deals with the specific case of 2d arrays.
The constructor is slightly different: because the dimensions are already known to be 2d, they don't need to be passed in as an array of integers, but simply the size of the x and y dimensions instead.
The `.get()` method equally only needs two integers for the x and y indices it refers to.
The `.set()` method needs two integers for the x and y indices, followed by the assigned value as a third argument.

For example a 2d matrix of size 4x4 and data type `<float>` can be instantiated like this:
```
Matrix<float> myMatrix(4,4);
```
On top of all the methods given above, inherited from `class Array<T>`, the Matrix class additionally implements:

| method | description |
| `Matrix<T>::dotproduct(const Matrix& other)` | returns the resulting new `Matrix<T>` given by the dotproduct of the current matrix and a second matrix |
| `Matrix<T>::transpose()` | returns the transpose of the current matrix as the resulting new `Matrix<T>` |

## `class Vector`
This is also a derived class of `class Array<T>` and it also inherits all its methods, but deals with the specific case of 1d arrays.
The constructor is slightly different: because the dimensions are already known to be 1d, only the number of elements is required.
The `.get()` method equally only needs a single integer for the index it refers to.
The `.set()` methods needs an integer for the index it refers to, followed by the assigned value as a second argument.

For example a 1d `Vector` with 100 elements of type `<double>` can be instantiated like this:
```
Vector<double> myVec(100);
```

[[return to main page]](../../../README.md)