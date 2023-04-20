#include "../headers/array.h"
#include <iostream>

int main(){
    Matrix<int> myMatrix{3,4};
    myMatrix=0;
    myMatrix.print("\nHere's a 3x4 Matrix<int> as an example.");
    
    myMatrix.fill_random_uniform(0,99);
    myMatrix.print("\nFilling the Matrix with random numbers from a uniform distribution 0-99):");

    myMatrix.print("\nLet's check the indices, too:", ", ", "\n", true);

    auto newMatrix = myMatrix.transpose();
    newMatrix->print("\nExample for Matrix transpose:", ", ", "\n", true);

    ++*(newMatrix);
    newMatrix->print("\nTesting the prefix increment operator++:");

    (*newMatrix)++;
    newMatrix->print("\nTesting the postfix increment operator++:");

    auto largeMatrix = newMatrix->asMatrix(5, 4);
    largeMatrix->print("\nMaking the Matrix a little bigger, padding with 0:");

    Vector<int> myVec(7);
    myVec.print("\nNow let's try with a vector");

    myVec.fill_range(100,-1);
    myVec.print("\nFilling the vector with a range of numbers, starting from 100, step -1:");

    auto Vec2=myVec.transpose();
    Vec2->print("\nTransposed version of same vector (this will actually create a 'single column matrix'):");

    Array<int> myArr{4,4,2,2};
    myArr.fill_random_uniform(0,9);
    myArr.print("\nTesting the print() function for a 4d array (4x4x2x2), without indices: ", " ", "\n",false);
    myArr.print("\n... and the same WITH indices: ", " ", "\n",true);
}