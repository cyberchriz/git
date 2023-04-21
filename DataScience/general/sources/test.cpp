#include "../headers/array.h"
#include <iostream>

int main(){
    Matrix<int> Matrix1{3,4};
    Matrix1.fill_zeros();
    Matrix1.print("\nHere's a 3x4 Matrix<int> as an example.");
    
    Matrix1.fill_random_uniform(0,99);
    Matrix1.print("\nFilling the Matrix with random numbers from a uniform distribution 0-99):");

    Matrix1.print("\nLet's check the indices, too:", ", ", "\n", true);

    auto Matrix2 = Matrix1.transpose();
    Matrix2->print("\nExample for Matrix transpose:", ", ", "\n", true);

    (*Matrix2)++;
    Matrix2->print("\nTesting the postfix increment operator++:");

    auto Matrix3 = Matrix2->asMatrix(5, 4);
    Matrix3->print("\nMaking the Matrix a little bigger, padding with 0:");

    Vector<int> Vector1(10);
    Vector1.fill_range(100,-1);
    Vector1.print("\nNow let's try with a vector, filled with a range of numbers, starting from 100, step -1:");

    auto Vector2 = Vector1.shuffle();
    Vector2->print("\nRandom shuffle:");
    
    std::cout << "\nMean=" << Vector2->mean() << std::endl;
    std::cout << "Median=" << Vector2->median() << std::endl;
    std::cout << "Standard Deviation=" << Vector2->stddev() << std::endl;
    std::cout << "Ranking: "; (Vector2->ranking())->print();
    std::cout << "Stationary (method=integer, degree=1): "; (Vector2->stationary(integer))->print();
    
    Vector<int> x_axis(Vector2->get_elements());
    x_axis.fill_range(0,1);
    auto regr_result = x_axis.linear_regression(Vector1);
    std::cout << "Linear Regression Slope=" << regr_result->slope << std::endl;
    std::cout << "Lin. Reg. r_squared=" << regr_result->r_squared << std::endl;

    Vector2->print();
    auto Matrix4 = Vector2->transpose();
    Matrix4->print("\nTransposed version of same vector (this will actually create a 'single column matrix'):");

    Array<int> myArr{2,2,2,2};
    myArr.fill_random_uniform(0,9);
    myArr.print("\nTesting the print() function for a 4d array (3x3x2x2), without indices: ", " ", "\n",false);
    myArr.print("\n... and the same WITH indices: ", " ", "\n",true);
}