//~/pulp_box/forejune_co_cuda/transformer6/util.h
#ifndef __UTIL_H__
#define __UTIL_H__
#include <vector>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <unordered_map>
using namespace std;
typedef vector<vector<double>>  matrixd;
// an activation function
double f(double z);
// Relu activation function with matrix input
matrixd relu(const matrixd &X);
// Derivative of Relu
matrixd relu_deriv (const matrixd &X);
// calculates entropy loss
double crossEntropy(const matrixd &P, const vector<int> &target);
// add two matrices
matrixd addMat(const matrixd &A, const matrixd &B);
// Transpose matrix n x m, B = A^T : m x n
matrixd transpose(const matrixd& A);
// Matrix multiplication   C = A X B   A: nxm, B: mxr, C: nxr
matrixd matmul(const matrixd& A, const matrixd& B);
// multiply a matrix by a scalar
matrixd mulMats (const matrixd &A, const double s);
// scale a matrix
void scaleMat(matrixd &A, const double s);
// apply masking to S
matrixd causalMask(matrixd S);
// update weight matrix W -= dW * eta
void updateWeight(matrixd &W, const matrixd &dW, const double eta);
// initialize an input matrix with certain random values
void initMatrix(matrixd& M, const int n, const int m);
void initMatrix(matrixd& M, const int n, const int m, double dModel);
// print a token dictionary
void printDictionary(unordered_map<string, int> &d);
void printVec (const vector<int> &v);
// print a matrix
void printMatrix(const matrixd &y);
// Add & Norm
matrixd addNnorm(const matrixd &A, const matrixd &B);
void clip(matrixd &A, double max_val);
// softmax, 1D vector input
vector<double> softmax(const vector<double> &v);
// softmax activation function, 2D input vector
matrixd softmax(const matrixd& input);
// derivative function of softmax
matrixd softmax_derivative(const matrixd &dL_dP, const matrixd &P);
#endif
