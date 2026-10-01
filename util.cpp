//~/pulp_box/forejune_co_cuda/transformer6/util.cpp  //https://forejune.co/cuda/
#include "util.h"
#include <random>
using namespace std;
// Activation function: using ReLU (Rectified Linear Unit)
double f(double z) { return max(0.0, z); }
// Relu activation function with matrix input
matrixd relu(const matrixd &X) {
    int rows = X.size();
    int cols = X[0].size();
    matrixd Y = X;    // same dimension of X
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)  Y[i][j] = max(0.0, X[i][j]);
    return Y;
}
// Derivative of Relu
matrixd relu_deriv(const matrixd &X) {
    int rows = X.size();
    int cols = X[0].size();
    matrixd Y = X;    // same dimension as X
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            if ( X[i][j] > 0 )  Y[i][j] = 1;  else  Y[i][j] = 0;
    return Y;
}
// calculates entropy loss 
double crossEntropy(const matrixd &P, const vector<int> &target) {
    double L = 0;
    int nt = P.size();
    for (int i = 0; i < nt; i++){
        int k = target[i];
        L -= log( P[i][k] + 1e-12 );
    }
    return L / nt;
}
// C = A + B
matrixd addMat(const matrixd &A, const matrixd &B) {
     int n = A.size();          // rows
     int m = A[0].size();       // cols
     matrixd C = A;       // same dimension as A
     for (int i = 0; i < n; i++)
         for(int j = 0; j < m; j++)  C[i][j] = A[i][j] + B[i][j];
     return C;
}
// Transpose matrix n x m, B = A^T : m x n
matrixd transpose(const matrixd& A) {
    int n = A.size();    //number of rows
    int m = A[0].size(); //number of columns
    matrixd B(m, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) B[j][i] = A[i][j];
    return B;
}
// Matrix multiplication   C = A X B   A: nxm, B: mxr, C: nxr
matrixd matmul(const matrixd& A, const matrixd& B) {
    int n  = A.size();
    int m = A[0].size();
    int r = B[0].size();
    matrixd C(n, vector<double>(r, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < r; j++)
            for (int k = 0; k < m; k++) C[i][j] += A[i][k] * B[k][j];
     return C;
}
// scale a matrix
void scaleMat(matrixd &A, const double s) {
    int m = A.size();    // number of rows
    int n = A[0].size();    // number of columns
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)  A[i][j] *= s;
}
// maultiply a matrix by a scalar s
matrixd mulMats (const matrixd &A, const double s) {
    int m = A.size();    // number of rows
    int n = A[0].size();    // number of columns
    matrixd B = A;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)  B[i][j] *= s;
    return B;
}
void initMatrix(matrixd& M, const int n, const int m) {
    // resizing 2D vector to n x m with initial values 0
    M.assign(n, vector<double>(m, 0));
    random_device rd;
    mt19937 gen(rd());      //psuedo random number generator 
    double sigma = 1.0 / n;
    normal_distribution<double> dist(0.0, sigma);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)  M[i][j] = dist(gen);
    for (int i = 0; i < n; ++i) 
        for (int j = 0; j < m; ++j)  M[i][j] = rand() % 10000 / 100000.0;
}
void initMatrix(matrixd& M, const int n, const int m, double sigma) {
      // resizing 2D vector to n x m with initial values 0
      M.assign(n, vector<double>(m, 0));
      random_device rd;
      mt19937 gen(rd());      //psuedo random number generator
      double d = n;
      normal_distribution<double> dist(0.0, sigma);  
      for (int i = 0; i < n; ++i) 
          for (int j = 0; j < m; ++j)  M[i][j] = dist(gen);
}
void printDictionary(unordered_map<string, int> &d) {
  unordered_map<string, int>::iterator it = d.begin();
  int k = 0;
  while ( it != d.end() ){
      cout << right << setw(12) << it->first << ": " << setw(3) << it->second;
      it++;  k++;
      if ( k % 5 == 0 )  cout << endl;
  }
}
void printVec (const vector<int> &v) {
    int n = v.size();
    for (int i = 0; i < n; i++){
        cout << v[i];
        if ( i < n - 1)  cout  << ", ";
    }
    cout << endl; 
}
void printMatrix(const matrixd &y) {
    int cols = y.size();
    int rows = y[0].size();
    for (int i = 0; i < cols; ++i) {
        cout << "Token " << i << ": [";
        for (int j = 0; j < rows; ++j) {
            cout << fixed << setw(7) << setprecision(3) << y[i][j];
            if (j < 4) cout << ", ";
        }
        cout << " ]\n";
    }
}
matrixd addNnorm(const matrixd &A, const matrixd &B) {
     auto x = addMat(A, B);     // add two matrices
     int n = x.size();          // rows
     int m = x[0].size();       // cols
     matrixd y = x;             // output same size as x 
     // normalize the result
     double epsilon = 1e-5;
     for (int i = 0; i < n; i++) {
         double sum = 0;          // sum of one row
         double std = 0;          //sigma_i square
         for (int j = 0; j < m; j++)  sum += x[i][j];
         double mean = sum / m;  // mu_i, mean of i-th row
         sum = 0;
         for (int j = 0; j < m; j++) {
            double diff = x[i][j] - mean;
            sum += diff * diff;
         }
         std = sum / m;
         for (int j = 0; j < m; j++) {
            double xd = (x[i][j] - mean) / sqrt(std + epsilon);
            y[i][j] = xd;         // gamma = 1, beta = 0;
         }
     }
     return y;  // final output of Add & norm
}
matrixd scaledAttention(const matrixd &Q, const matrixd &K, const matrixd &V, bool mask) {
    double d_k, scale;
    matrixd KT, A;    // K transpose, attention
    d_k = Q[0].size();
    scale = 1.0 / sqrt( d_k );
    KT = transpose ( K );
    A = matmul(Q, KT);
    scaleMat(A, scale); 
    if ( mask ) {
       for (int i = 0; i < A.size(); i++)
           for (int j = i+1; j < A[0].size(); j++)  A[i][j] = -1e9;    // causal mask
    }
    A = softmax( A );
    matrixd output = matmul(A, V);
    return output;
}
matrixd causalMask(matrixd S) {
    int n = S.size();
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            S[i][j] = -1e9;  // effectively -âˆž
        }
    }
    return S;
}
void updateWeight(matrixd &W, const matrixd &dW, const double eta) {
    for (int i = 0; i < W.size(); i++)
        for (int j = 0; j < W[0].size(); j++)  W[i][j] -= eta * dW[i][j];
}
void clip(matrixd &A, double max_val = 5.0) {
    for (auto &row : A)
        for (auto &x : row)
            if (x > max_val) x = max_val; else if (x < -max_val) x = -max_val;
}
// softmax activation function, 1D input vector
vector<double> softmax(const vector<double> &z) {
    int m = z.size();
    vector<double> y(m, 0);
    //maximum value of vector
    double maxVal = *max_element(z.begin(), z.end());
    double sum = 0;
    // Subtract max for numerical stability
    for (int j = 0; j < m; j++) {
        y[j] = exp(z[j] - maxVal);
        sum += y[j];
    }
    // Normalize
    for (int j = 0; j < m; j++)  y[j] /= sum;
    return y;
}
// softmax activation function, 2D input vector
matrixd softmax(const matrixd& input) {
    int n = input.size(), m = input[0].size();    // n x m matrix 
    // 2D output vector n x m y[n][m]
    vector<vector<double>> y(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        //maximum value of the row
        double maxVal = *max_element(input[i].begin(), input[i].end());
        double sum = 0;
        // Subtract max for numerical stability
        for (int j = 0; j < m; j++) {
            y[i][j] = exp(input[i][j] - maxVal);
            sum += y[i][j];
        }
        // Normalize
        for (int j = 0; j < input[i].size(); j++)  y[i][j] /= sum;
    }
  return y;
}
// derivative function of softmax
matrixd softmax_derivative(const matrixd &dL_dP, const matrixd &P) {
    int nt = P.size(), ns = P[0].size();
    matrixd dL_dS(nt, vector<double>(ns,0));
    for (int i = 0; i < nt; i++) {
          double dot = 0.0;
          for (int j = 0; j < ns; j++)
              dot += P[i][j] * dL_dP[i][j];  // dot product of two row-vectors
          for (int j = 0; j < ns; j++)  // dot used by all columns of dL/dA
               dL_dS[i][j] = P[i][j] * (dL_dP[i][j] - dot);
    }
    return dL_dS;
}
