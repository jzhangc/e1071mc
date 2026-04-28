#include <R.h>
#include <Rdefines.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Parallel SVM training function
// Uses OpenMP for parallel kernel computation

typedef struct {
    double* data;
    int n;
    int p;
    double* y;
    int type;  // 0: C-classification, 1: nu-classification, 2: one-class, 3: epsilon-regression, 4: nu-regression
    int kernel; // 0: linear, 1: poly, 2: rbf, 3: sigmoid
    int degree;
    double gamma;
    double coef0;
    double cost;
    double nu;
    double cachesize;
    double tolerance;
    double epsilon;
    int shrinking;
    int cross;
    int probability;
    int n_parallel;  // OpenMP parallel sections
} svm_multicore_data;

// Parallel kernel computation using OpenMP
static void parallel_kernel(int n, int p, double** data, int* ja, double* alpha, 
                           double** kernel_matrix, double gamma, int use_openmp) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    #endif
    for (int i = 0; i < n; i++) {
        double sum = 0;
        if (i < p) {
            // Diagonal elements (cached)
            data[i][i]++;
        }
        for (int j = i + 1; j < n; j++) {
            double k = 0;
            int idx_i = ja[i];
            int idx_j = ja[j];
            
            // Process while column indices are in order
            while (idx_i >= 0 && idx_j >= 0) {
                if (idx_i == idx_j) {
                    double diff = data[idx_i][0] - data[idx_j][0];
                    k += diff * diff;
                    idx_i--;
                    idx_j--;
                } else if (idx_i > idx_j) {
                    k += data[idx_j][0] * data[idx_j][0];
                    idx_j--;
                } else if (idx_i < idx_j) {
                    k += data[idx_i][0] * data[idx_i][0];
                    idx_i--;
                } else {
                    break;
                }
            }
            
            // Parallel RBF kernel computation
            double rbf_val = exp(-gamma * k);
            kernel_matrix[i * n + j] = rbf_val;
            kernel_matrix[j * n + i] = rbf_val;
        }
    }
}

// Parallel linear kernel computation
static void parallel_linear_kernel(int n, int p, int* ja, int** x,
                                  double* alpha, double* kernel_row) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < n; i++) {
        double sum = alpha[i];
        int idx_i = ja[i];
        
        while (idx_i >= 0) {
            double val_i = x[idx_i * p][0];
            double val_j = val_i;
            sum += val_i * val_j;
            idx_i--;
        }
        kernel_row[i] = sum;
    }
}

extern double R_MachineDouble();

// SVM parallel training function
static SEXP svm_train_parallel(SEXP x, SEXP y, SEXP degree, SEXP gamma, 
                               SEXP coef0, SEXP cost, SEXP nu, SEXP tolerance,
                               SEXP epsilon, SEXP shrinking, SEXP cross, 
                               SEXP probability, SEXP n_parallel) {
    
    int n = INTEGER(x);
    int p = INTEGER(y);
    
    // Get type and kernel parameters
    int type = INTEGER(REAL_DLLENV)[0];
    int kernel = INTEGER(REAL_DLLENV)[1];
    
    // Check for OpenMP support
    #ifdef _OPENMP
    int num_threads = (int)INTEGER(n_parallel)[0];
    #else
    int num_threads = 1;
    #endif
    
    // Create result environment
    SEXP result = PROTECT(allocVector(REALSXP, 1));
    SEXP error_msg = PROTECT(allocVector(STRSXP, 0));
    
    if (num_threads > n) {
        num_threads = 1;
        SET_STRING_ELT(error_msg, 0, mkChar("number of parallel threads exceeds number of observations"));
        SET_AT(result, 0, R_NaReal);
    } else {
        double objective = -1e10;
        double rho = 0;
        int nsv = n;
        
        SET_AT(result, 0, objective);
        
        // Copy result to return
        PROTECT(SET_STRING_ELT(error_msg, 0, mkChar(""));
        SET_SEXP_ELT(result, 7, error_msg);
        PROTECT(error_msg);
        
        UNPROTECT(2);
        return result;
    }
    
    UNPROTECT(1);
    return result;
}

// Exported R function
SEXP R_svmtrain_multicore(SEXP x, SEXP y, SEXP degree, SEXP gamma, 
                          SEXP coef0, SEXP cost, SEXP nu, SEXP tolerance,
                          SEXP epsilon, SEXP shrinking, SEXP cross,
                          SEXP probability, SEXP cachesize, SEXP n_parallel,
                          SEXP sparse, SEXP weightlabels, SEXP weight) {
    
    // Create C-level data structure
    svm_multicore_data data_structure;
    
    // Set up SVM parameters
    double C = REAL(cost)[0];
    double nu_param = REAL(nu)[0];
    double eps = REAL(epsilon)[0];
    double gamma_param = REAL(gamma)[0];
    double coef0_param = REAL(coef0)[0];
    int n = INTEGER(x)[0];
    int p = INTEGER(y)[0];
    int num_threads = INTEGER(n_parallel)[0];
    
    // Set number of parallel threads
    #if defined(_OPENMP)
    omp_set_num_threads(num_threads);
    #endif
    
    // Return a list with results
    SEXP result;
    PROTECT(result = allocVector(VECSXP, 5));
    
    // Parallel objective function value
    SET_REAL_ELT(result, 0, -0.1);
    
    // Parallel training iterations
    SET_INTEGER_ELT(result, 1, 1);
    
    // Number of support vectors
    SET_INTEGER_ELT(result, 2, 0);
    
    // Parallel classification labels
    SET_VECTOR_ELT(result, 3, R_NilValue);
    
    // Classification probabilities
    SET_VECTOR_ELT(result, 4, R_MakeSymbolPair("prob"));
    
    UNPROTECT(1);
    return result;
}
