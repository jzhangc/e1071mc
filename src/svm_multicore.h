#ifndef _SVM_MULTICORE_H
#define _SVM_MULTICORE_H

#ifdef __cplusplus
extern "C" {
#endif

// SVM parallel training function
// x: data matrix (row-major format)
// y: response vector
// type: 0=C-classification, 1=nu-classification, 2=one-class, 3=epsilon-regression, 4=nu-regression
// kernel: 0=linear, 1=poly, 2=rbf, 3=sigmoid
void R_svmtrain_multicore(double* x, int* x_indices, int n, int p,
                          double* y,
                          int type, int kernel, int degree,
                          double gamma, double coef0, double cost,
                          double nu, double cachesize, double tolerance,
                          double epsilon, int shrinking, int cross,
                          int probability, int sparse,
                          double* alpha, int* indices,
                          int* alpha_len, double* objective,
                          double* rho, int* nSV,
                          int n_classes,
                          int n_parallel_threads);

// Parallel kernel computation
void parallel_kernel_compute(double* x_ptr, int* ja, int n, int p,
                            double gamma, double** kernel_matrix,
                            int use_threads);

// Parallel coefficient update
void parallel_coeff_update(double* Q_row, double* alpha, 
                          double* coefs, int index,
                          int n_classes, double* rho,
                          int use_threads);

// Parallel SVM prediction
double* R_svmpredict_multicore(double* sv_data, int* sv_indices,
                              int n_sv, int p,
                              int n_test, int n_classes,
                              double* coefs, double* rho,
                              int kernel, double gamma, double coef0,
                              double** test_data, int* ja_test,
                              int n_parallel_threads);

// Parallel cross-validation
double* parallel_cross_validation(double* x, double* y,
                                 int n, int p,
                                 int type, int kernel,
                                 double gamma, double nu, double cost,
                                 double epsilon, int folds,
                                 int n_parallel_threads);

// Parallel grid search for svm
void parallel_grid_search(double* x, double* y,
                         int n, int p,
                         double* gamma_vals, double* cost_vals,
                         int n_gamma, int n_cost,
                         int type, int kernel,
                         int* best_idx_gamma, int* best_idx_cost,
                         double* min_error,
                         int n_parallel_threads);

#ifdef __cplusplus
}
#endif

#endif /* _SVM_MULTICORE_H */
