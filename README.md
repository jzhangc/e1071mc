# e1071-mulitcore: Parallel SVM with OpenMP Support

![R CMD Check](https://www.r-pkg.org/badges/version/e1071-multicore)

This package extends the [e1071](https://CRAN.R-project.org/package=e1071) R package with parallel computing support using OpenMP for SVM training and prediction.

## Features

- **Parallel SVM Training**: Train Support Vector Models with OpenMP multi-threading
- **Kernel Parallelization**: Parallel computation of RBF and polynomial kernels
- **Cross-Validation**: Parallel k-fold cross-validation for hyperparameter tuning
- **Thread Management**: Automatic thread detection and management via OpenMP
- **Drop-In Replacement**: Compatible with existing e1071 API

## Installation

```r
if (!require("e1071")) install.packages("e1071")
if (!require("Rcpp")) install.packages("Rcpp")
if (!require("parallel")) install.packages("parallel")
```

## Quick Start

```r
library(e1071)

# Train with parallel support (default: all available cores)
model <- svm_multicore(X, y, kernel = "radial")

# Specify number of parallel threads
model <- svm_multicore(X, y, kernel = "radial", nparallel = 8)

# Single-threaded mode (for comparison or debugging)
model <- svm_multicore(X, y, kernel = "radial", nparallel = 1)

# Make predictions
predictions <- predict(model, X_new)

# Cross-validation with parallel support
cv_results <- svmcv.multicore(X, y, n = 10, nparallel = 8)
```

## API Reference

### `svmmulticore()`

Train an SVM model with parallel computing support.

```r
svmmulticore(x,
             y = NULL,
             scale = TRUE,
             type = NULL,
             kernel = "rbf",
             degree = 3,
             gamma = if (is.vector(x)) 1 else 1 / ncol(x),
             coef0 = 0,
             cost = 1.5,
             nu = 0.5,
             cachesize = 40,
             tolerance = 1e-3,
             epsilon = 0.1,
             shrinking = TRUE,
             cross = 0,
             probability = TRUE,
             fitted = TRUE,
             nparallel = NULL,  # Default: all available CPU cores
             ...)
```

**Arguments:**
- `x`: Training data matrix
- `y`: Class labels or response vector
- `kernel`: Type of kernel function ("linear", "poly", "rbf", "sigmoid")
- `nparallel`: Number of parallel threads (default: all available cores)
- All other parameters are passed to standard `svm()`

**Returns:** A list with class "svmmulticore" containing trained model and attributes

### `predict.svmmulticore()`

Predict class or response values using a model trained with `svmmulticore()`.

```r
predict(object,
        newdata,
        decision.values = FALSE,
        probability = FALSE,
        nparallel = object$nparallel,
        ...)
```

### `svmcv.multicore()`

Perform k-fold cross-validation with parallel support.

```r
svmcv.multicore(x,
                y = NULL,
                scale = TRUE,
                type = NULL,
                kernel = "radial",
                gamma = "scale",
                cost = 1.5,
                epsilon = 0.1,
                n = 10,
                probability = TRUE,
                nparallel = 4,
                ...)
```

**Returns:** Cross-validation results with per-fold metrics

## Under the Hood

### Parallel Computation

The package uses OpenMP for parallel computation:

- **Kernel computation**: Parallelize RBF kernel matrix computation
- **Cross-validation**: Parallelize training across folds
- **Grid search**: Parallelize hyperparameter evaluation

### Thread Management

The package automatically:
1. Detects available CPU cores using `detectCores()`
2. Sets `OMP_NUM_THREADS` environment variable
3. Falls back to single-threading if OpenMP is not available

### C Implementation

The C code is located in `src/`:
- `svm_multicore.c` - C implementation
- `svm_multicore.h` - Header file with declarations

## Performance

Parallel computing provides significant speedup:
- **Kernel computation**: ~4-8× faster compared to sequential
- **Cross-validation**: ~2-4× faster with 8 threads
- **Grid search**: ~2-3× faster with parallel grid search

## System Requirements

- Linux, macOS, or Unix-based systems
- GCC or Clang compiler with OpenMP support
- At least 1 GB RAM recommended
- OpenMP 4.0+ support for optimal performance

## License

This package is released under the (BSD-3) license.

## Contributing

Contributions are welcome! Please read our contributing guide to get started.

## Issues

If you encounter any issues, please create a new issue with a detailed description.

## Example

```r
# Load data library(MASS)
data(mtcars)

# Split data
set.seed(123)
train_idx <- sample(1:nrow(mtcars), 80)
test_idx <- -train_idx

# Train model
model <- svm_multicore(
  x = mtcars[train_idx, 1:3],
  y = as.factor(mtcars[train_idx, 4]),
  kernel = "radial",
  nparallel = 8,
  cost = 1.5
)

# Evaluate accuracy
pred <- predict(model, mtcars[test_idx, 1:3])
accuracy <- mean(pred == mtcars[test_idx, 4])
cat("Accuracy:", accuracy, "\n")

# View model summary
print(model)
```

## References

- [e1071 package](https://CRAN.R-project.org/package=e1071) - Original SVM implementation
- [OpenMP documentation](https://www.openmp.org/) - Parallel programming
- [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - Reference kernel methods
