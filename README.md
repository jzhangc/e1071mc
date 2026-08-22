# e1071mc: Multicore Support Vector Machines

`e1071mc` extends the [e1071](https://CRAN.R-project.org/package=e1071)
package with a multicore interface for Support Vector Machines. It exposes
`svm_mc()`, a drop-in for `svm()` that additionally parallelises
k-fold cross-validation and multi-row prediction, and an OpenMP-parallel
prediction path in the C core.

The original `svm()` behaviour is left unchanged: `svm_mc()` reuses the
unmodified `svm.default` and `predict.svm` under the hood.

## Features

- **Parallel k-fold cross-validation**: each fold trains an independent
  libsvm model; the folds are distributed across cores with
  `parallel::mclapply` (`svm_mc(..., cross = k, n_cores = p)`).
- **Parallel prediction**: large test sets are split into row chunks, one per
  core, each predicted in parallel (`predict.svm_multicore`).
- **OpenMP prediction in C**: the per-row prediction loops in `src/Rsvm.c`
  carry `#pragma omp parallel for`, compiled in when the build is
  configured with `-fopenmp`.
- **Drop-in replacement**: identical argument list to `svm()`, plus one
  additional argument (`n_cores`). The returned object has class
  `svm_multicore` (a subclass of `svm`), so `print`, `summary`, `plot`,
  and `coef` all keep working.
- **Exact serial fallback**: calling with `n_cores = 1` (or leaving `cross = 0`)
  defers to `svm.default`/`predict.svm`, so results are byte-identical to
  `svm()`/`predict()`.

## Installation

```r
# from a source checkout
R CMD INSTALL .

# or, build and install the tarball
R CMD build .
R CMD INSTALL e1071mc_1.7-17.tar.gz
```

`e1071mc` requires an OpenMP-capable C/C++ compiler for the C-level prediction
speed-up. When the build has no `-fopenmp` (e.g. stock Apple clang), the
package still builds and runs; the C-level pragmas compile to a strict
sequential loop and the R-level `mclapply` parallelism covers CV and
prediction. See `R/svm_multicore.R` and `src/Rsvm.c`.

## Quick Start

```r
library(e1071mc)

# Classification, parallel 5-fold cross-validation on 2 cores
data(iris)
model <- svm_mc(Species ~ ., data = iris, cross = 5, n_cores = 2)
summary(model)
model$accuracies

# Matrix interface
x <- subset(iris, select = -Species)
y <- Species
model <- svm_mc(x, y, cross = 5, n_cores = 3)
pred  <- predict(model, x, n_cores = 3)
table(pred, y)

# Probabilities + decision values
model <- svm_mc(x, y, probability = TRUE, n_cores = 2)
pred  <- predict(model, x, probability = TRUE, decision.values = TRUE,
                 n_cores = 2)

# Regression
set.seed(1)
xr  <- 1:200
yr  <- 0.5 * xr + sin(xr / 5) + rnorm(200, sd = 0.3)
mr  <- svm_mc(xr, yr, cross = 5, n_cores = 3, type = "eps-regression")
mr$MSE; mr$tot.MSE; mr$scorrcoeff
```

## API Reference

### `svm_mc()`

Multicore training with parallel cross-validation. Identical to
`svm.default()` plus one argument, `n_cores`.

```r
svm_mc(x,
       y              = NULL,
       scale          = TRUE,
       type           = NULL,
       kernel         = "radial",
       degree         = 3,
       gamma          = if (is.vector(x)) 1 else 1 / ncol(x),
       coef0          = 0,
       cost           = 1,
       nu             = 0.5,
       class.weights  = NULL,
       cachesize      = 40,
       tolerance      = 0.001,
       epsilon        = 0.1,
       shrinking      = TRUE,
       cross          = 0,
       probability    = FALSE,
       fitted         = TRUE,
       n_cores        = NULL,    # default: detectCores() - 1
       ...,
       subset,
       na.action      = na.omit)
```

All arguments other than `n_cores` have the same meaning as in
`svm.default`. `n_cores` sets the number of parallel workers; if
`NULL`, it is set to `parallel::detectCores() - 1`. A value `<= 1`
disables parallelism, and the result is then identical to `svm()`.

If `cross > 0` and `n_cores > 1`, the k folds are trained in parallel via
`parallel::mclapply`; otherwise `svm.default` is used directly.

A formula variant, `svm_mc.formula()`, is provided:
`svm_mc(formula, data = NULL, ..., subset, na.action = na.omit,
scale = TRUE, n_cores = NULL)`.

**Returns** an object of class `svm_multicore` (a subclass of `svm`).
For `cross > 0`, it additionally carries the same per-fold fields as
`svm` (`accuracies`/`tot.accuracy` for classification, `MSE`/`tot.MSE`/
`scorrcoeff` for regression).

### `predict.svm_multicore()`

Parallel prediction. Row-chunks `newdata`, predicts each chunk with the
unmodified `predict.svm` in parallel, and concatenates the results in
original row order.

```r
predict(object,
        newdata,
        decision.values = FALSE,
        probability = FALSE,
        n_cores = object$n_cores,
        ...,
        na.action = na.omit)
```

`n_cores` defaults to `object$n_cores` (the number of cores used during
training). If `n_cores <= 1`, delegates to `predict.svm`.

### `cross`

Set `cross = k` to obtain k-fold cross-validation. The folds run in
parallel when `n_cores > 1`.

## Under the Hood

### Parallel Computation

- **Cross-validation folds**: each fold trains an independent libsvm model;
  the folds are dispatched across cores with `parallel::mclapply`
  (`R/svm_multicore.R:246`).
- **Prediction**: `newdata` is split into `n_cores` contiguous row chunks,
  each predicted in parallel; results are concatenated in original order
  (`R/svm_multicore.R:378`).
- **C-level prediction**: the per-row prediction loops in `src/Rsvm.c`
  (the probability, plain, and decision-value paths) carry
  `#pragma omp parallel for schedule(dynamic)` (`src/Rsvm.c:403,409,418`).
  When the build has `-fopenmp`, these loops run in parallel; otherwise
  they compile to a sequential loop.

### Thread Management

- `n_cores` defaults to `parallel::detectCores() - 1` if not supplied
  (`R/svm_multicore.R:415`).
- The code does **not** set `OMP_NUM_THREADS`; the number of OpenMP
  worker threads used inside `Rsvm.c` is controlled by the compiler/
  toolchain's normal mechanism (or by the user, e.g.
  `Sys.setenv(OMP_NUM_THREADS = N)`).

### C Implementation

The C-level parallelism is implemented in `src/Rsvm.c` (the existing
`Rsvm.c` that wraps the libsvm C++ core). No new source files are added.

## Performance

The speedup comes from parallelising the independent, repeated work,
not from parallelising a single SVM solve (the libsvm SMO solver is
inherently sequential).

- **k-fold CV** with `n_cores = p`: up to roughly `p×` faster, provided
  each fold's training time dominates the per-process spawn overhead.
  Empirically, a 10-fold CV on n = 8000, 6 features shows ~**3.4×
  speed-up with 6 cores** (measured during development; your mileage
  will vary with problem size and per-fold training time).
- **Prediction** on large `newdata`: scales roughly linearly with the
  number of cores while the per-row kernel cost dominates.
- **Single-train, no CV**: no speed-up on the training solve (the libsvm
  SMO solve is serial); the code calls the unmodified `svm.default`, whose
  fitted-value prediction still benefits from the OpenMP C loop when the
  build has `-fopenmp`.

## Build & Test

```sh
R CMD build .
R CMD check e1071mc_1.7-17.tar.gz
R CMD INSTALL .
```

The following checks are what to verify after a build:

- `n_cores = 1` (or `cross = 0`) produces results byte-identical to
  `svm()` (same support vectors, coefficients, and predictions).
- `predict.svm_multicore` with `n_cores > 1` equals `predict` with
   `n_cores <= 1` for the same model and data.
- k-fold CV with `n_cores > 1` returns per-fold accuracies / MSEs that
  are deterministic across runs (`parallel::mclapply` with
   `mc.set.seed = TRUE`).
- Regression (`type = "eps-regression"` and `"nu-regression"`) and
  all classification modes (C, nu, one-class) train and predict
  correctly.

## License

This package is released under the **GPL-2 | GPL-3** license (same as
the upstream `e1071`). See `DESCRIPTION`.

## References

- [e1071 package](https://CRAN.R-project.org/package=e1071) — upstream
  SVM implementation
- [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) — the C++ library
  `e1071` is built on
- R documentation for `parallel::mclapply` and `parallel::detectCores`
