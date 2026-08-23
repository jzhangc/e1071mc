# e1071mc: Multicore Support Vector Machines

`e1071mc` extends the [e1071](https://CRAN.R-project.org/package=e1071)
package with a multicore interface for Support Vector Machines. It exposes
`svm_mc()`, a drop-in for `svm()` that additionally parallelises k-fold
cross-validation (one process per fold, via `parallel::mclapply`), and a
C-level OpenMP-parallel prediction path (the per-row prediction loops in
`src/Rsvm.c`).  The inner per-iteration loops of the libsvm SMO solver in
`src/svm.cpp` are also guarded with OpenMP pragmas, so a single training solve
can run across threads.

The original `svm()` behaviour is left unchanged: `svm_mc()` reuses the
unmodified `svm.default` and `predict.svm` under the hood, and only
*additively* raises the C-level OpenMP thread count around its own calls.

## Features

- **Parallel k-fold cross-validation**: each fold trains an independent
  libsvm model; the folds are distributed across cores with
 `parallel::mclapply`, one process per fold (`svm_mc(..., cross = k, n_cores = p)`).
- **Parallel prediction at the C level**: `predict.svm_multicore` raises a
  C-level OpenMP thread count (the `e1071mc_threads` global in `src/Rsvm.c`)
  and calls the unmodified `predict.svm` exactly once.  The per-row prediction
  loops inside `svmpredict` are guarded with `#pragma omp parallel for`, so the
  rows of the test matrix are classified across threads.
- **OpenMP training in C**: inside the libsvm SMO solver (`src/svm.cpp`) the
  per-iteration gradient and `G_bar` updates, and the RBF `x_square`
  pre-compute, also carry `#pragma omp parallel for`; these speed up a *single*
  training solve when compiled with `-fopenmp`.  The SMO outer working-set loop
  itself is inherently sequential and is not split.
- **Drop-in replacement**: identical argument list to `svm()`, plus one
  additional argument (`n_cores`). The returned object has class
  `svm_multicore` (a subclass of `svm`), so `print`, `summary`, `plot`,
  and `coef` all keep working.
- **Exact serial fallback**: calling with `n_cores = 1` (or leaving `cross = 0`)
  defers to `svm.default`/`predict.svm`, so results are byte-identical to
  `svm()`/`predict()`.

## Installation

`e1071mc` (the multicore SVM variant) is developed on the `nightly` branch of
the `jzhangc/e1071mc` repository; the default `master` branch still ships the
upstream `e1071` (v1.7-17). Install from `nightly` unless another branch is
intended.

### From GitHub with `devtools`

```r
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")

# default branch (master, i.e. upstream e1071)
devtools::install_github("jzhangc/e1071mc")

# a specific branch or commit hash via the `ref` argument
devtools::install_github("jzhangc/e1071mc", ref = "nightly")    # the e1071mc source
devtools::install_github("jzhangc/e1071mc", ref = "parallel")
devtools::install_github("jzhangc/e1071mc", ref = "5f1204b")    # a commit hash
```

`install_github` forwards `ref` to the underlying `remotes::install_github`;
`ref` accepts a branch name, a commit hash, or (if present) a git tag, and
defaults to `"HEAD"` (the repository's default branch). The repository currently
has no tags; use a branch name or commit hash.

### From GitHub with `pak`

`pak::pkg_install` has no `ref` argument; instead the branch is encoded in the
URL as `https://github.com/OWNER/REPO/tree/<branch>` (or `/tree/<tag>` /
`/tree/<commit-hash>`).

```r
if (!requireNamespace("pak", quietly = TRUE))
    install.packages("pak")

# default branch (master, i.e. upstream e1071)
pak::pkg_install("jzhangc/e1071mc")

# a specific branch or commit hash via the /tree/<ref> URL form
pak::pkg_install("https://github.com/jzhangc/e1071mc/tree/nightly")    # e1071mc source
pak::pkg_install("https://github.com/jzhangc/e1071mc/tree/parallel")
pak::pkg_install("https://github.com/jzhangc/e1071mc/tree/5f1204b")    # a commit hash
```

The bare `owner/repo` form is equivalent to `master` (the default branch).

### From source

```r
# from a source checkout
R CMD INSTALL .

# or, build and install the tarball
R CMD build .
R CMD INSTALL e1071mc_1.7-17-1-20260822.tar.gz
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

Parallel prediction via C-level OpenMP. It raises the `e1071mc_threads`
global and calls the unmodified `predict.svm` exactly once; the per-row
prediction loops inside `svmpredict` (`src/Rsvm.c`) then run across threads.
The result is identical to a serial `predict.svm`.

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
  the folds are dispatched across cores with `parallel::mclapply`, one
  process per fold
   (`R/svm_multicore.R:262`).
- **Prediction (C-level OpenMP)**: `predict.svm_multicore` raises the
   `e1071mc_threads` global and calls the unmodified `predict.svm` exactly
   once (`R/svm_multicore.R:376`); it does *not* split `newdata` into row
  chunks in R.  The per-row prediction loops inside `src/Rsvm.c` (the
  probabilistic, plain, and decision-value paths) then run across threads.
- **Training (C-level OpenMP)**: inside the libsvm SMO solver in
  `src/svm.cpp` the per-iteration gradient / `G_bar` updates and the RBF
  `x_square` pre-compute carry `#pragma omp parallel for`
   (`src/svm.cpp:296,733,756,767`); a single training solve can run across
  threads, clamped to one thread for small problems (`l < 2000`).

These OpenMP pragmas are standard C and compile to a strict sequential loop on
any toolchain; they become parallel only when the package is compiled with
`-fopenmp`.

### Thread Management

- `n_cores` defaults to `parallel::detectCores() - 1` if not supplied
  (`R/svm_multicore.R:415`).
- The code does **not** set `OMP_NUM_THREADS`; the number of OpenMP
  worker threads used inside `Rsvm.c` is controlled by the compiler/
  toolchain's normal mechanism (or by the user, e.g.
  `Sys.setenv(OMP_NUM_THREADS = N)`).

### C Implementation

The C-level parallelism is spread across two existing source files, with no
new files added:

- `src/Rsvm.c` — the `svmpredict` routine's per-row loops (probabilistic,
  plain, and decision-value paths) are guarded with
  `#pragma omp parallel for num_threads(nt)`.
- `src/svm.cpp` — the libsvm SMO solver's per-iteration gradient / `G_bar`
  updates and the RBF `x_square` pre-compute carry
  `#pragma omp parallel for num_threads(...)`.

Both are bounded by the `e1071mc_threads` global (default 1), so the original
`svm` / `predict.svm` path stays single-threaded and bit-identical to upstream.

## Performance

The largest, most reliable speed-up comes from parallelising the independent,
repeated work (CV folds and prediction rows).  A single SVM solve cannot be
fully parallelised, because the libsvm SMO outer working-set loop is inherently
sequential; however the per-iteration gradient / `G_bar` updates and the RBF
`x_square` pre-compute are OpenMP-parallelised, so large single solves also
benefit (clamped to one thread below `n < 2000`, where the thread-spawn cost
would outweigh the parallel work).

- **k-fold CV** with `n_cores = p`: up to roughly `p×` faster, provided
  each fold's training time dominates the per-process spawn overhead.
  Empirically, a 10-fold CV on n = 8000, 6 features shows ~**3.4×
  speed-up with 6 cores** (measured during development; your mileage
  will vary with problem size and per-fold training time).
- **Prediction** on large `newdata`: scales roughly linearly with the
  number of cores while the per-row kernel cost dominates (C-level OpenMP).
- **Single-train, no CV** (`cross = 0`): the SMO outer loop is serial, but
  the inner per-iteration loops (and the RBF `x_square` pre-compute) run
  across threads when the build has `-fopenmp` and `n >= 2000`; the
  unmodified `svm.default` is used, so results are identical to `svm()`.

## Build & Test

```sh
R CMD build .
R CMD check e1071mc_1.7-17-1-20260822.tar.gz
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
