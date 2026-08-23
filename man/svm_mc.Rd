\name{svm_mc}
\alias{svm_mc}
\alias{svm_mc.default}
\alias{svm_mc.formula}
\alias{predict.svm_multicore}
\title{Multicore Support Vector Machines}
\description{
\code{svm_mc} is a multicore-capable variant of \code{\link{svm}}. It shares
the same interface and returns an object compatible with the base \code{"svm"}
class (it has the additional class \code{"svm_multicore"}), but performs
cross-validation folds and predictions in parallel using
\code{\link[parallel]{mclapply}}.
}
\usage{
\method{svm_mc}{formula}(formula, data = NULL, ..., subset, na.action =
 na.omit, scale = TRUE, n_cores = NULL)
\method{svm_mc}{default}(x, y = NULL, scale = TRUE, type = NULL,
 kernel = "radial", degree = 3,
 gamma = if (is.vector(x)) 1 else 1 / ncol(x), coef0 = 0, cost = 1,
 nu = 0.5, class.weights = NULL, cachesize = 40, tolerance = 0.001,
 epsilon = 0.1, shrinking = TRUE, cross = 0, probability = FALSE,
 fitted = TRUE, n_cores = NULL, ..., subset, na.action = na.omit)

\method{predict}{svm_multicore}(object, newdata,
 decision.values = FALSE, probability = FALSE,
 n_cores = object$n_cores, ..., na.action = na.omit)
}
\arguments{
    \item{formula}{a symbolic description of the model to be fit.}
    \item{data}{an optional data frame containing the variables in the model.}
    \item{x}{a data matrix, a data frame, or a vector.}
    \item{y}{a response vector with one label for each row/component of
      \code{x}. Can be either a factor (for classification tasks) or a numeric
      vector (for regression).}
    \item{scale}{a logical vector indicating the variables to be scaled. If
      \code{scale} is of length 1, the value is recycled as many times as
      needed.}
    \item{type}{\code{svm_mc} can be used as a classification machine, as a
      regression machine, or for novelty detection. Valid options are
      \code{C-classification}, \code{nu-classification},
      \code{one-classification}, \code{eps-regression}, and
      \code{nu-regression}.}
    \item{kernel}{the kernel used in training and predicting. Valid options are
      \code{linear}, \code{polynomial}, \code{radial}, and \code{sigmoid}.}
    \item{degree}{parameter needed for kernel of type \code{polynomial}
      (default: 3).}
    \item{gamma}{parameter needed for all kernels except \code{linear}
      (default: 1/data dimension).}
    \item{coef0}{parameter needed for kernels of type \code{polynomial} and
      \code{sigmoid} (default: 0).}
    \item{cost}{cost of constraints violation, i.e.\ the \eqn{C}-constant of the
      regularization term in the Lagrange formulation (default: 1).}
    \item{nu}{parameter needed for \code{nu-classification},
      \code{nu-regression}, and \code{one-classification} (default: 0.5).}
    \item{class.weights}{a named vector of weights for the different classes,
      used for asymmetric class sizes. Specifying \code{"inverse"} will choose
      the weights inversely proportional to the class distribution.}
    \item{cachesize}{cache memory in MB (default 40).}
    \item{tolerance}{tolerance of termination criterion (default: 0.001).}
    \item{epsilon}{epsilon in the insensitive-loss function (default: 0.1).}
    \item{shrinking}{option whether to use the shrinking-heuristics
      (default: \code{TRUE}).}
    \item{cross}{if an integer value \eqn{k > 0} is specified, a \eqn{k}-fold
      cross validation on the training data is performed; the folds are
      distributed across cores.}
    \item{probability}{logical indicating whether the model should allow for
      probability predictions.}
    \item{fitted}{logical indicating whether the fitted values should be
      computed and included in the model or not (default: \code{TRUE}).}
    \item{n_cores}{the number of parallel workers to use. Defaults to
      \code{detectCores() - 1} from package \pkg{parallel}. A value
      \code{<= 1} disables parallelism (results are then computed exactly as by
      \code{\link{svm}}). For \code{predict.svm_multicore} the default is
      inherited from the trained model via
      \code{object$n_cores}.}
     \item{object}{a fitted \code{svm_multicore} object, as returned by
      \code{svm_mc}.}
    \item{newdata}{the newdata to predict on.}
    \item{decision.values}{logical indicating whether decision values should be
      returned as an attribute of the prediction.}
    \item{subset}{an index vector specifying the cases to be used in the
      training sample (note: if given, this argument must be named).}
    \item{na.action}{a function to specify the action to be taken if
      \code{NA}s are found; the default is \code{na.omit}.}
    \item{\dots}{additional arguments forwarded to \code{\link{svm}} or
      \code{\link{predict.svm}}.}
}
\value{
  An object of class \code{"svm_multicore"} (which derives from \code{"svm"}),
  so that all base-class methods
   (\code{\link{print.svm}}, \code{\link{summary.svm}}, \code{\link{plot.svm}},
   \code{\link{coef.svm}}) continue to work. The field \code{n_cores} records
  the number of workers used. For \code{predict.svm_multicore}, the return value
  has the same form as \code{\link{predict.svm}} (a factor, a logical vector, or
  a numeric vector, with optional \code{decision.values} and
   \code{probabilities} attributes).
}
 \details{
    \bold{Cross-validation (\code{cross > 0}).}  K-fold CV trains
    \code{cross} independent models, so the folds are distributed across cores
   with \code{\link[parallel]{mclapply}} (one process per fold).  This is the
   regime where the parallel speed-up is largest.

    \bold{Single training (\code{cross = 0}).}  A single model is trained by the
   compiled libsvm SMO (Sequential Minimal Optimization) solver.  SMO's outer
   working-set loop is inherently sequential (each iteration updates the whole
   gradient, which the next iteration reads), so the \emph{outer} loop cannot be
   split across cores.  The \emph{inner} work of each iteration \emdash the
   per-row gradient update and the \code{G_bar} update
   (\file{src/svm.cpp}) and the RBF \code{x_square} pre-compute \emdash is
   guarded with \code{\#pragma omp parallel for} and becomes parallel when the
   package is compiled with \code{-fopenmp}.  Because the work of each iteration
   is small relative to the thread-spawn cost, this C-level parallelism helps
   only on large problems; on small problems it can be slightly slower than the
   serial path.

    \bold{Prediction.}  The per-row prediction loops inside the compiled
   \code{Rsvm.c} routine (the main prediction, the decision-value loop, and the
   probabilistic loop) are guarded with \code{\#pragma omp parallel for} and,
   since the rows are independent, give a large, result-preserving speed-up when
   compiled with \code{-fopenmp}.

    \bold{Compile-time behaviour.}  All of the above OpenMP pragmas are standard
   C and compile to a strict sequential loop on any toolchain; they become
   parallel automatically only when the package is compiled with \code{-fopenmp}
    (on this machine the active R \file{Makeconf} is configured with
    \code{-fopenmp}).  A global C flag \code{e1071mc_threads} bounds every
   OpenMP region to \code{n_cores} and resets to 1 after each call, so the
   original suite (\code{\link{svm}} / \code{predict.svm}) is unaffected and, in
   the serial build, byte-identical to upstream.
 }
 \note{
   The base \code{\link{svm}} and \code{\link{predict.svm}} functions are left
   completely unchanged; \code{svm_mc} / \code{predict.svm_multicore} are
   independent, additive implementations that reuse \code{svm.default} and
    \code{predict.svm}, and only \emph{additively} raise the C-level OpenMP
   thread count around their own calls.
 }
\seealso{
    \code{\link{svm}}, \code{\link{predict.svm}}, \code{\link{tune.svm}},
    \code{detectCores} from \pkg{parallel}
}
\examples{
data(iris)

# classification, parallel cross-validation
m <- svm_mc(Species ~ ., data = iris, cross = 5, n_cores = 2)
print(m)
summary(m)
m$accuracies

# traditional / matrix interface
x <- subset(iris, select = -Species)
y <- Species
m <- svm_mc(x, y, cross = 5, n_cores = 2)

# parallel prediction
pred <- predict(m, x, n_cores = 2)
table(pred, y)

# regression, vector x
set.seed(1)
xr <- 1:100
yr <- 0.5 * xr + rnorm(100, sd = 0.3)
mr <- svm_mc(xr, yr, cross = 5, n_cores = 2)
pred <- predict(mr, xr, n_cores = 2)
}
\keyword{neural}
\keyword{classif}
\keyword{parallel}
