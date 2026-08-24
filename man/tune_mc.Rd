\name{tune_mc}
\alias{tune_mc}
\title{Multicore Parameter Tuning by Grid Search}
\description{
\code{tune_mc} is a multicore-capable variant of \code{\link{tune}}. It shares
the same interface and returns an object compatible with the base \code{"tune"}
class (it has the additional class \code{"tune_mc"}), but distributes the
independent parameter combinations across cores using
\code{\link[parallel]{mclapply}} (one \R process per parameter combination,
exactly as \code{\link{svm_mc}} distributes cross-validation folds).
}
\usage{
tune_mc(METHOD, train.x, train.y = NULL, data = list(),
 validation.x = NULL, validation.y = NULL, ranges = NULL,
 predict.func = predict, tunecontrol = tune.control(),
 n_cores = NULL, set.seed = NULL, ...)
}
\arguments{
   \item{METHOD}{either the function to be tuned, or a character string
     naming such a function.}
   \item{train.x}{either a formula or a matrix of predictors.}
   \item{train.y}{the response variable if \code{train.x} is a predictor
     matrix. Ignored if \code{train.x} is a formula.}
   \item{data}{data, if a formula interface is used. Ignored, if
     predictor matrix and response are supplied directly.}
   \item{validation.x}{an optional validation set. Depending on whether a
     formula interface is used or not, the response can be
     included in \code{validation.x} or separately specified using
      \code{validation.y}. Only used for bootstrap and fixed validation
     set (see \code{\link{tune.control}}).}
   \item{validation.y}{if no formula interface is used, the response of
     the (optional) validation set. Only used for bootstrap and fixed validation
     set (see \code{\link{tune.control}}).}
   \item{ranges}{a named list of parameter vectors spanning the sampling
     space. The vectors will usually be created by \code{seq}.}
   \item{predict.func}{optional predict function, if the standard \code{predict}
     behavior is inadequate.}
   \item{tunecontrol}{object of class \code{"tune.control"}, as created by the
     function \code{tune.control()}. If omitted, \code{tune.control()}
     gives the defaults.}
   \item{n_cores}{the number of parallel workers to use. Defaults to
     \code{detectCores() - 1} from \pkg{parallel} (via the same resolver used
     by \code{\link{svm_mc}}). A value \code{<= 1} disables parallelism and the
     work is deferred to the unmodified \code{\link{tune}}, giving an exact
     drop-in result.}
   \item{set.seed}{optional integer. If supplied, \code{set.seed()} is called
     before the training-fold split is drawn, so that the fold assignment agrees
     between serial and parallel runs (and across machines) and the result is
     reproducible.}
   \item{\dots}{Further parameters passed to the training functions.}
}
\value{
  An object of class \code{"tune_mc"} (which derives from \code{"tune"}), so
  that all base-class methods
    (\code{\link{print.tune}}, \code{\link{summary.tune}}, \code{\link{plot.tune}})
  continue to work. In addition to the components returned by
  \code{\link{tune}} (\code{best.parameters}, \code{best.performance},
  \code{performances}, \code{train.ind}, \code{best.model}), the field
  \code{n_cores} records the number of workers used.
}
\details{
  The independent unit of work is one parameter combination. Each combination is
  evaluated over all its (already fixed) training/validation folds, exactly as
  \code{\link{tune}} does: for each fold the model is trained on the in-fold
  rows and scored on the hold-out rows, and the per-fold errors are aggregated
  with \code{repeat.aggregate} and \code{sampling.aggregate}. The combinations are
  then distributed across cores with \code{\link[parallel]{mclapply}}, mirroring how
  \code{\link{svm_mc}} distributes cross-validation folds.
  Because each fold is precomputed once (the train/hold-out split is fixed before
  dispatch), and the aggregation order is fixed, the parallel result is identical
  to the serial \code{\link{tune}} (up to floating-point summation order, which
  \code{mclapply}'s \code{mc.set.seed = TRUE} makes deterministic per task).
  Parallelism is at the \emph{process} level; the per-model training itself runs
  serially (the C-level OpenMP thread count is left at 1, so no core
  oversubscription occurs, exactly as \code{\link{svm_mc}} avoids).

  The base \code{\link{tune}} function is left completely unchanged;
  \code{tune_mc} is an independent, additive implementation that
  re-uses the original tuning logic and only parallelises the grid search.
}
\seealso{\code{\link{tune}}, \code{\link{tune.control}},
   \code{\link{svm_mc}}, \code{detectCores} from \pkg{parallel}}
\examples{
  data(iris)

  ## tune `svm' for classification with RBF-kernel, in parallel across cores,
  ## using one fixed training/validation split.
  obj <- tune_mc(svm, Species~., data = iris,
                 ranges = list(gamma = 2^(-1:2), cost = 2^(2:5)),
                 tunecontrol = tune.control(sampling = "fix"),
                 n_cores = 2, set.seed = 123)
  summary(obj)
  plot(obj)

  ## 5-fold cross validation, parallel grid search, reproducible split
  obj2 <- tune_mc(svm, Species~., data = iris,
                  ranges = list(gamma = 2^(-2:-1), cost = 2^(3:4)),
                  tunecontrol = tune.control(sampling = "cross", cross = 5),
                  n_cores = 2, set.seed = 1)
  summary(obj2)

  ## n_cores = 1 falls back to the unmodified tune() (exact drop-in)
  obj3 <- tune_mc(lm, mpg~., data = mtcars,
                  n_cores = 1, set.seed = 7)
  print(obj3)
}
\keyword{models}
\keyword{parallel}
