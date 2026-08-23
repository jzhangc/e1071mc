svm_mc <-
function (x, ...)
    UseMethod ("svm_mc")

svm_mc.formula <-
function (formula, data = NULL, ..., subset, na.action = na.omit,
          scale = TRUE, n_cores = NULL)
{
    call <- match.call()
    if (!inherits(formula, "formula"))
        stop("method is only for formula objects")

    m <- match.call(expand.dots = FALSE)
    if (inherits(eval.parent(m$data), "matrix"))
        m$data <- as.data.frame(eval.parent(m$data))
    m$... <- NULL
    m$scale <- NULL
    m$n_cores <- NULL
    m[[1L]] <- quote(stats::model.frame)
    m$na.action <- na.action
    m <- eval(m, parent.frame())
    Terms <- attr(m, "terms")
    attr(Terms, "intercept") <- 0
    x <- model.matrix(Terms, m)
    y <- model.extract(m, "response")
    attr(x, "na.action") <- attr(y, "na.action") <- attr(m, "na.action")
    attr(x, "xlevels") <- .getXlevels(Terms, m)
    if (length(scale) == 1)
        scale <- rep(scale, ncol(x))
    if (any(scale)) {
        remove <- unique(c(which(labels(Terms) %in%
                                  names(attr(x, "contrasts"))),
                            which(!scale)
                              )
                            )
        scale <- !attr(x, "assign") %in% remove
      }
    class(x) <- c("svm.formula", class(x))

    ret <- svm_mc.default(x, y, scale = scale, n_cores = n_cores,
                          na.action = na.action, ...)

    ret$call <- call
    ret$call[[1]] <- as.name("svm_mc")
    ret$terms     <- Terms
    if (!is.null(attr(m, "na.action")))
        ret$na.action <- attr(m, "na.action")
    class(ret) <- c("svm.formula", class(ret))
    return (ret)
}

svm_mc.default <-
function (x,
          y              = NULL,
          scale          = TRUE,
          type           = NULL,
          kernel         = "radial",
          degree         = 3,
          gamma          = if (is.vector(x)) 1 else 1 / ncol(x),
          coef0          = 0,
          cost           = 1,
          nu             = 0.5,
          class.weights = NULL,
          cachesize      = 40,
          tolerance      = 0.001,
          epsilon        = 0.1,
          shrinking      = TRUE,
          cross          = 0,
          probability    = FALSE,
          fitted         = TRUE,
          n_cores        = NULL,
             ...,
          subset,
          na.action = na.omit)
{
    n_cores <- .svm_mc_resolve_cores(n_cores)

        # Resolve model type / kernel the same way svm.default does, purely to
      # decide how to aggregate cross-validation results.  The actual fit and
      # CV use the original svm.default so behaviour matches the base package.

        # C-level OpenMP thread count for a single SMO train.  svm.default
      # (and predict.svm) always run single-threaded, because e1071mc_threads
      # defaults to 1: only the wrapper below temporarily raises it, then
      # restores 1, so the original suite is bit-identical to upstream.
    if (is.null(type)) type.res <-
        if (is.null(y)) "one-classification"
        else if (is.factor(y)) "C-classification"
        else "eps-regression"
    else
        type.res <- type
    type.i <- pmatch(type.res, c("C-classification",
                                   "nu-classification",
                                   "one-classification",
                                   "eps-regression",
                                   "nu-regression"), 99) - 1
     if (type.i > 10) stop("wrong type specification!")

          # subset / sparse / extra args have no default in svm.default; forward them
         # only when actually supplied.  missing() is evaluated here, in
         # svm_mc.default's own frame, and the captured lists are spread into each
         # training call via do.call().
    subset.args <- if (!missing(subset)) list(subset = subset) else list()
    extra.args  <- list(...)

          # Fast path: no cross-validation OR only one core.
         # Deferring to the unmodified svm.default gives an exact, drop-in result.
    if (cross <= 0L || n_cores <= 1L) {
        m <- .svm_mc_run_train(function() do.call(svm.default, c(list(
             x = x, y = y, scale = scale, type = type,
             kernel = kernel, degree = degree, gamma = gamma,
             coef0 = coef0, cost = cost, nu = nu,
             class.weights = class.weights,
             cachesize = cachesize, tolerance = tolerance,
             epsilon = epsilon, shrinking = shrinking,
             cross = cross, probability = probability,
             fitted = fitted,
             na.action = na.action),
             subset.args, extra.args)), n_cores)
        m$n_cores <- as.integer(n_cores)
        class(m) <- c("svm_multicore", class(m))
        return (m)
       }

          # Parallel path: cross > 0 and n_cores > 1.
        # 1. Train the final full-data model with the unmodified svm.default,
           # cross forced to 0 (CV handled below, in parallel).
    final.model <- .svm_mc_run_train(function() do.call(svm.default, c(list(
        x = x, y = y, scale = scale, type = type,
        kernel = kernel, degree = degree, gamma = gamma,
        coef0 = coef0, cost = cost, nu = nu,
        class.weights = class.weights,
        cachesize = cachesize, tolerance = tolerance,
        epsilon = epsilon, shrinking = shrinking,
        cross = 0L, probability = probability,
        fitted = FALSE,
        na.action = na.action),
        subset.args, extra.args)), n_cores)

    final.model$n_cores <- as.integer(n_cores)

      # 2. Parallel / serial cross-validation, folds distributed across cores.
    if (cross > 0L) {
        cv <- .svm_mc_cross_validate(
            x = x, y = y,
            type.res = type.res, scale = scale,
            kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0,
            cost = cost, nu = nu, class.weights = class.weights,
            cachesize = cachesize, tolerance = tolerance, epsilon = epsilon,
            shrinking = shrinking, probability = probability,
            n_cores = n_cores, cross = cross,
            subset = subset, na.action = na.action
            )
        if (type.i > 2) {
            final.model$MSE         <- cv$results
            final.model$tot.MSE     <- cv$total1
            final.model$scorrcoeff  <- cv$total2
           } else {
            final.model$accuracies    <- cv$results
            final.model$tot.accuracy  <- cv$total1
           }
        }

      # 3. Fitted values / residuals (mirrors svm.default with fitted = TRUE).
    if (fitted) {
        xhold <- x
        fit <- na.action(predict(final.model, xhold, decision.values = TRUE))
        final.model$fitted <- fit
        final.model$decision.values <- attr(fit, "decision.values")
        attr(fit, "decision.values") <- NULL
        if (type.i > 1)
            final.model$residuals <- na.action(y) - final.model$fitted
       }

    class(final.model) <- c("svm_multicore", "svm")
    return (final.model)
}

#
# Cross-validation: scale the training data *once* globally (exactly as
# svm.default does), then train each fold with the unmodified svm.default on
# its in-fold rows (scale = FALSE, since data is already scaled) and score the
# held-out fold.  The folds are independent and are run in parallel via
# parallel::mclapply.
#
.svm_mc_cross_validate <-
function (x, y, type.res, scale, kernel, degree, gamma, coef0,
           cost, nu, class.weights, cachesize, tolerance, epsilon, shrinking,
           probability, n_cores, cross,
           subset, na.action)
{
      # Replicate svm.default's preprocessing (subsetting, NA, global scaling)
     # so the CV runs on the same scaled data the base package would use.
    pp <- .svm_mc_preprocess(x = x, y = y, scale = scale, type.res = type.res,
                             subset = subset, na.action = na.action)
    xs  <- pp$x
    ys  <- pp$y
    ysc <- pp$y.scale
    type.i <- pmatch(type.res, c("C-classification","nu-classification",
                                 "one-classification","eps-regression",
                                 "nu-regression"), 99) - 1

    n      <- nrow(xs)
    if (cross > n)
        stop(sQuote("cross"), " cannot exceed the number of observations!")

    fold.id <- 1L + ((0L:(n - 1L)) %% cross)

      # Per-fold training arguments.  scale is FALSE because xs is already
     # globally scaled; everything else is forwarded unchanged.
    fold.args <- function (tr, te) list(
        scale         = FALSE,
        type          = type.res,
        kernel        = kernel,
        degree        = degree,
        gamma         = gamma,
        coef0         = coef0,
        cost          = cost,
        nu            = nu,
        class.weights = if (type.i < 3) class.weights else NULL,
        cachesize     = cachesize,
        tolerance     = tolerance,
        epsilon       = epsilon,
        shrinking     = shrinking,
        cross         = 0L,
        probability   = probability,
        fitted         = FALSE,
        na.action      = na.pass
      )

    one.fold <- function (f) {
        tr <- which(fold.id != f)
        te <- which(fold.id == f)

        a   <- fold.args(tr, te)
        a$x <- xs[tr, , drop = FALSE]
        a$y <- ys[tr]
        m   <- do.call(svm.default, a)
        p   <- as.vector(predict(m, xs[te, , drop = FALSE]))

        if (type.i > 2) {
              # Regression / one-class.  Both p and ys[te] are in the
            # (possibly scaled) response units, exactly as svm.default feeds the
            # C code; we therefore accumulate stats in scaled units here and
            # convert to the original scale only at aggregation (using the same
            # crossprod(y.scale$scale) factor svm.default's C path uses).
            yy   <- as.numeric(ys[te])
            err <- sum((p - yy)^2)
            list(mse = err / length(te), ntest = length(te),
                 sumv = sum(p), sumy = sum(yy),
                 sumvv = sum(p^2), sumyy = sum(yy^2),
                 sumvy = sum(p * yy), total_error = err)
           } else {
            yy <- as.vector(ys[te])
            ok <- sum(p == yy)
            list(acc = 100.0 * ok / length(te),
                 correct = ok, ntest = length(te))
          }
      }

    if (n_cores > 1L)
        res <- mclapply(seq_len(cross), one.fold,
                        mc.cores = n_cores, mc.set.seed = TRUE)
    else
        res <- lapply(seq_len(cross), one.fold)

    if (type.i > 2) {
        total_error <- sum(vapply(res, function (r) r$total_error, numeric(1)))
        sumv   <- sum(vapply(res, function (r) r$sumv,  numeric(1)))
        sumy   <- sum(vapply(res, function (r) r$sumy,  numeric(1)))
        sumvv <- sum(vapply(res, function (r) r$sumvv, numeric(1)))
        sumyy <- sum(vapply(res, function (r) r$sumyy, numeric(1)))
        sumvy <- sum(vapply(res, function (r) r$sumvy, numeric(1)))
        # The per-fold errors above are in (scaled) response units; convert to
        # the original scale exactly as svm.default does, via the
        # crossprod(y.scale$scale) factor (= sd^2 for a single regressor).
         sf <- if (!is.null(ysc))
                  as.numeric(crossprod(ysc$"scaled:scale"))
                  else 1
        results     <- vapply(res, function (r) r$mse, numeric(1)) * sf
        total1      <- (total_error / n) * sf
        total2      <- (n * sumvy - sumv * sumy)^2 /
                       ((n * sumvv - sumv^2) * (n * sumyy - sumy^2))
        list(results = results, total1 = total1, total2 = total2)
        } else {
        results       <- vapply(res, function (r) r$acc, numeric(1))
        total.correct <- sum(vapply(res, function (r) r$correct, integer(1)))
        total1        <- 100.0 * total.correct / n
        list(results = results, total1 = total1, total2 = NA_real_)
      }
}

#
# Global preprocessing that replicates svm.default's subsetting, NA handling and
# scaling, so that CV folds train on already-scaled data.  Returned y is in the
# same (possibly scaled) units svm.default feeds to the C code.
#
.svm_mc_preprocess <-
function (x, y = NULL, scale = TRUE, type.res, subset, na.action)
{
    x <- as.matrix(x)
    if (length(scale) == 1)
        scale <- rep(scale, ncol(x))

    if (!missing(subset)) {
        x <- x[subset, ]
        y <- y[subset]
      }
    if (is.null(y)) {
        x <- na.action(x)
      } else {
        df <- na.action(data.frame(y, x, check.names = FALSE))
        y  <- df[, 1]
        x  <- as.matrix(df[, -1, drop = FALSE], rownames.force = TRUE)
      }

    y.scale <- NULL
    if (any(scale)) {
        co <- !apply(x[, scale, drop = FALSE], 2, var)
        if (any(co)) {
            warning(paste("Variable(s)",
                          paste(sQuote(colnames(x[, scale, drop = FALSE])[co]),
                                sep = "", collapse = " and "),
                      "constant. Cannot scale data.")
                    )
            scale <- rep(FALSE, ncol(x))
          } else {
            xtmp <- scale_data_frame(x[, scale])
            x[, scale] <- xtmp
            # (x.scale kept from final.model, not needed for CV here)

             # Regression / nu-svr with scaling: svm.default scales the response
               # globally; mirror that so CV runs on the same data.  The check
               # is the same as in svm.default: type index > 2 (eps/nu
               # regression) and y numeric.
            ti <- pmatch(type.res, c("C-classification","nu-classification",
                    "one-classification","eps-regression","nu-regression"),
                    99L) - 1L
            if (!is.null(y) && (is.numeric(y)) && (ti > 2)) {
                ys   <- scale(y)
                y.scale <-
                    attributes(ys)[c("scaled:center", "scaled:scale")]
                y <- as.vector(ys)
              }
          }
      }

    list(x = x, y = y, y.scale = y.scale)
}

predict.svm_multicore <-
function (object, newdata,
          decision.values = FALSE,
          probability = FALSE,
          n_cores = object$n_cores,
            ...,
          na.action = na.omit)
{
    if (missing(newdata))
        return (fitted(object))

    n_cores <- .svm_mc_resolve_cores(n_cores)

        # Single core: identical path to the original predict.svm (serial C loop).
    if (n_cores <= 1L)
        return (predict.svm(object, newdata,
                            decision.values = decision.values,
                            probability = probability,
                            na.action = na.action))

        # Multi-core: C-level OpenMP.  Raise the C thread count and call
       # predict.svm exactly once; the per-row loop in src/Rsvm.c (svmpredict)
       # then runs across the OpenMP team, and each row is independent, so the
       # result is identical to predict.svm run serially.  The thread count is
       # restored to 1 on exit so the original suite stays serial.
    .svm_mc_set_threads(n_cores)
    on.exit(.svm_mc_set_threads(1L), add = TRUE)
    return (predict.svm(object, newdata,
                        decision.values = decision.values,
                        probability = probability,
                        na.action = na.action))
}

.svm_mc_resolve_cores <- function (n_cores)
{
    if (is.null(n_cores)) {
        if (requireNamespace("parallel", quietly = TRUE))
            n_cores <- parallel::detectCores() - 1L
        else
            n_cores <- 1L
       }
    n_cores <- as.integer(n_cores)
    if (is.na(n_cores) || n_cores < 1L)
        n_cores <- 1L
    return (n_cores)
}

#
# Set (and restore) the C-level OpenMP thread count used by a *single*
# SMO training call.  The original suite (svm / predict.svm) always runs
# single-threaded because this is 1 by default; only svm_mc temporarily raises
# it around its own training, then puts it back to 1, so the original suite
# remains bit-identical to upstream.
#
.svm_mc_set_threads <- function (n)
{
    n <- as.integer(n)
    if (is.na(n) || n < 1L) n <- 1L
    .C(R_svm_mc_set_threads, as.integer(n))[1L]
}

#
# Run a single svm.default training with the C-level OpenMP SMO inner loops
# parallelised over `n_cores` threads, then restore the C thread count to 1 on
# exit (so surrounding serial C calls, e.g. the CV folds, are not oversubscribed).
#
.svm_mc_run_train <- function (fn, n_cores)
{
    .svm_mc_set_threads(n_cores)
    on.exit(.svm_mc_set_threads(1L), add = TRUE)
    fn()
}

#
# subset is an argument of svm.default (and svm_mc.default) with no default;
# referencing an unbound subset errors.  In svm_mc.default the value
#     subset.args <- if (!missing(subset)) list(subset = subset) else list()
# is built once (the test is evaluated in svm_mc.default's frame, where
# missing() is valid) and spread into every training call via do.call.
#
