#
# tune_mc: multicore / parallel variant of tune().
#
# Design concept (mirrors svm_mc and the prediction functions):
#   * A separate, additive function -- the original tune() is left untouched, so
#     the serial result is byte-identical to upstream.
#   * Fast path: when n_cores <= 1 the work is deferred to the unmodified
#     tune(), giving an exact drop-in result.
#    * Parallel path: the independent parameter combinations are distributed
#     across cores with parallel::mclapply, exactly as svm_mc distributes
#     cross-validation folds.  Each task trains one parameter combination over
#     all its (already fixed) folds and scores each on its hold-out set,
#     reproducing tune() unit-for-unit, so the aggregated errors are identical
#     to the serial run (mclapply's mc.set.seed makes per-task work deterministic).
#    * Parallelism is at the *process* level (one process per parameter
#     combination), NOT at the C level.  The C-level OpenMP thread count
#     (.svm_mc_set_threads) is intentionally left at 1: raising it per task would
#     oversubscribe the cores, exactly as svm_mc avoids by serialising its CV
#     folds at the C level.  The per-model training therefore runs serially, and
#     the speed-up comes from running many parameter combinations in parallel.
#
# Independent work unit: one parameter combination, evaluated over all its
# folds.  For each fold the model is trained on the in-fold rows and scored on
# the hold-out rows (the inner work of tune()); the nrepeat inner repetitions are
# kept inside the fold, as tune() does.  The number of tasks equals the number
# of parameter combinations.
#

tune_mc <-
function (METHOD, train.x, train.y = NULL, data = list(),
          validation.x = NULL, validation.y = NULL,
          ranges = NULL, predict.func = predict,
          tunecontrol = tune.control(),
          n_cores = NULL, set.seed = NULL,
          ...)
{
    n_cores <- .svm_mc_resolve_cores(n_cores)

    # Honor set.seed before branching so both the fast path (serial) and the
    # parallel path get an identical, reproducible fold assignment.
    if (!is.null(set.seed))
        set.seed(set.seed)

     # Fast path: defer to the unmodified tune() for an exact drop-in result.
    if (n_cores <= 1L) {
        return (tune(METHOD, train.x = train.x, train.y = train.y,
                     data = data,
                     validation.x = validation.x, validation.y = validation.y,
                     ranges = ranges, predict.func = predict.func,
                     tunecontrol = tunecontrol, ...))
        }

    call <- match.call()

    # ---------- internal helpers (identical to tune()) --------------------------------
    resp <- function(formula, data) {
        model.response(model.frame(formula, data))
    }

    classAgreement <- function (tab) {
        n <- sum(tab)
        if (!is.null(dimnames(tab))) {
            lev <- intersect(colnames(tab), rownames(tab))
            p0 <- sum(diag(tab[lev, lev])) / n
         } else {
            m <- min(dim(tab))
            p0 <- sum(diag(tab[1:m, 1:m])) / n
         }
        p0
    }

    # ---------- parameter handling (identical to tune()) -----------------------------
    if (tunecontrol$sampling == "cross")
        validation.x <- validation.y <- NULL
    useFormula <- is.null(train.y)
    if (useFormula && (is.null(data) || length(data) == 0))
        data <- model.frame(train.x)
    if (is.vector(train.x)) train.x <- t(t(train.x))
    if (is.data.frame(train.y))
        train.y <- as.matrix(train.y)

    if (!is.null(validation.x)) tunecontrol$fix <- 1
    n <- nrow(if (useFormula) data else train.x)
    perm.ind <- sample(n)
    if (tunecontrol$sampling == "cross") {
        if (tunecontrol$cross > n)
            stop(sQuote("cross"), " must not exceed sampling size!")
        if (tunecontrol$cross == 1)
            stop(sQuote("cross"), " must be greater than 1!")
    }
    train.ind <- if (tunecontrol$sampling == "cross")
        tapply(1:n, cut(1:n, breaks = tunecontrol$cross), function(x) perm.ind[-x])
    else if (tunecontrol$sampling == "fix")
        list(perm.ind[1:trunc(n * tunecontrol$fix)])
    else ## bootstrap
        lapply(1:tunecontrol$nboot,
               function(x) sample(n, n * tunecontrol$boot.size, replace = TRUE))

    # ---------- prepare the grid of independent tasks -------------------------------
    parameters <- if (is.null(ranges))
        data.frame(dummyparameter = 0)
    else
        expand.grid(ranges)
    p <- nrow(parameters)
    if (!is.logical(tunecontrol$random)) {
        if (tunecontrol$random < 1)
            stop("random must be a strictly positive integer")
        if (tunecontrol$random > p) tunecontrol$random <- p
        parameters <- parameters[sample(1:p, tunecontrol$random),]
        p <- nrow(parameters)
    }
    nfold <- length(train.ind)

    extra.args   <- list(...)

    # Precompute, for every fold, the training index, the hold-out data, and the
    # true response -- the same slicing tune() does for the hold-out / true.y.
    precompute.fold <- function (sample) {
        ti <- train.ind[[sample]]
        xout <- if (!is.null(validation.x))
            validation.x
        else if (useFormula)
            data[-ti,,drop = FALSE]
        else if (inherits(train.x, "matrix.csr"))
            train.x[-ti,]
        else
            train.x[-ti,,drop = FALSE]

        true.y <- if (!is.null(validation.y))
            validation.y
        else if (useFormula) {
            if (!is.null(validation.x))
                resp(train.x, validation.x)
            else
                resp(train.x, data[-ti,])
        } else
            train.y[-ti]

        list(train = ti, xout = xout, true.y = true.y)
    }

    folds <- lapply(seq_len(nfold), precompute.fold)

    # Train one parameter combination over all its folds and return its
    # aggregated error / variance.  This reproduces tune() exactly: the nrepeat
    # inner repetitions are kept inside the fold (as tune() does), and each fold
    # is an independent task distributed across cores via mclapply, mirroring how
    # svm_mc treats each CV fold as an independent task.
    one.combination <- function (para.set) {
        fold.errors <- numeric(nfold)
        for (sample in seq_len(nfold)) {
            fold <- folds[[sample]]
            ti   <- fold$train
            pars <- if (is.null(ranges))
                NULL
            else
                lapply(parameters[para.set,,drop = FALSE], unlist)

            repeat.errors <- numeric(tunecontrol$nrepeat)
            for (reps in seq_len(tunecontrol$nrepeat)) {
                model <- if (useFormula)
                    do.call(METHOD, c(list(train.x,
                                           data = data,
                                           subset = ti),
                                      pars, extra.args))
                else
                    do.call(METHOD, c(list(train.x[ti,],
                                           y = train.y[ti]),
                                      pars, extra.args))

                pred   <- predict.func(model, fold$xout)
                true.y <- fold$true.y
                if (is.null(true.y)) true.y <- rep(TRUE, length(pred))

                repeat.errors[reps] <- if (!is.null(tunecontrol$error.fun))
                    tunecontrol$error.fun(true.y, pred)
                else if ((is.logical(true.y) || is.factor(true.y)) &&
                          (is.logical(pred) || is.factor(pred) || is.character(pred)))
                    1 - classAgreement(table(pred, true.y))
                else if (is.numeric(true.y) && is.numeric(pred))
                    crossprod(pred - true.y) / length(pred)
                else
                    stop("Dependent variable has wrong type!")
             }
            fold.errors[sample] <- tunecontrol$repeat.aggregate(repeat.errors)
         }
        list(error = tunecontrol$sampling.aggregate(fold.errors),
             var    = tunecontrol$sampling.dispersion(fold.errors))
     }

    # Distribute parameter combinations across cores, mirroring svm_mc's
    # mclapply over folds.  Each task is one parameter combination.
    if (n_cores > 1L)
        results <- mclapply(seq_len(p), one.combination,
                            mc.cores = n_cores, mc.set.seed = TRUE)
    else
        results <- lapply(seq_len(p), one.combination)

    model.errors   <- vapply(results, function (r) r$error, numeric(1))
    model.variances <- vapply(results, function (r) r$var,   numeric(1))

    # ---------- assemble and return (identical shape to tune()) ----------------------
    best <- which.min(model.errors)
    pars <- if (is.null(ranges))
        NULL
    else
        lapply(parameters[best,,drop = FALSE], unlist)
    structure(list(best.parameters   = parameters[best,,drop = FALSE],
                   best.performance = model.errors[best],
                   method            = if (!is.character(METHOD))
                   deparse(substitute(METHOD)) else METHOD,
                   nparcomb          = nrow(parameters),
                   train.ind         = train.ind,
                   sampling          = switch(tunecontrol$sampling,
                   fix = "fixed training/validation set",
                   bootstrap = "bootstrapping",
                   cross = if (tunecontrol$cross == n) "leave-one-out" else
                   paste(tunecontrol$cross,"-fold cross validation", sep="")
                    ),
                   performances      = if (tunecontrol$performances) cbind(parameters, error = model.errors, dispersion = model.variances),
                   n_cores           = as.integer(n_cores),
                   best.model        = if (tunecontrol$best.model) {
                       modeltmp <- if (useFormula)
                           do.call(METHOD, c(list(train.x, data = data),
                                             pars, extra.args))
                       else
                           do.call(METHOD, c(list(x = train.x,
                                                  y = train.y),
                                             pars, extra.args))
                       call[[1]] <- as.symbol("best.tune")
                       modeltmp$call <- call
                       modeltmp
                   }
                   ),
             class = c("tune_mc", "tune")
            )
}
