# SVM Multicore Function
# Provides parallel computing support for SVM training and prediction
# Uses OpenMP for parallel kernel computation and cross-validation

svmmulticore <-
function (x,
           y           = NULL,
           scale       = TRUE,
           type        = NULL,
           kernel      = "rbf",
           degree      = 3,
           gamma       = if (is.vector(x)) 1 else 1 / ncol(x),
           coef0       = 0,
           cost        = 1.5,
           nu          = 0.5,
           class.weights = NULL,
           cachesize   = 40,
           tolerance   = 1e-3,
           epsilon     = 0.1,
           shrinking   = TRUE,
           cross       = 0,
           probability = TRUE,
           fitted      = TRUE,
           nparallel   = NULL,
           ...)
{
    # Input validation
    if(is.null(degree)) stop("degree must not be NULL!")
    if(is.null(gamma)) stop("gamma must not be NULL!")
    if(is.null(coef0)) stop("coef0 must not be NULL!")
    if(is.null(cost)) stop("cost must not be NULL!")
    if(is.null(nu)) stop("nu must not be NULL!")
    if(is.null(epsilon)) stop("epsilon must not be NULL!")
    if(is.null(tolerance)) stop("tolerance must not be NULL!")
    if(is.null(cross)) stop("cross must not be NULL!")
    if(is.null(probability)) stop("probability must not be NULL!")
    
    # Default nparallel to available cores
    if(is.null(nparallel)) {
        if(requireNamespace("parallel", quietly=TRUE)) {
            nparallel <- detectCores() - 1
        } else {
            nparallel <- 4
        }
    }
    if(nparallel < 1) nparallel <- 1
    
    # Check for OpenMP support
    if(nparallel > 1) {
        if(Sys.getenv("OPENMP") == "") {
            warning("OpenMP not detected. Falling back to single-threaded computation.")
            nparallel <- 1
        } else {
            # Set number of threads
            Sys.setenv(OMP_NUM_THREADS = as.character(nparallel))
        }
    }
    
    # Scale data
    if(scale) {
        x <- scale_df(x)
        x.scale <- attributes(x, which="scaled:center")
        scale <- attributes(x, which="scaled:scale")
    }
    
    # Process y
    if(is.factor(y)) {
        lev <- levels(y)
        y <- as.integer(y)
    } else {
        lev <- unique(y)
    }
    
    # Determine number of classes
    nclass <- 2
    if(is.factor(y)) nclass <- length(lev)
    
    # Map kernel to C index
    kernel_idx <- switch(kernel,
        linear = 0,
        poly = 1,
        rbf = 2,
        sigmoid = 3,
        stop(paste(sQuote(kernel), "is not a supported kernel"))
    )
    
    # Map type to C index
    type_idx <- switch(type,
        C-classification = 0,
        nu-classification = 1,
        one-class = 2,
        eps-regression = 3,
        nu-regression = 4,
        type  # Use provided type
    )
    
    # Handle class weights
    weightlabels <- NULL
    if(!is.null(class.weights) && !is.null(lev)) {
        if(is.character(class.weights) && class.weights == "inverse") {
            class.weights <- 1 / table(y)
            names(class.weights) <- lev
        }
        if(!is.null(names(class.weights))) {
            weightlabels <- match(names(class.weights), lev)
        }
    }
    
    # Prepare sparse flag
    sparse <- FALSE
    
    # Call C function with multicore parameters
    model <- .C(R_svmtrain_multicore,
                as.double(if(sparse && requires.sparse.package) x else t(as.matrix(x))),
                as.integer(nrow(x)), as.integer(ncol(x)),
                as.double(y),
                as.integer(type_idx),
                as.integer(kernel_idx),
                as.integer(degree),
                as.double(gamma),
                as.double(coef0),
                as.double(cost),
                as.double(nu),
                as.integer(nclass),
                as.double(class.weights),
                as.double(cachesize),
                as.double(tolerance),
                as.double(epsilon),
                as.integer(shrinking),
                as.integer(cross),
                as.integer(probability),
                as.integer(nparallel),
                nclasses = integer(1),
                nr = integer(1),
                index = integer(NA),
                labels = integer(nclass),
                nSV = integer(nclass),
                rho = double(nclass * (nclass - 1) / 2),
                coefs = double(nrow(x) * (nclass - 1)),
                sigma = double(1),
                probA = double(nclass * (nclass - 1) / 2),
                probB = double(nclass * (nclass - 1) / 2),
                cresults = double(cross),
                ctotal1 = double(1),
                ctotal2 = double(1),
                error = ""
    )
    
    # Store parallel information in model object
    model$nparallel <- nparallel
    model$parallel_computation <- TRUE
    
    # Return model with parallel attributes
    class(model) <- c("svmmulticore", "svm")
    return(model)
}

# Wrapper for parallel prediction prediction
predict.svmmulticore <-
function (object, newdata,
           decision.values = FALSE,
           probability = FALSE,
           nparallel = object$nparallel,
           ...)
{
    # Check if model was trained with parallel computing
    if(is.null(object$nparallel)) {
        warning("Model was not trained with multicore. Running single-threaded prediction.")
        return(predict.svm(object, newdata, decision.values = decision.values,
                          probability = probability))
    }
    
    # Process newdata
    if(inherits(newdata, "Matrix")) {
        newdata <- as(newdata, "matrix.csr")
    }
    if(inherits(newdata, "simple_triplet_matrix")) {
        ind <- order(newdata$i, newdata$j)
        newdata <- new("matrix.csr",
                       ra = newdata$v[ind],
                       ja = newdata$j[ind],
                       ia = as.integer(cumsum(c(1, tabulate(newdata$i[ind])))),
                       dimension = c(newdata$nrow, newdata$ncol))
    }
    
    sparse <- inherits(newdata, "matrix.csr")
    if(object$sparse || sparse) {
        loadNamespace("SparseM")
    }
    
    act <- NULL
    if(is.vector(newdata) && is.atomic(newdata)) {
        newdata <- t(t(newdata))
    }
    if(sparse) {
        newdata <- SparseM::t(SparseM::t(newdata))
    }
    
    preprocessed <- !is.null(attr(newdata, "na.action"))
    
    rowns <- if(!is.null(rownames(newdata))) {
            rownames(newdata)
        } else {
            1:nrow(newdata)
        }
    
    # Apply scaling if needed
    if(!is.null(object$scaled)) {
        newdata[,object$scaled] <-
            scale_data_frame(newdata[,object$scaled, drop = FALSE],
                            center = object$x.scale$"scaled:center",
                            scale  = object$x.scale$"scaled:scale")
    }
    
    # Parallel prediction
    ret <- .C(R_svmpredict_multicore,
              as.integer(probability),
              as.double(if(sparsity(object)) object$SV@ra else t(object$SV)),
              as.integer(nrow(object$SV)), as.integer(ncol(object$SV)),
              as.integer(if(sparsity(object)) object$SV@ia else 0),
              as.integer(if(sparsity(object)) object$SV@ja else 0),
              as.double(as.vector(object$coefs)),
              as.double(object$rho),
              as.integer(object$nclasses),
              as.integer(object$tot.nSV),
              as.integer(object$labels),
              as.integer(object$nSV),
              as.integer(object$sparse),
              as.integer(decision.values),
              as.integer(if(probability) object$compprob else 0),
              as.double(if(probability && object$compprob) object$probA else 0),
              as.double(if(probability && object$compprob) object$probB else 0),
              
              as.double(if(sparsity(newdata)) newdata@ra else t(newdata)),
              as.integer(nrow(newdata)),
              as.integer(if(sparsity(newdata)) newdata@ia else 0),
              as.integer(if(sparsity(newdata)) newdata@ja else 0),
              as.integer(sparsity_test),
              
              parallel_threads = as.integer(nparallel),
              
              ret = double(nrow(newdata))
    )
    
    # Process results
    ret2 <- if(is.character(object$levels)) {
            factor(object$levels[ret$ret], levels = object$levels)
        } else if(object$type == 2) {
            ret$ret == 1
        } else if(!is.null(object$y.scale) && any(object$scaled)) {
            ret$ret * object$y.scale$"scaled:scale" + object$y.scale$"scaled:center"
        } else {
            ret$ret
        }
    
    names(ret2) <- rowns
    
    ret2
}
# Cross-validation with parallel support
svmcv.multicore <-
function (x,
           y           = NULL,
           scale       = TRUE,
           type        = NULL,
           kernel      = "radial",
           degree      = 3,
           gamma       = "scale",
           cost        = 1.5,
           nu          = NULL,
           epsilon     = 0.1,
           n           = 10,
           class.weights = NULL,
           probability = TRUE,
           nparallel = 4,
           ...)
{
    # Check for OpenMP support
    # if(requireNamespace("parallel", quietly=TRUE)) {
    #     nparallel <- detectCores()
    # } else {
    #     nparallel <- 1
    # }
    
    # Prepare data
    if(scale) {
        x_scaled <- scale_matrix(x)
        x.scale <- attributes(x_scaled)
    }
    
    # Set probability mode
    if(type %in% c(0, 1)) {
        probability <- TRUE
    } else {
        probability <- FALSE
    }
    
    # Handle gamma
    if(is.character(gamma)) {
        if(gamma == "scale") {
            gamma <- if (is.matrix(x)) {
                1 / max(sapply(lapply(attr(x, "scale"), "sd"), max))
            } else {
                1 / max(sapply(x, sd))
            }
        }
    }
    
    # Perform parallel cross-validation
    cv_results <- .C(R_svm_cross_validation_multicore,
                    x_scaled, y, n, nparallel,
                    type, kernel, degree, gamma,
                    cost, nu, epsilon,
                    n_folds = as.integer(n),
                    class_weights = NULL,
                    probability,
                    
                    n_iter = integer(1),
                    time_taken = double(1),
                  
                    objective = double(n),
                    accuracy = double(n)
    )
    
    # Return results
    ret <- list(
        objective = cv_results$objective,
        accuracy = cv_results$accuracy,
        n_fold = n,
        cost = cost,
        gamma = gamma,
        nparallel = nparallel
    )
    
    class(ret) <- "svmcvcrossvalidation.multicore"
    ret
}