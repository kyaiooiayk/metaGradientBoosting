"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access XGBoost imports
"""

import os
from .log import get_logger

from MGB.Modules import Modules
from MGB.Decorators import timer, logger
from MGB.XGBoostWrapper import XGBoostWrapper
from MGB.CatBoostWrapper import CatBoostWrapper

M = Modules()


class MetaGradientBoosting():
    """Meta gradient boosting.

    An ensamble of different gradient boosting algorithms.
    """

    # Initialise the object
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.frameworks = [i.lower() for i in params_dict["frameworks"]]
        self.X = params_dict["X"]
        self.y = params_dict["y"]
        self.split_method = params_dict["split_method"]
        self.test_size = params_dict["test_size"]
        self.eval_metric = [i.lower() for i in params_dict["eval_metric"]]
        self.verbose = params_dict["verbose"]
        self.metrics = [i.lower() for i in params_dict["metrics"]]
        self.problem = params_dict["problem"].lower()
        self.early_stopping_rounds = params_dict["early_stopping_rounds"]
        self.CV_type = params_dict["CV_type"]
        self.CV_n_splits = params_dict["CV_n_splits"]
        self.random_state = params_dict["state"]
        self.modelXGB = None
        self.modelXGB_not_fitted = None
        self.Set = None
        self.verbose = params_dict["verbose"]
        self.resultsXGB = None

        self.log_file_name = "MGB_logfile"
        self.log_file_dir = "./"
        try:
            # If the file is already present, then remove it!
            os.remove(os.path.join(self.log_file_dir,
                      self.log_file_name + ".log"))
        except:
            pass

        self.logger = get_logger(
            log_file_name=self.log_file_name, log_dir=self.log_file_dir)

        self._welcome_message()
        self._check_module()

    def _welcome_message(self):
        """Print welcome message.

        This is a private method.
        """

        self.logger.info("|")
        self.logger.info("|-----------------------|")
        self.logger.info("|- MGB                  |")
        self.logger.info("|- MetaGradientBoosting |")
        self.logger.info("|-----------------------|")
        self.logger.info("|")

    @logger
    def _check_module(self):
        """Check framework module availability
        """

        def _print_error_message(framework_name):
            self.logger.error("|")
            self.logger.error("|- " + framework_name + " is not installed!")
            self.logger.error("|- Either install it or use another framework!")
            self.logger.error("|")
            M.sys.exit()

        for framework_temp in self.frameworks:
            if framework_temp.lower() == "xgboost":
                try:
                    import xgboost
                    self.logger.debug("|-" + framework_temp +
                                      " successfully imported!")
                except:
                    _print_error_message(framework_temp)
            if framework_temp.lower() == "lightgbm":
                try:
                    import lightgbm
                    self.logger.debug("|-" + framework_temp +
                                      " successfully imported!")
                except:
                    _print_error_message(framework_temp)
            if framework_temp.lower() == "catboost":
                try:
                    import catboost                    
                    self.logger.debug("|-" + framework_temp +
                                      " successfully imported!")
                except:
                    _print_error_message(framework_temp)


    @logger
    def data_summary(self):
        """Data summary.

        Get the data summary. This includes things such as shape, type 
        etc ....

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.logger.info("|")
        self.logger.info("|- Feature X shape: " + str(self.X.shape))
        self.logger.info("|- Target y shape: " + str(self.y.shape))
        self.logger.info("|")

    @logger
    def modelling_summary(self):
        """Modelling summary.

        Get the modelling summary. This includes things such chosen frameworks,
        splitting and cross-validation strategies.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.logger.info("|")
        self.logger.info("|- Chosen frameworks: " + str(self.frameworks))
        self.logger.info("|- Problem's type: " + str(self.problem))
        self.logger.info("|- Split method: " + str(self.split_method))
        self.logger.info("|- Test size: " + str(self.test_size))
        self.logger.info("|- Evaluation metric: " + str(self.eval_metric))
        self.logger.info("|- Verbose: " + str(self.verbose))
        self.logger.info("|- Metrics: " + str(self.metrics))
        self.logger.info("|- Early stopping rounds: " +
                         str(self.early_stopping_rounds))
        self.logger.info("|")

    @logger
    def _split_dataset(self):
        """Split dataset.

        Decide how to spit the dataset.
        method = 2 splits dataset in train + test
        method = 3 splits datset in train + validation + test

        Parameters
        ----------
        method : int
            2 split the set into training and test
            3 split the set into training, validation and test

        test_size : float
            Provides in percentage the test size

        Returns:
        --------
        Set : dict[str, float]
            Dictionary containing the split dataset
        """

        self.logger.info("|- Splitting the dataset")
        self.logger.info("|- Chosen splitting strategy: " +
                         str(self.split_method))

        def _minimumSplit():
            Set = {}
            X_, X_test, y_, y_test = M.train_test_split(self.X,
                                                        self.y,
                                                        test_size=self.test_size,
                                                        random_state=self.random_state,
                                                        shuffle=True)
            Set["X_test"] = X_test
            Set["X_train"] = X_
            Set["y_test"] = y_test
            Set["y_train"] = y_

            return Set

        def _threeSetSplit():
            Set = {}
            Set_min = _minimumSplit()
            X_train, X_val, y_train, y_val = M.train_test_split(Set_min["X_train"],
                                                                Set_min["y_train"],
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                shuffle=True)
            Set["X_test"] = Set_min["X_test"]
            Set["X_train"] = X_train
            Set["X_val"] = X_val
            Set["y_test"] = Set_min["y_test"]
            Set["y_train"] = y_train
            Set["y_val"] = y_val

            return Set

        if int(self.split_method) == 3:
            Set = _threeSetSplit()
        else:
            Set = _minimumSplit()

        """
        is split method =!3 hence is 2
        We'll use the test as a validation test, but will cause some leakage
        in doing so
        """
        if "X_val" not in Set.keys():
            Set["X_val"] = Set["X_test"]
            Set["y_val"] = Set["y_test"]

        # Print on console the dimensions of the sets
        for temp_key in Set.keys():
            self.logger.info("|- Checking " + temp_key + " set dimensions: " + str(len(
                Set[temp_key])) + ", " + str(len(Set[temp_key])))

        return Set

    @logger
    def fit(self):
        """Fit the models.

        This is where each submodle is explictly instantiate and run
        """

        self.Set = self._split_dataset()

        self.logger.info("|- Framework list: " + str(self.frameworks))
        for model in self.frameworks:
            if "xgboost" in self.frameworks:
                XGB = XGBoostWrapper(self.logger, self.params_dict, self.Set)
                self.modelXGB_not_fitted = XGB.model_not_fitted
                model = XGB._fit()
                self.modelXGB = model
                self.resultsXGB = self.modelXGB.evals_result()
            if "catboost" in self.frameworks:
                CB = CatBoostWrapper(self.logger, self.params_dict, self.Set)
                self.modelCB_not_fitted = CB.model_not_fitted
                model = CB._fit()
                self.modelCB = model
                self.resultsCB = self.modelCB.evals_result()

    @logger
    def _fancy_plot(self, ax, col_No=3):
        """Fancy plot.

        Just add some fancy grid and label. Essentially some
        boiler plate code moved into a function.

        Parameters
        ----------
        ax : axis instance
            Axis instance
        col_No : int
            No of columns

        Returns
        -------
        ax : axis instance
            Modified axis instance
        legendObejct : legend instance
            Modified legend instance
        """

        ax.tick_params(which='major', direction='in', length=10, width=2)
        ax.tick_params(which='minor', direction='in', length=6, width=2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.grid()
        ax.minorticks_on()

        legend_object = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                  ncol=col_No, fontsize=20, fancybox=True, shadow=False,
                                  facecolor="w", framealpha=1)
        legend_object.get_frame().set_linewidth(2.0)
        legend_object.get_frame().set_edgecolor("k")

        return ax, legend_object

    @logger
    @timer
    def plot_lc(self, model_names):
        """Plot learning curves.        

        Parameters
        ----------

        Returns
        -------        
        """

        self.logger.info("|")
        self.logger.info("|- Plotting learnig curves for: " + str(model_names))

        for model_name_temp in model_names:
            if model_name_temp.lower() == "xgboost":
                results = self.resultsXGB

            M.rcParams['font.size'] = 20
            M.rcParams['figure.figsize'] = 15, 6

            # retrieve performance metrics
            epochs = len(results['validation_0'][self.eval_metric[0]])
            x_axis = range(0, epochs)

            # Plotting in figure per metric
            for temp_metric in self.eval_metric:
                fig = M.plt.figure()
                ax = fig.add_subplot(111)

                ax.plot(x_axis, results['validation_0']
                        [temp_metric], "k-", lw=3, label='Train Set')
                ax.plot(x_axis, results['validation_1']
                        [temp_metric], "r-", lw=3, label='Validation Set')
                ax.legend()
                M.plt.xlabel("Boosting rounds")
                M.plt.ylabel(temp_metric)

                ax, legenfObejct = self._fancy_plot(ax, col_No=2)
                M.plt.show()

    @logger
    @timer
    def get_metrics(self):
        """Get metrics.

        Get metrics on non-CV validate results.

        These are (regression):
        MSE  : Mean  Square Error
        RMSE : RootMean  Square Error
        MAE  : Mean Absolute Error
        R2   : coefficient of determination. This is a value between 
               0 and 1 for no-fit and perfect fit respectively. 

        Parameters
        ----------
        None

        Returns
        -------
        String = "Succcess"
            This is used for unittesting purpouses only

        To Do
        -----
        It would be better to return a dictionary
        """

        self.logger.info("|")
        self.logger.info(
            "|- Selected metrics [no CV]: " + str(self.params_dict["metrics"]))

        # Selecting the right model
        for model_name_temp in self.params_dict["frameworks"]:
            self.logger.info("|- Getting metrics for: " +
                             model_name_temp + " model")
            if model_name_temp.lower() == "xgboost":
                model = self.modelXGB

        preds_test = model.predict(self.Set["X_test"])
        preds_train = model.predict(self.Set["X_train"])

        for temp_metric in self.params_dict["metrics"]:
            if temp_metric.lower() == "r2":
                r2_test = M.r2_score(self.Set["y_test"], preds_test)
                r2_train = M.r2_score(self.Set["y_train"], preds_train)
                self.logger.info("|- [R2]_on_test" ": %.4f" % r2_test)
                self.logger.info("|- [R2]_on_train" ": %.4f" % r2_train)

            if temp_metric.lower() == "mse":
                mse_test = M.mean_squared_error(self.Set["y_test"], preds_test)
                mse_train = M.mean_squared_error(
                    self.Set["y_train"], preds_train)
                self.logger.info("|- [MSE]_on_test" ": %.4f" % mse_test)
                self.logger.info("|- [MSE]_on_train" ": %.4f" % mse_train)

            if temp_metric.lower() == "rmse":
                mse_test = M.mean_squared_error(self.Set["y_test"], preds_test)
                mse_train = M.mean_squared_error(
                    self.Set["y_train"], preds_train)
                rmse_test = M.np.sqrt(mse_test)
                rmse_train = M.np.sqrt(mse_train)
                self.logger.info("|- [RMSE]_on_test" ": %.4f" % rmse_test)
                self.logger.info("|- [RMSE]_on_train" ": %.4f" % rmse_train)

            if temp_metric.lower() == "mae":
                mae_test = M.mean_absolute_error(
                    self.Set["y_test"], preds_test)
                mae_train = M.mean_absolute_error(
                    self.Set["y_train"], preds_train)
                self.logger.info("|- [MAE]_on_test" ": %.4f" % mae_test)
                self.logger.info("|- [MAE]_on_train" ": %.4f" % mae_train)

            if temp_metric.lower() == "precision":
                prec_test = M.precision_score(self.Set["y_test"], preds_test)
                prec_train = M.precision_score(
                    self.Set["y_train"], preds_train)
                self.logger.info("|- [Precision]_on_test" ": %.4f" % prec_test)
                self.logger.info(
                    "|- [Precision]_on_train" ": %.4f" % prec_train)

            if temp_metric.lower() == "precision":
                rec_test = M.recall_score(self.Set["y_test"], preds_test)
                rec_train = M.recall_score(self.Set["y_train"], preds_train)
                self.logger.info("|- [Recall]_on_test" ": %.4f" % rec_test)
                self.logger.info("|- [Recall]_on_train" ": %.4f" % rec_train)

            if temp_metric.lower() == "f1":
                f1_test = M.f1_score(self.Set["y_test"], preds_test)
                f1_train = M.f1_score(self.Set["y_train"], preds_train)
                self.logger.info("|- [F1]_on_test" ": %.4f" % f1_test)
                self.logger.info("|- [F1]_on_train" ": %.4f" % f1_train)

    @logger
    @timer
    def get_CV_metrics(self):
        """Get the CV scores of the model.

        To see the available keys
        sklearn.metrics.SCORERS.keys()

        The loocv does not have R2 because the function returns aNaN. 
        This could be fixed but I have not looked into it.

        Parameters
        ----------
        model : model object not fitted!

        train_target : pandas dataframe
            target

        name : string
            name to be used in returns pandas dataframe

        n_splits : int, default=10
            No of splits

        state : int, default=42
            No for the random state pseudo number generator

        test_size : float
            Size of the test set between 0 and 1

        type_ : string, default="k-fold"
            Type of cross-validation used

        verbose : string, default=False
            If True print on screen the results, otherwise it does 
            not print anything on console.

        Returns
        -------
        table : pandas dataframe
            Table containing the mean and std for each metrics.

        split_strategy : iterator
            Iterator used in the call, which can be used to access each fold or
            training set splits.
        """

        self.logger.info("|")
        self.logger.info(
            "|- Selected metrics [no CV]: " + str(self.params_dict["metrics"]))
        self.logger.info("|- Type of CV selected: " +
                         str(self.CV_type) + "=" + str(self.CV_n_splits))

        # Selecting the right model
        for model_name_temp in self.params_dict["frameworks"]:
            if model_name_temp.lower() == "xgboost":
                model = self.modelXGB_not_fitted

            # Remember that in CV the splitting is done internally
            train_set = self.X
            train_target = self.y

            # Manipulate the scoring spelling
            scoring = self._user_to_sk_keywords(self.params_dict["metrics"])
            metrics_acronyms = self.params_dict["metrics"]

            # k-Fold
            if self.CV_type == "k-fold":
                split_strategy = M.KFold(
                    n_splits=self.CV_n_splits, random_state=self.random_state, shuffle=True)
                result = M.cross_validate(model, train_set, train_target,
                                          scoring=scoring, cv=split_strategy, n_jobs=-1, return_train_score=True)
            # Repeated train-test
            elif self.CV_type == "repeated_tt":
                split_strategy = M.ShuffleSplit(
                    n_splits=self.CV_n_splits, test_size=self.test_size, random_state=self.random_state)
                result = M.cross_validate(model, train_set, train_target,
                                          scoring=scoring, cv=split_strategy, n_jobs=-1, return_train_score=True)

            elif self.CV_type == "loocv":
                split_strategy = M.LeaveOneOut()
                result = M.cross_validate(model, train_set, train_target,
                                          scoring=scoring, cv=split_strategy, n_jobs=-1, return_train_score=True)

            mean, std = [], []
            for j, scoring_temp in enumerate(scoring):
                self.logger.info("|- ["+metrics_acronyms[j]+"]_on_test_set: %.4f" %
                                 abs(result["test_"+scoring_temp].mean()))
                self.logger.info("|- ["+metrics_acronyms[j]+"]_on_train_set: %.4f" %
                                 abs(result["train_"+scoring_temp].mean()))

            # Get pndas dataframe
            table = self._get_dframe(
                scoring, mean, std, metrics_acronyms, result, model_name_temp)

        return table

    @logger
    def _get_dframe(self, scoring, mean, std, metrics_acronyms, result, model_name_temp):
        """Get Pandas dataframe

        """

        # Create pandas dataframe
        table = M.pd.DataFrame()

        for i in scoring:
            mean.append(abs(result["test_" + i].mean()))
            std.append(abs(result["test_" + i].std()))

        table["metrics"] = metrics_acronyms
        table["mean_CV_"+self.CV_type+"_"+model_name_temp] = mean
        table["std_CV_"+self.CV_type+"_"+model_name_temp] = std

        return table

    @logger
    def _user_to_sk_keywords(self, user_metrics):
        """User to Scikit-Learn keywords.

        Fins the correct spelling for the user-defined
        metrics. For the complete list see:
        sorted(sklearn.metrics.SCORERS.keys())

        Parameters
        ----------
        user_metrics : list of string
            List of metrics as defined by the user

        Returns
        -------
        sk_metrics : list of string
            List of metrics understood by sk-learn
        """

        self.logger.debug("|")
        self.logger.debug("|- From user to sk-learn metrics")
        self.logger.debug("|- User metrics: " + str(user_metrics))

        sk_metrics = []
        for temp in user_metrics:

            # Regression metrics
            if temp.lower() == "mae":
                sk_metrics.append("neg_mean_absolute_error")
            elif temp.lower() == "mse":
                sk_metrics.append("neg_mean_squared_error")
            elif temp.lower() == "rmse":
                sk_metrics.append("neg_root_mean_squared_error")
            elif temp.lower() == "r2":
                sk_metrics.append("r2")

            # Classification metrics
            elif temp.lower() == "precision":
                sk_metrics.append("precision")
            elif temp.lower() == "recall":
                sk_metrics.append("recall")
            elif temp.lower() == "f1":
                sk_metrics.append("f1")
            elif temp.lower() == "accuracy":
                sk_metrics.append("accuracy")

        self.logger.debug("|- Sk-learn metrics: " + str(sk_metrics))
        return sk_metrics
