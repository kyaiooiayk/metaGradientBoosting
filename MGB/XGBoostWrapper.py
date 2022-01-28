"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access XGBoost imports
"""
from MGB.Modules import Modules
from MGB.Decorators import timer, logger
try:
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError:
    pass
M = Modules()


class XGBoostWrapper():
    """Modelling class.

    Contains all the actions performed while the model is built 
    and run. It also contains all the diagnostics for checking 
    the soundness of the model.
    """

    def __init__(self, logger, params_dict, Set):
        """Initialise the object

        """

        self.log_file_name = "MGB_logfile"
        self.log_file_dir = "./"
        self.logger = logger
        self.Set = Set
        self.model = None
        self.problem = params_dict["problem"].lower()
        self.eval_metric = params_dict["eval_metric"]
        self.verbose = params_dict["verbose"]
        self.early_stopping_rounds = params_dict["early_stopping_rounds"]
        self.verbose = int([1 if params_dict["verbose"] == True else 0][0])
        self.XGB_params = params_dict["XGBoost_params"]

        if self.problem.lower() == "classification":
            self.model_not_fitted = XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", verbosity=0)
        elif self.problem.lower() == "regression":
            self.model_not_fitted = XGBRegressor()

        self.logger.info("|")
        self.logger.info("|- Creating XGBoost object")

    @logger
    def _fit(self):

        if self.problem == "classification":
            model = self._fit_classifier()
        elif self.problem == "regression":
            model = self._fit_regressor()

        return model

    @logger
    @timer
    def _fit_classifier(self):
        """ Fit classifier
        """

        model = XGBClassifier(use_label_encoder=False,
                              eval_metric="logloss", verbosity=self.verbose)

        eval_set = [(self.Set["X_train"], self.Set["y_train"]),
                    (self.Set["X_val"], self.Set["y_val"])]
        model.fit(self.Set["X_train"], self.Set["y_train"], early_stopping_rounds=self.early_stopping_rounds,
                  eval_metric=self.eval_metric, eval_set=eval_set, verbose=self.verbose)

        self.model = model
        return model

    @logger
    @timer
    def _fit_regressor(self):
        """ Fit regressor
        """

        # Valisate parameters
        valid_params = self._validate_params()

        # Instantiate model with user-defined paramaters
        model = XGBRegressor(**valid_params)

        # Print on console used paramters
        self._get_default_params(model)

        eval_set = [(self.Set["X_train"], self.Set["y_train"]),
                    (self.Set["X_val"], self.Set["y_val"])]
        model.fit(self.Set["X_train"], self.Set["y_train"], early_stopping_rounds=self.early_stopping_rounds,
                  eval_metric=self.eval_metric, eval_set=eval_set, verbose=self.verbose)

        self.model = model

        # Get the default params after the model is fitted
        self._get_default_params(model)

        return model

    def _get_default_params(self, model):
        """Get default parameters.

        Parameters
        ----------
        model : object
            XGboost Model object

        Return
        ------
        default_params : dict
            Dictionary of the default parameters
        """

        for i in model.get_params():
            self.logger.debug("|- Parameter: " + str(i) +
                              "=" + str(model.get_params()[i]))

    def _validate_params(self):
        """Validate XGB parameters
        """

        self.logger.info("|")
        self.logger.info("|- Validating parameters")

        model_dummy = XGBRegressor()
        param_dict = model_dummy.get_params()
        validated_params = {}
        for user_param_temp in self.XGB_params:
            #print(user_param_temp, self.XGB_params[user_param_temp])
            if not user_param_temp in param_dict.keys():
                self.logger.warning(
                    "|--> Paramter: " + user_param_temp + " is invalid! Paramter will NOT be passed!")
            else:
                validated_params[user_param_temp] = self.XGB_params[user_param_temp]

        # Updating verbosity
        validated_params["verbosity"] = self.verbose
        return validated_params
