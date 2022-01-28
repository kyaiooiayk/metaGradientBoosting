"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access Catoost imports
"""
from MGB.Modules import Modules
from MGB.Decorators import timer, logger
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ModuleNotFoundError:
    pass
M = Modules()


class CatBoostWrapper():
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
        # self.verbose = int([1 if params_dict["verbose"] == True else 0][0])
        self.verbose = params_dict["verbose"]
        self.CB_params = params_dict["CatBoost_params"]

        if self.problem.lower() == "classification":
            self.model_not_fitted = CatBoostClassifier(
                use_label_encoder=False, eval_metric="logloss", verbosity=0)
        elif self.problem.lower() == "regression":
            self.model_not_fitted = CatBoostRegressor()

        self.logger.info("|")
        self.logger.info("|- Creating Catoost object")

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

        model = CatBoostClassifier(use_label_encoder=False,
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
        valid_params, model_dummy = self._validate_params()

        # Instantiate model with user-defined paramaters
        model = CatBoostRegressor(**valid_params)

        # Print on console used paramters
        self._get_current_params(model)

        print("Find out to insert the evaluation metrics!")
        stop

        eval_set = [(self.Set["X_train"], self.Set["y_train"]),
                    (self.Set["X_val"], self.Set["y_val"])]
        model.fit(self.Set["X_train"], self.Set["y_train"], early_stopping_rounds=self.early_stopping_rounds,
                  eval_metric=self.eval_metric, eval_set=eval_set, verbose=self.verbose)

        self.model = model

        return model

    @logger
    def _get_current_params(self, model_dummy):
        """Get default parameters.

        Parameters
        ----------
        model_dummy : object
            Cat Model object fit with some dummy value
            just to get the parameters list

        Return
        ------
        default_params : dict
            Dictionary of the default parameters
        """
        
        model_dummy.fit(self.Set["X_train"][:2], self.Set["y_train"][:2])
        param_dict=model_dummy.get_all_params()

        for i in param_dict:
            self.logger.debug("|- Parameter: " + str(i) +
                              "=" + str(param_dict[i]))

    def _validate_params(self):
        """Validate Cat parameters.

        To get a list of all default parameter we need to fit the model!
        Since we'd like to do this as quickly as possible we pass just
        two values and we do it silently!
        """

        self.logger.info("|")
        self.logger.info("|- Validating parameters")

        model_dummy=CatBoostRegressor(verbose = False)
        model_dummy.fit(self.Set["X_train"][:2], self.Set["y_train"][:2])
        param_dict=model_dummy.get_all_params()
        # print(param_dict)

        validated_params={}
        for user_param_temp in self.CB_params:
            # print(user_param_temp, self.Cat_params[user_param_temp])
            if not user_param_temp in param_dict.keys():
                self.logger.warning(
                    "|--> Paramter: " + user_param_temp + " is invalid! Paramter will NOT be passed!")
            else:
                validated_params[user_param_temp] = self.CB_params[user_param_temp]

        # Updating verbosity
        validated_params["verbose"] = self.verbose

        return validated_params, model_dummy
