"""
The purpouses of this script is to show how to
call the KPT mdodule from a python script. It is
however suggested to use a jupyter notebook to use
the plotting routines.
"""

# Import modules
import MGB

from MGB.MetaGradientBoosting import MetaGradientBoosting
import MGB.Modules as M
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings("ignore")


def main():

    # Clean up the logger file id present
    if M.os.path.exists("./MGB_logfile.log"):
        M.os.remove("MGB_logfile.log")

    """
    Read-in the dataset
    It is assumed the dataset has already been cleaned!
    """

    """
    # Classification dataset
    dataset = M.loadtxt('../DATASETS/pima-indians-diabetes.csv', delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    """
    X, Y = load_boston(return_X_y=True)

    XGBoost_params = {
            "base_score": 0.5, 
            "booster" : 'gbtree', 
            "colsample_bylevel" : 1,
            "colsample_bynode" : 1, 
            "colsample_bytree" : 1, 
            "gamma" :0,
            "importance_type" : 'gain',             
            "learning_rate" : 0.300000012,             
            "max_depth" : 6,
            "min_child_weight" : 1,             
            "n_estimators" : 100, 
            "n_jobs" : 8,                         
            "subsample" : 1
    }

    CatBoost_params = {            
            "learning_rate" : 0.30,             
            "depth" : 5,
    }
    params_dict_regression = {
        "frameworks": ["Catboost"],
        "XGBoost_params": XGBoost_params,
        "CatBoost_params": CatBoost_params,
        "problem": "regression",
        "X": X,
        "y": Y,
        "test_size": 0.1,
        "split_method": 3,
        "eval_metric": ["error", "logloss", "mae", "rmse"],
        "verbose": False,
        "metrics": ["R2", "MSE", "MAE", "RMSE"],
        "early_stopping_rounds": 8,
        "CV_type": "k-fold",
        "CV_n_splits": 10,
        "state": 42,
        "verbose": False
    }

    params_dict_classification = {
        "frameworks": ["XGBoost"],
        "XGBoost_params": XGBoost_params,
        "CatBoost_params": CatBoost_params,
        "problem": "classification",
        "X": X,
        "y": Y,
        "test_size": 0.2,
        "split_method": 2,
        "eval_metric": ["error", "logloss"],
        "verbose": False,
        "metrics": ["precision", "recall", "f1", "accuracy"],
        "early_stopping_rounds": 15,
        "CV_type": "k-fold",
        "CV_n_splits": 10,
        "state": 42,
        "verbose": False
    }

    # createMGB object
    MGB = MetaGradientBoosting(params_dict_regression)
    # MGB = MetaGradientBoosting(params_dict_classification)    
    
    # Get some info on data
    MGB.data_summary()    

    # Get some info on chosen modelling options
    MGB.modelling_summary()    

    # Start the fitting
    MGB.fit()    
    stop

    # Get metrics on train-test set
    MGB.get_metrics()

    # Get CV metrics on train-test set
    pandas = MGB.get_CV_metrics()

    # Print on console exit message logo
    MGB._welcome_message()

    print("")
    print("validate parameters also for CV model!")
    print("validate parameters also for CV model!")
    print("validate parameters also for CV model!")
    print("validate parameters also for CV model!")
    print("validate parameters also for CV model!")
    print("")

if __name__ == '__main__':
    main()
