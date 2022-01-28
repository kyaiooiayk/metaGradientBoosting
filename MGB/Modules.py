# XGBoost modules

# Keras modules

#LightGBM

# CatBoost

# Others
import time
import random
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
# Metrics for regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Metrics for classification
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
import functools
import logging
from numpy import loadtxt
import os
import pandas as pd

class XGBoostModules():
    def __init__(self):
        super(XGBoostModules, self).__init__()
        #self.test = test

class Others():
    def __init__(self):
        super(Others, self).__init__()
        self.plt = plt
        self.np = np
        self.seed = seed
        self.train_test_split = train_test_split
        self.r2_score = r2_score
        self.mean_squared_error = mean_squared_error
        self.mean_absolute_error = mean_absolute_error
        self.rcParams = rcParams
        self.sys = sys
        self.random = random
        self.time = time
        self.functools = functools
        self.logging = logging
        self.loadtxt = loadtxt
        self.os = os
        self.precision_score = precision_score
        self.recall_score = recall_score
        self.f1_score = f1_score
        self.KFold = KFold
        self.cross_validate = cross_validate
        self.pd = pd

"""
Alternatively, if you want to mantain 
the dot notation you can use the following.
If so, remove the super() from each closs.
"""
"""
class Modules_():
    def __init__(self):
        self.PT = PyTorchModules()
        self.K = KerasModules()
        self.O = Others()
"""


class Modules(XGBoostModules, Others):
    """Modules.

    Multi-inheritance of three base classes. Module is
    the children class which has access via self to all
    the base class instance attributes.

    This simplifies the import on each script. Instead
    of a having the same boiler plate element in all scripts,
    this is done once only and then called by a single line
    as:
    from KPT.Modules import Modules
    M = Modules() 
    """

    def __init__(self):
        super(Modules, self).__init__()
