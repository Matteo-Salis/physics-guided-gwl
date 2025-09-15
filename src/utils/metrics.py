import numpy as np
import pandas as pd


def compute_test_rmse_per_sensor(ds_true, ds_pred):
    residuals = ds_true - ds_pred
    residuals = residuals**2
    rmse =  np.sqrt(residuals.mean())
    
    return rmse

def compute_test_mape_per_sensor(ds_true, ds_pred):
    residuals = (ds_true - ds_pred)/ds_true
    residuals = np.abs(residuals)
    rmse =  residuals.mean()
    
    return rmse

def compute_test_nse_per_sensor(ds_true, ds_pred, true_mean):
    
    numerator = ds_true - ds_pred
    numerator = numerator**2
    numerator = numerator.sum()
    
    denominator = ds_true - true_mean
    denominator = denominator**2
    denominator = denominator.sum()
    
    nse = 1 - (numerator/denominator)
    
    return nse

def compute_test_kge_per_sensor(ds_true, ds_pred):
    
    rho = ds_true.corrwith(ds_pred, axis=0, method = "pearson")
    
    bias = ds_pred.mean()/ds_true.mean()
    
    variability = ds_pred.std()/ds_true.std()
    
    kge = 1-np.sqrt((rho-1)**2 + (bias-1)**2 + (variability-1)**2)
    
    return kge