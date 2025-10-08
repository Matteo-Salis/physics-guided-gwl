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
    mape =  residuals.mean()*100
    
    return mape

def compute_test_nbias_per_sensor(ds_true, ds_pred, true_iv = None):
    
    if true_iv is None:
        true_min = ds_true.min()
        true_max = ds_true.max()
        true_iv = true_max - true_min
        
    nbias = ds_pred - ds_true 
    nbias = nbias.mean()
    nbias = nbias/true_iv
    
    return nbias

def compute_test_nse_per_sensor(ds_true, ds_pred, true_mean = None):
    
    if true_mean is None:
        true_mean = ds_true.mean()
    
    numerator = ds_true - ds_pred
    numerator = numerator**2
    numerator = numerator.sum()
    
    denominator = ds_true - true_mean
    denominator = denominator**2
    denominator = denominator.sum()
    
    nse = 1 - (numerator/denominator)
    
    return nse

def compute_test_kge_per_sensor(ds_true, ds_pred, true_mean = None, true_sd = None):
    
    if true_mean is None:
        true_mean = ds_true.mean()
        
    if true_sd is None:
        true_sd = ds_true.std()
    
    rho = ds_true.corrwith(ds_pred, axis=0, method = "pearson")
    
    bias = ds_pred.mean()/true_mean
    
    variability = ds_pred.std()/true_sd
    
    kge = 1-np.sqrt((rho-1)**2 + (bias-1)**2 + (variability-1)**2)
    
    return kge