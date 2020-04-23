# -*- coding: utf-8 -*-

# Title          : omelet app v2
# Author         : Yoon Kim
# Revision_Date  : 04/23/2020
# Version        : '0.2.1'

import warnings
warnings.filterwarnings("ignore")

import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import sys
import codecs

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

import os

import logging
from colorlog import ColoredFormatter

import shutil

from omelet.core import *



def run(f_file, sample, m_type, p_file, d_dis=True):
    """
    omelet application run
    """
    #########################################
    ##  Run Clients
    #########################################

    h2o.init(nthreads=-1, max_mem_size='15G')
    client = MlflowClient()
    logger = colorlogger()
    
    model_type = m_type
    prof = Config("profile/" + p_file)

    p_mlflow = prof.section('mlflow')
    p_path = prof.section('path')
    p_model = prof.section('model')

    try:
        experiment = mlflow.create_experiment(p_mlflow['exp_name'])
    except:
        experiment = client.get_experiment_by_name(p_mlflow['exp_name'])
        mlflow.set_experiment(p_mlflow['exp_name'])

    #########################################
    ##  Data Load Step
    #########################################
    data = pd.read_csv(p_path['data'] + d_file)
    # data = pd.read_pickle('data/credit.pkl')

    # Sampling if
    if sample is not -1:    
        if data.shape[0] > sample:
            data = data.sample(sample)

    print('[Success] Data Load Successfully Done !')
    print("Variable List : " + ", ".join(data.columns))

    while True:
        try:
            target_name: str = input("Input Target Column Name. : ")
            break
        except TypeError:
            print("[Error] Key Input Error !")


    if d_dis:
        profile = ProfileReport(data, 
                                minimal=True, 
                                title='Omelet Data Profiling', 
                                plot={'histogram': {'bins': 8}}, 
                                pool_size=True, 
                                progress_bar=True)


    # if target_nunique bigger than 2 shap_opt enabled, else shap_opt disabled
    target_nunique = data[target_name].nunique(dropna = True) 
        
    feature_name = [col for col in data.columns if col not in target_name]

    hf = h2o.H2OFrame(data)

    if model_type is not 'numeric':
        hf[target_name] = hf[target_name].asfactor()
    else:
        pass

    #########################################
    ##  Train / Valid / Test Split
    #########################################
    train, valid, test = hf.split_frame(ratios=[.7, .15])


    #########################################
    ##  Auto ML Run
    #########################################
    print('[Success] Auto ML !')
    model = H2OAutoML(max_models=p_model['max_model'], 
                    max_runtime_secs=p_model['max_run_sec'], 
                    nfolds=p_model['nfold'],
                    include_algos=["DRF", "GBM", "XGBoost"])

    model.train(x=feature_name, y=target_name,
                training_frame=train, 
                validation_frame=valid,
                leaderboard_frame=valid)
    print('[Success] Auto ML Successfully Done !')



    #########################################
    ##  Get Leader Board & Model IDs
    #########################################
    lb = model.leaderboard

    # Get Model Ids
    model_ids = list(lb['model_id'].as_data_frame().iloc[:, 0])
    model_number = h2o.H2OFrame(list(range(1, len(model_ids)+1)), column_names=['rank'])

    print(model_number.cbind(lb))

    try:
        for mid in model_ids:
            if not os.path.exists("result"):
                os.makedirs("result")

            selected_model = h2o.get_model(mid)
            confusion_matrix_plot(selected_model, valid, target_name)
            variable_importance_plot(selected_model, log=logger)
            profile.to_file(output_file="result/data_profiling.html")

            # For Binary Classification
            if model_type is "binary":
                roc_curve_plot(selected_model, valid)
                shap_value, expected_value, shap_data = shap_values(selected_model, test, feature_name, sample = 1000)

                #TODO: It makes blank pdf file, but on jupyterlab it works
                #shap_force_plot(shap_value, expected_value, shap_data)

                shap_plots(shap_value, expected_value, shap_data)
            else:
                pass

            send_to_mlflow(p_mlflow['exp_id'], selected_model.model_id, selected_model, valid=valid, log=logger, model_type=model_type)

            shutil.rmtree("result")
            
            
        print('\n')

    except Exception as _e:
        logger.error(_e)
        raise
        

    print("-"*40)
            

    print('Everything Alright !\n')
    print('Please Visit: http://192.168.100.172:9081')



    # # load the model
    # saved_model = h2o.load_model(model_path)
    # pred = saved_model.predict(test_data=test)
    # print(pred.head(10))
