# -*- coding: utf-8 -*-

# Title          : omelet core v2
# Author         : Yoon Kim
# Revision_Date  : 04/23/2020
# Version        : '0.2.1'

import warnings
warnings.filterwarnings("ignore")

import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import pandas as pd
import numpy as np
import shap

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve, auc

import sys
import os
import codecs
import yaml

import logging
from colorlog import ColoredFormatter

shap.initjs()


#########################################
##  related to model
#########################################
def confusion_matrix_plot(model, data, target_name, result_path="result"):
    """
    Draw Confusion Matrix Plot
    """
    plt.tight_layout()
    skplt.metrics.plot_confusion_matrix(data[target_name].as_data_frame(),
                                        model.predict(data).as_data_frame()['predict'],
                                        normalize=False)
    plt.savefig(result_path + "/" + 'confusion_matrix.png', dpi=600)
    plt.close()

def variable_importance_plot(model, log, result_path="result"):
    """
    Draw Variable Importance Plot
    """
    algo = model.params['model_id']['actual']['name']

    if model.varimp() is not None:
        plt.tight_layout()
        var_imp = model._model_json['output']['variable_importances'].as_data_frame()

        x = var_imp['scaled_importance']
        x.index = var_imp['variable']

        plt.figure(figsize=(8, 8))
        plt.title("Variable Importance : " + algo)
        plt.xlabel("Relative Importance")

        x.sort_values().plot(kind='barh')
        plt.savefig(result_path + "/" + "variable_importance.png", dpi=600)
        plt.close()
    else:
        log.warning("[" + algo + "]" + "This model doesn't have variable importances")

def shap_values(model, data, feature_name, sample=1000):
    """
    Get SHAP Values (Default: 1000)
    """
    contributions = model.predict_contributions(data)
    contributions_matrix = contributions.as_data_frame().values

    # shap values are calculated for all features
    shap_value = contributions_matrix[:sample, 0:data.shape[1]-1]

    # expected values is the last returned column
    expected_value = contributions_matrix[:sample, 0:data.shape[1]-1].min()

    shap_data = data[feature_name].as_data_frame().iloc[:sample,:]

    return shap_value, expected_value, shap_data

def shap_force_plot(shap_value, expected_value, shap_data, force_num = 5, result_path="result"):
    """
    Draw SHAP Force Plot
    """
    shap.force_plot(expected_value, shap_value, shap_data, show=False)
    plt.savefig(result_path + "/" + "shap_force_plot_"+ "full" + ".pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

def shap_plots(shap_value, expected_value, shap_data, force_num = 5, result_path="result"):
    """
    Draw SHAP Plots
    """
    cli_no = range(1, force_num+1)

    # Draw Shap Force Plot
    for i in cli_no:
        shap.force_plot(expected_value, shap_value[i,:], shap_data.iloc[i,:], matplotlib=True, show=False)
        plt.savefig(result_path + "/" + "shap_force_plot_client"+ str(i) +".png", format='png', dpi=600, bbox_inches='tight')
        plt.close()

    # Draw Shap Summary Plot
    shap.summary_plot(shap_value, shap_data, show=False)
    plt.savefig(result_path + "/" + "shap_summary_plot.png", format='png', dpi=600, bbox_inches='tight')
    plt.close()

    # Draw Shap Bar Plot
    shap.summary_plot(shap_value, shap_data, plot_type="bar", show=False)
    plt.savefig(result_path + "/" + "shap_summary_bar_plot.png", format='png', dpi=600, bbox_inches='tight')
    plt.close()

def roc_curve_plot(model, valid, result_path="result"):
    """
    Draw ROC Curve Plot
    """
    perf = model.model_performance(valid)

    auc = perf.auc()
    fpr = perf.fprs
    tpr = perf.tprs
    algo = model.params['model_id']['actual']['name'].split('_')[0]
    lw = 2

    plt.style.use('classic')
    plt.figure()
    plt.tight_layout()


    plt.plot(fpr, tpr,
            color = 'darkorange',
            lw = lw,
            label = 'ROC Curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')

    plt.title('ROC Curve : %s (AUC = %0.2f)' % (algo, auc))
    plt.savefig(result_path + "/" + 'roc_curve_plot.png', dpi=600)

    plt.close()

#########################################
##  related to util
#########################################
class Config:
    """
    설정 값을 불러오는 클래스
    """

    def __init__(self, yaml_path, encoding='utf-8'):
        """
        설정 정보 초기설정

        Parameters
        ----------
        yaml_path : str
            yaml 설정파일 경로
        encoding : str
            file encoding (default: utf-8)

        Return
        ------
        None
        """
        yaml_path: str
        encoding: str

        self.yaml_path = yaml_path
        self.encoding = encoding

        with open(yaml_path, 'r', encoding=encoding) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.cfg = cfg

    def info(self):
        """
        설정 값을 보여주는 함수

        Parameters
        ----------
        None

        Return
        ------
        cfg : dict
            전체 설정 정보
        """
        cfg = self.cfg

        return cfg

    def section(self, section):
        """
        설정 값을 Section별로 Return하는 함수

        Parameters
        ----------
        section

        Return
        ------
        cfg_section : dict
            section별 설정 정보
        """
        section: str

        cfg_section = self.cfg[section]

        return cfg_section

class DuplicateFilter:
    """
    Filters away duplicate log messages.
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeFilter(self)

def colorlogger(logPath="log", fileName="syslog", console=False):
    """
    Set Logger
    """
    LOG_FORMAT = "%(log_color)s%(asctime)-8s%(reset)s | %(log_color)s%(levelname)-6s%(reset)s | %(log_color)s%(message)s%(reset)s"

    formatter = ColoredFormatter(LOG_FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.handlers.RotatingFileHandler("{}/{}.log".format(logPath, fileName), maxBytes=1024*1024*100, backupCount=50)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if console == True:
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
    else:
        pass

    return logger

def send_to_mlflow(exp_id, run_name, model, valid, log, model_type, result_path="result"):
    model_params = model.params
    algo = model_params['model_id']['actual']['name'].split('_')[0]

    try:
        with DuplicateFilter(log):
            # mlflow start run
            run = mlflow.start_run(experiment_id=exp_id, run_name = run_name, nested=True)

            # mlflow Model Parameter
            log.info("[" + run_name + "] Running  | Model Parameter Save")
            mlflow.log_param("algorithm", algo)
            mlflow.log_param("response_column", model_params['response_column']['actual']['column_name'])
            mlflow.log_param("nfolds", model_params['nfolds']['actual'])

            mlflow.log_param("ntrees", model_params['ntrees']['actual'])
            mlflow.log_param("max_depth", model_params['max_depth']['actual'])
            mlflow.log_param("min_rows", model_params['min_rows']['actual'])
            mlflow.log_param("stopping_metric", model_params['stopping_metric']['actual'])
            mlflow.log_param("stopping_tolerance", model_params['stopping_tolerance']['actual'])

            # mlflow Metrics Save
            log.info("[" + run_name + "] Running  | Model Metrics Save")

            perf = model.model_performance(valid)

            mlflow.log_metric("r2", perf.mse())
            mlflow.log_metric("logloss", perf.logloss())
            mlflow.log_metric("rmse", perf.rmse())
            mlflow.log_metric("mse", perf.mse())

            if model.auc() is not None:
                mlflow.log_metric("auc", perf.auc())
                
            if model.mean_per_class_error() is not None:
                mlflow.log_metric("mean_per_class_error", perf.mean_per_class_error())

            # mlflow Model Save (Remote)
            log.info("[" + run_name + "] Running  | Model Save (Remote)")
            mlflow.h2o.log_model(model, 'model')

            # mlflow Model Save (Local)
            log.info("[" + run_name + "] Running  | Model Save (Local)")
            mlflow.h2o.save_model(model, './model/' + run_name)

            # mlflow Save Artifacts (HTML / PNG File)
            log.info("[" + run_name + "] Running  | Artifacts Save (Remote)")
            mlflow.log_artifact(result_path)

            # mlflow end run
            mlflow.end_run()
            log.info("[" + run_name + "] Finished | All Works Successfully Done !")

    except Exception as _e:
        log.error(_e)
        raise
