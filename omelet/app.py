# -*- coding: utf-8 -*-

# Title          : omelet app v2
# Author         : Yoon Kim
# Revision_Date  : 04/23/2020
# Version        : '0.2.1'


import warnings
warnings.filterwarnings("ignore")

from omelet.core import *

from pandas_profiling import ProfileReport
from tabulate import tabulate

import shutil
import time



def run(f_file, sample, c_clf, p_file, d_dis=True):
    """
    omelet application run
    """
    #########################################
    ##  Run Clients
    #########################################

    client = MlflowClient()
    logger = colorlogger()
    
    prof = Config("profile/" + p_file + ".yml")

    h2o.no_progress()
    h2o.init(nthreads=-1, max_mem_size='15G')

    logger.info("[Success] H2O / mlflow Client Connected.")


    p_mlflow = prof.section('mlflow')
    p_path = prof.section('path')
    p_model = prof.section('model')

    try:
        experiment = mlflow.create_experiment(p_mlflow['exp_name'])
    except:
        experiment = client.get_experiment_by_name(p_mlflow['exp_name'])
        mlflow.set_experiment(p_mlflow['exp_name'])

    print('\n' * 2)
   
    time.sleep(1.5)

   
    #########################################
    ##  Data Load Step
    #########################################
    data = pd.read_csv(p_path['data'] + f_file)
    # data = pd.read_pickle('data/credit.pkl')

    # Sampling if
    if sample is not None:    
        if data.shape[0] > sample:
            data = data.sample(sample)

    logger.info("[Success] Data Load Successfully Done")

    print('\n')
    print(" Column Info")
    tab = pd.DataFrame(data.iloc[0,:])
    print(tabulate(tab, ['Column Name','Example'], tablefmt="fancy_grid", floatfmt=".8f"))

    print('\n')

    while True:
        try:
            target_name: str = input("Input Target Column Name. : ")
            break
        except TypeError:
            print("[Error] Key Input Error !")

    print('\n' * 2)

    time.sleep(3)

    if d_dis:
        profile = ProfileReport(data, 
                                minimal=True, 
                                title='Omelet Data Profiling', 
                                plot={'histogram': {'bins': 8}}, 
                                pool_size=True, 
                                progress_bar=False)

        logger.info("[Success] Data Profiling Successfully Done")


    # if target_nunique bigger than 2 shap_opt enabled, else shap_opt disabled
    target_nunique = data[target_name].nunique(dropna = True)         
    feature_name = [col for col in data.columns if col not in target_name]

    hf = h2o.H2OFrame(data)

    if c_clf:
        hf[target_name] = hf[target_name].asfactor()

    # Get Model Type
    m_type = model_type(hf[target_name])
    logger.info("[Success] Get Model Type Successfully Done")


    #########################################
    ##  Train / Valid / Test Split
    #########################################

    train, valid, test = hf.split_frame(ratios=[.7, .15])
    logger.info("[Success] Data Split Successfully Done")


    #########################################
    ##  Automated Machine Learning
    #########################################
    logger.info("[Success] Automated Machine Learning is Running ...")


    #TODO: GLM Param. to mlflow
    # if m_type is "regression":
    #     model = H2OAutoML(max_models=p_model['max_model'], 
    #                     max_runtime_secs=p_model['max_run_sec'], 
    #                     nfolds=p_model['nfold'],
    #                     exclude_algos=["DeepLearning", "StackedEnsemble"])
    # else:
    h2o.show_progress()
    model = H2OAutoML(max_models=p_model['max_model'], 
                max_runtime_secs=p_model['max_run_sec'], 
                nfolds=p_model['nfold'],
                include_algos=["DRF", "GBM", "XGBoost"])

    model.train(x=feature_name, y=target_name,
                training_frame=train, 
                validation_frame=valid,
                leaderboard_frame=valid)
    logger.info("[Success] Automated Machine Learning Successfully Done !")
    h2o.no_progress()
    print('\n' * 2)

    #########################################
    ##  Get Leader Board & Model IDs
    #########################################
    lb = model.leaderboard

    # Get Model Ids
    model_ids = list(lb['model_id'].as_data_frame().iloc[:, 0])
    model_number = h2o.H2OFrame(list(range(1, len(model_ids)+1)), column_names=['rank'])

    lb_df = model_number.cbind(lb).as_data_frame()
    lb_df.set_index('rank')

    print(tabulate(lb_df, lb_df.columns, tablefmt="fancy_grid", floatfmt=".4f"))

    print('\n')

    try:
        for mid in model_ids:
            if not os.path.exists("result"):
                os.makedirs("result")

            selected_model = h2o.get_model(mid)
            logger.info("[" + selected_model.model_id + "] Running  | Saving Result ...")

            variable_importance_plot(selected_model, log=logger)

            if d_dis:
                profile.to_file(output_file="result/data_profiling.html")

            if m_type is not "regression":
                confusion_matrix_plot(selected_model, valid, target_name)
        
            if m_type is "binomial":
                roc_curve_plot(selected_model, valid)
                shap_value, expected_value, shap_data = shap_values(selected_model, test, feature_name, sample = 1000)

                #TODO: It makes blank pdf file, but on jupyterlab it works
                #shap_force_plot(shap_value, expected_value, shap_data)

                shap_plots(shap_value, expected_value, shap_data)

            send_to_mlflow(p_mlflow['exp_id'], selected_model.model_id, selected_model, valid=valid, log=logger, model_type=m_type)

            shutil.rmtree("result")
            print('\n')
            
    except Exception as _e:
        logger.error(_e)
        raise
        

    print('\n' * 2)
    print('Everything Alright !\n')
    print('Please Visit: http://192.168.100.172:9081')
    print('\n' * 3)

    # # load the model
    # saved_model = h2o.load_model(model_path)
    # pred = saved_model.predict(test_data=test)
    # print(pred.head(10))
