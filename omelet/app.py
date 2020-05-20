# -*- coding: utf-8 -*-

# Title          : omelet app v2
# Author         : Yoon Kim
# Revision_Date  : 04/23/2020
# Version        : '0.2.1'


from omelet.core import *

from pandas_profiling import ProfileReport
from tabulate import tabulate

import shutil
import time

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)


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
        if data.shape[0] > int(sample):
            data = data.sample(int(sample))

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

        logger.info("[Success] Data Profiling is Running ...")

        logo_string = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAToAAABpCAIAAAAlTXjmAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAABSsSURBVHhe7Z0JlBRFmoCrIrOurru6KrNBxRXFg1lw5JgBGtkVBNcTGAVkPJBDRhwWBBUEHJdxRWVRUcTx2mEUlfFkPThU7qPlEEXOvu9CGHB05XBXBXr+iOyqysrqrsrKzD6y+d/7Hq+ojozIOL6IyMioTEvdfg5BEFOAuiKIaUBdEcQ0oK4IYhpQVwQxDagrgpgG1BVBTAPqiiCmAXVFENOAuiKIaUBdEcQ0oK4IYhpQVwQxDagrgpgG1BVBTAPqiiCmAXVFENOAuiKIaUBdEcQ0oK4IYhpQVwQxDagrgpgG1BVBTAPqiiCmAXVFENOAuiKIaUBdEcQ0oK4IYhpQVwQxDagrgpgG1BVBTAPqiiCmoeV0LeHqKkhdFaOaIX2GL0u5usKU8HrYx5KDyCsZUkJFKcEMAc5cSiKeVhlRhklPuexw/UBUEKEiicaQJw0foNCg6BRhDEdRYvqBqKAJKVKJY3hyqUDk6ss8G5pdV5AE8nOA/LSbq1ht2/qm4/3nXM/Pzln4kHvps671i51ln9hPfMnXRZnAxSmHa6OIO/IZv+8ju5yTe9nJKELqBJpCMVf6SVJCBzbY1PY+hdzpfVzlKpv8cP1UrbGdZpErk5MjJb06KelvPuObqlOLU8j9uIsrWpFI1BAObWqkzAu5k3uMTy6VilU2KM8MZZ49zagrnHoNdHvk83ccD0309u8V9AUiFovAEBn0s9sb6XVZaMpo39rFzlNwSK0Rw2ANmTPFQ5MggoUTLFaa4tOz3LRTUITUSQ2BHsflYjmChACLeMeNftrXqqm5Eu6HnXyf7qHE4fqxiFf2Df68h80vFMnJKeF+3s317RmsTxoKyiLOm+aBjlUZ0ljKSenH9txcaAkGZZmd+dQxXjoqpJZ5GTlcYAsEWcNTHGggFrHHpaH/+4rPUObZ01y6whS3kqx/zTl0UMDpYnVDYYUrKQRIn2P28jahf+/QshdddWVsQFZEmBVxXUHU+lTE9u3CX2+01VWkBNYMtPhC7oo+rMVLqbCERv0mO117d2O6xmPQiUUckK9W1/wesZNnPVqz6RoMsSYhP23NsDOfklbX+nFCcaCBWMTuXU2qKxshoxttdw732+00J/UdGy9D0lX+DcDK3WIVbxscqFln0zXM1pBHpzJd5alYxGnjvDRaNSKp4QBZ8kROfTuQpTL6pux0ze/BdI3HoBOLOPBytbpe/iumKxwFBWURn5jeTLqGw0xX+Wlrhp35vWPT6cp6h5QDDcQi9vylGXUFwaJk2cvOjh3CtD7kZSTZSMlTfoA/JQfr0D78zkIXnbtqu5ptWFfBH4jsX2anF8mK8BooI9/v5LtckmKafl2lMtFOXu/uIZPpmjh5beTdfYuPrvc0pOvfNtusNv1JpCfvok65P5hMV1Crmjw3252Tw0onURn0v4QTfnFx7rgR/tkTPfOme56c4Zk90Qstu/PFuaS+wmLh2X85XnhqpptOqjUY26CuLNpbBvtpnPovj6Nk/gx3g0lo1xWiImKvbiFofOOG+7Ux5kb/UzPcpzKuq7USXVnpXdc/MEFflt9+2tXw4nAxd2wHP2mUb+xNyqMMBE7gkSmen6CL1N+ukmkyXeFEq8j8mUwSqSZijSDHFRk73LdpiePoTp6qctAag97zgDFq818dY4b5Ews20rFM8gUPusG9rEuhMV2tAm8XNix20imx4pCsqOCiG2zt8tgMQh4/oFNXi7jwD+66I1Yag2ZKVeSulejKJlbb3nbUHdaaZWhR8G+aYQ0qIh6s6ShLSdcImkzXKIEezsouu2k1SDVhEfv3Dm2Byqi10iko9H8K8eC/8CX8qYZsftPRrydruPG5MYzJRHjlsRw6K1bT+uM0piuL84r84ElIV/O8Bc6khtxzh6+x+HXq+jTMKcAZOEPNqEu69ei6+Q0nrWJFLrIifZYVgZuCrNqnappG1xqy50N7SFruk1otdVWYeKvv2C52byZjZiBAlHy/i4NDaCTxxRuL6PVGtoPwWV1wptGVdShvPskujBVHqaSa7Hrf4fZCZlMiB3TrCnNsvYO/GlqTrhtfd0KVKUMiTaJrCXd8D9cvXvEAVUJ46PceevWf1SShnN7CmXmXl0YlM7ZPj+AJuDBIs3NFQaqu0CxkEXbtHDq6k65DKA/MCPSjFWTEdf6kzManAyxy1DUdrVxXmDPDmcDYwGZ8tIU0wzavxmkCXWvJ/BnMDckHVvH03mNl2q1hjQGHVBC4fKcRSrKxCJ96IJtNDgpdiXBuhzC9+SY7w+wilICaqyWrFzmtEIOkKBHsLuH883LrYwZQ1/S0Wl2hvgrp/qf1rzkLljg2L3HAh8rVNo33JgzCaF0ryIGNtnPOki26wNh1Se7ft/PaNyRUkGM7uR6Xxhoxi/Ps9uHq9Tba+SkCN4hCV4s4ZFBg2p1s0I59c077MJy52ggloIkXcv1+LZtHWMTbhvgn3sYm8LFvUNd0tFpdYd5Uwt10TcBKRJtT4B0CfJh+p7fJCyQtRutaSx6exMSQGqtV4G3C8hd1XBlK1JI1rzgdTogwocHsf/eobccpul7bP3hwo+2C83PrT5W1zvvGZrlrIkoWz82hMUhjqZVeVxevsE8dwzqC2HlmpeuJnfylneGsEveiH7/Xo7f01NByukKJgQ+yLItrX9W9Vq8fpusNAwPslOD0aIFMHd2WdC3lvt/B06mgVOWssQ7qF6Q3EnROIaDsqsjIG2SXiBax4z+F/3ebujvRKbr2zw9ChIvmMNliX3p9bNcEXGArDm+QMprZizslZfb+cd66g9bxN2sdXYu5H3fzM37nHTwgOHRQABgyIPjh8069ezDV0FK6lpKvN9huH+qHnNL8XklzvQ9qIatpTlPAdB16FdM1ViC0Q287ukbJ0mdddD4jjTZEIJwI3xgzONTQq0QCpVY/lFHeW6Au8oZ0PV3I/bif65k8x751MLvGhqpSxJBKlPzXtKQ4z2oXjm6ywfd3jtCqqwR0bXLUnIx+WkpXCUWW1ZdV09HGdYXslXN3/TappV54Qe6xL3gtK0yplHD/v5vr+cskuybd7qNLdhlrtyFdf9rF1R0iH/7JZSGxW7tWwWYTNr6m4sKpgkTX2fJEuERnB7K6nDfNXfc1XTzUqyuUZPO33ZbVtUWynJ42rivo9BV3Qcekq8FJo9TppJIDZPr4pMvCSzuHjn9BJ1TKkAoa1HU37V/ArsHS9UkszoF92a4JaDSKSOJAdmoI9BTyCDtfmHsUOqYKGqFeXVuEltW1FdLGda0ghcvtdKYqWyb54E+uzCOVeqrIypedscglhKLlKq5zGtMVrnuryY53HA5XpH6AZbXy9nxXuqWOKvLle/YcN5uQwyE0y+Jfn2R7raCO9esKk5EydotPJ1lNalpQVygZxZlrxsAt9W1e161vOfr9Oti3e7Bvz1Df7qEr84P0PpWBawZl5OBG28DLg327hWgSPYL5PUJ0h5MeXaGtVJPfjfTTXiYmWJfOuSd2NjKHL+JOl5Pf/FvSgAyxnYR4IDbdup7ex323jT+02Xa4QBcQw7dbefpAg5QkGqaldC3kTu7lvvmM/5sRWT76Oa+MXzNtXFcAWiSYE0daszF2FwhEKE8CUKNBGl3hr5WkbJVdCMMAm6gYulMXRsvUyGvIJ//t5G2JFS+bQ1gXv/GgU9diuKDgf3tD4MLzwhd3ytUDxDDsmgB9jI7KAaeldC0j1WtsvbuHLupoQJZnTfDSKyOoBUUqGmj7ugLQKOUo/moIGpJIrysQJQ9PTgpwzlnsWROgWTwSoIT7aT+06aTlrpHXyxaTderKtkl07yK/76qZvC6XZLPO11K6lpOSlXa3B5IzIMu3DvFD+aOuJiejrmXk79v5TsnrZJNH+ZSjd5T85THZvgiL4PVF9n0ku1VrhK6JXU16YA80MIWuSbua9GARxwxDXdXDGmsCaCgqRz/1QIQQrTwVNXWTUVcAVHxU9ugWi3j70ORfrpdx327nL+gY21/J6o/ui5AvSrESMFhX+Kxl5GnX6fzcozvMqSs9fy1ZHn5tQG2TyAhE0pZ1LaRN7fBm25GCeuqXOow29rttfDwJgF6eZUxCja4w0d3H9ZFUsYiRcLjk4+Q15yh55J6kSM6GCfOm5LU0Y3WFhIg47Br/C3NzXvijOztm0wfB0jyqbLutRFfWV943zqcty+sXO2l+DWlybVzXGrJ0ocvni/i8EV8g4vNEzj8vtylWhumPaTwsCUjLE1nyZE7mbYNqdAVqycqXXBytmLy597OduvGKryRVa21CJGlopZvvFZVnuK4W8aWHc+pOWGTP3FBNVvufW4mu7N7Yrg8cdcc1ZVnlBlI1tHFdK0nBEgfNWP1+aJq9DWp2CKmnkmyGJKD4ZElsedNBV3oUIRWo1LWYyjaoX7DjubnH5DdyoNFXk99LP5SPxdC1c+j4TjYzjx8ONIGu+IuclqGN61rKfbeVF4Sk8eeJBzxG7mqqJfOSt+mKYuTIZhsYogypQKWuQCUB/1e87Ezqp6vJjncdzpzYVgq2L+K9Zxraroy6Zgvqmg3G6QrZK+YUG/roz3HAJUN0hfjLyQ1XJsV/bf8A3Z+QZsOghHpdAfhS/sgLlu6QQcp8nZJ2t8aDxQOjrlmBumaDkUtNMNo8P1v28E6rCJeXSfc59FBF6zUYSNrM8MwsuHq0KkOmkpWugFytKFn2gos+S1VaMSaCwwHtydFwe0JdswV1zQbjdAUqSMWn9lAo6VES08Yb9KT8KPmj/IfvFsHtiRStUPfDyGx1jVPK/bCLVzzI4rahAXq1DNWpCAygrtmCumaDobpCi6wkI6+X/4hcECLh2rU2vQNsJfl6k+3s9kkdAUxQ6U9qGtRGgWZdD5CnZ7L5QuxmbCAQKYQ+orHFLSN0TbzSil0kN6euildaNZuuiVdaWagYqGtjGKorUE3WLHJytvpuUsrk2OF+qmvGK8zGgIKrIXffwhyACCFaQp8p8+mfVVeqNl3LScUq2cO+WV5mTmBPnGnMPSN0veyf5ZsQ8+be1wRvykuF6dq9K/QUiaSh0OgveBUhjaWcFK+0O+l7HhL7Itbpf1C7fs4IXYu50yXc1f8ayyQA/bRVpM/yPmhV1WQVwCFRsnSBi+6ql7oA5sB1AwL0JgqUqSJ8g2jQFdKtJv8xEY7Kix91bofwkc946p4icByduhZxP+/hX/pP9/3jvLMmeGdO8E4f7y1Y4kiXolEUcaf2cvKk4cOmN1T82kknxdw3W/g5UzwPjKfpzrjLO+tub+VqOxSjMmQzc0boClTTN7jChSWd2NQ3WcHni6xe7KBZzcpYCHzQuv1dB11hksUGnfG2t7J5Mri20bWKrPqzM/bCHsrChzINdDp1lYCZCIwtccqa5RXmEsqkm+WxulBoMEuSpwsTsWbLcmOcKbpCu6yVvf441mrDwciaV5x0cpXxilEC6ixK1r7m7CB/DCq7pnrs3ixv52rTFSqsgky4xSfmRoAr+gRPfJVpC64huiKtgTNFV6CU+7mYG3Y1ZDU2k2QNF8bYBX9wUw+lHlRxVBwoKQhQSl6ek+P30QPrY6BFljdkYOAnOFal8xKal5pgbr+PvrMMUBUedW0znEG6ApXk0Ba+12Vs6UJmLHBV3+Ani5zS4ElnX5Wxl3bBB2kyVsZtecsxgtrOZqHSsczV/B6hI9t4GjKryZJmXQE4TxhRgTT9SxxDdIXLRSgHY4GTV1NiTZF0xhk1lIxU9QYCEUJdKBLKijNLV6ihKvrS9F702YV5iVkxm83Cv//SK/To/Z51rzpLPrVH19uA4k/t6xc7H5/m7t8naHOwlf342hJ8sIiX9wwdKtB0T0iPrlmhU1f2JJRNbzj/51nXB88Zw/sLKQc22DJktpDOIyDppcYlDSxd4Nr7oT3dFUQRd/wLfuVLTjhJxbGagaggQvrjQTU9bGOcWbpKVJNoAX/9gORxEmD6MQSXJ3JW+zDgdLP1pNj3icDsmxHX+g9/zmtxFTCLruxGzq+kh7PSojCMt+a7MqzMsRs5vbqx+64ph+uA/co/zd0g9jQJv5/1zspjNSP6/BHaTcAYq0hOPWeirkAlObmfmzPV4/XSDNPSlNqxVAR0sJW+Z3+C/0pGSbA/5QYj9C3M0FNqLn1T6dpb2iYBRWEItAwF+jZxFbrmS9skFDHowSJOGZ1B19KPY+8WVRyrGYsYDLHdr6irFkrpy/Z2LHXceFXAZqfZZiSbGYeWS30YV44w+kb/3mV2OFyXWqbSNbEJ0RCg+VqFd55RpWtiE6JRWMSpYzLrmtiEaAjQxYdRVz1AY4V5bCn9weqk233ndQjbHTT/DeJwChDgnjt8295lv2WFA9W09TTUkgfv9los7ehVNKVdt66hptP1Zvq610RaN10doEs46nQ98SX/i4tyZYcbw+vzMj3wmena5RLoKYxNut34m/11BxtPuoIUrbAzxwxMtx1EuPt9fb8tYboOyIf+K1GVd41M2/U0Pc2lqwQUAfTxleT4F/zaV53PPOieMso3dpj/+gHBa64Iwig09Q4ffLl+sfN7uEwFUbO6uZqGMq5gieORyd5Hpngok72vP5Fzai87H0VIncDZFnPLX3TK0/roBZfaNQ+6q4l79fGcxOH6uYdSuDzteg/AdjUtnmto0sBk76pFznTvCi3mvt3KPzWDhlQeq5nJXojwSIHqh7Y2CFRlEQezksSJTfau/ouTblNXhGxGmldXCSgIKEdQEToq+LecnN7PndpHpaIjAHwJnSIEMERUCYgcxrdDVspB9m9tU/aRkCkpISkt+K+amygSkGs4t/jh+oGoAJVbo4xNGoDYoDbTJA1/gk4TKt3YLEOEKu9dpQcapHRi8G/GvDQ9LaGrHNaH0ZIF4IOBiiJIm6OldUUQRDWoK4KYBtQVQUwD6oogpgF1RRDTgLoiiGlAXRHENKCuCGIaUFcEMQ2oK4KYBtQVQUwD6oogpgF1RRDTgLoiiGlAXRHENKCuCGIaUFcEMQ2oK4KYBtQVQUwD6oogpgF1RRDTgLoiiGlAXRHENKCuCGIaUFcEMQ2oK4KYBtQVQUzCfu4f643QLDHwSUAAAAAASUVORK5CYII="

        profile = ProfileReport(data, 
                                minimal=False,
				explorative=True,
                                title='Data Profiling', 
                                plot={'histogram': {'bins': 8}}, 
				html={"style": {"logo": logo_string}},
                                pool_size=4, 
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
    print('\n')

    #########################################
    ##  Automated Machine Learning
    #########################################


    print(" Model Parameters")
    
    table = [
             ["target_name",        target_name],
             ["classifier",         c_clf],
             ["model_type",        m_type],
             ["max_models",         p_model['max_model']],
             ["max_runtime_secs",   p_model['max_run_sec']],
             ["nfolds",             p_model['nfold']]
            ]

    print(tabulate(table, ['Parameter','Value'], tablefmt="fancy_grid", floatfmt=".4f"))


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
    lb_df.set_index('rank', inplace = True)

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
    print('All Models Successfully Generated !\n')
    print('Please Visit: http://192.168.100.172:9081')
    print('\n' * 3)

    # # load the model
    # saved_model = h2o.load_model(model_path)
    # pred = saved_model.predict(test_data=test)
    # print(pred.head(10))
