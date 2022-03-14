import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.utils import version
from pycaret.classification import *
import os
logging.basicConfig(filename="101916086-log.txt", level=logging.INFO)
def preprocess():
    para=len(sys.argv)
    if para<2:
        # print(para)
        logging.error("Give parameter->csv input filename")
        print("Give parameter->csv input filename")
        logging.shutdown() 
    elif para>2:
        logging.error("Invalid number of arguments,needed parameters: 1 csv input filename")
        print("Invalid number of arguments,needed parameters: 1 csv input filename")
        logging.shutdown() 
    else:
        df = sys.argv[1]
        f = pd.read_csv(df)   
#         dfff.columns
#         f=dfff[['peptideSequence','length_sequence','instaIndex','ppeptide','boman','hmoment_alpha','vhse1','target']]
#         print(f)
        s = setup(data=f, target='target', silent=True) 
        cm = compare_models()
        results = pull()
        Accuracy=results["Accuracy"]
        # results.Model.tolist()
        # print(type(results))
#         df1=pd.DataFrame(results.iloc[:,:2])
#         df1.rename(columns = {'Accuracy':'Accuracy without Normalization'}, inplace = True)
#         final1 = df1.copy()
        setup(data=f, target='target', normalize = True, normalize_method = 'zscore', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_Zcore=results["Accuracy"]
        setup(data=f, target='target', normalize = True, normalize_method = 'minmax', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_minmax=results["Accuracy"]
        setup(data=f, target='target', normalize = True, normalize_method = 'maxabs', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_maxabs=results["Accuracy"]
        setup(data=f, target='target', normalize = True, normalize_method = 'robust', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_robust=results["Accuracy"]
#         print(final1)
#         print(final1.shape)
        final1=pd.concat([Accuracy,Accuracy_Zcore,Accuracy_minmax,Accuracy_maxabs,Accuracy_robust],axis=1)
        final1.columns=["Accuracy without normalization","Accuracy with zscore","Accuracy with minmax","Accuracy with maxabs","Accuracy with robust"]
        final1.to_csv("output-101916086-Normalization.csv",index=True)
#         final2 = df1.copy()
        setup(data=f, target='target', feature_selection = True, feature_selection_method = 'classic',feature_selection_threshold=0.2, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_classic_2=results['Accuracy']
        setup(data=f, target='target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.5, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_classic_5=results['Accuracy']
        setup(data=f, target='target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.2, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_boruta_2=results['Accuracy']
        setup(data=f, target='target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.5, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_boruta_5=results['Accuracy']
#         print(final2)
#         print(final2.shape)
        final2=pd.concat([Accuracy,Accuracy_classic_2,Accuracy_classic_5,Accuracy_boruta_2,Accuracy_boruta_5],axis=1)
        final2.columns=["Accuracy without feature_selection","Accuracy Classic=0.2","Accuracy Classic=0.5","Accuracy Boruta=0.2","Accuracy Boruta=0.5"]
        final2.to_csv("output-101916086-FeatureSelection.csv",index=True)
#         final3 = df1.copy()
        setup(data=f, target='target', remove_outliers = True, outliers_threshold = 0.02, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_Threshold_2=results['Accuracy']
        setup(data=f, target='target', remove_outliers = True, outliers_threshold = 0.04, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_Threshold_4=results['Accuracy']
        setup(data=f, target='target', remove_outliers = True, outliers_threshold = 0.06, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_Threshold_6=results['Accuracy']
        setup(data=f, target='target', remove_outliers = True, outliers_threshold = 0.08, silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_Threshold_8=results['Accuracy']
#         print(final3)
        final3=pd.concat([Accuracy,Accuracy_Threshold_2,Accuracy_Threshold_4,Accuracy_Threshold_6,Accuracy_Threshold_8],axis=1)
        final3.columns=["Accuracy without Outliers_removal","Accuracy Threshold=0.02","Accuracy Threshold=0.04","Accuracy Threshold=0.06","Accuracy Threshold=0.08"]
        final3.to_csv("output-101916086-OutlierRemoval.csv",index=True)
#         final4 = df1.copy()
#         final4 = df1.copy()
        setup(data=f, target='target',pca = True, pca_method = 'linear', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_linear=results['Accuracy']
        setup(data=f, target='target',pca = True, pca_method = 'kernel', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_kernel=results['Accuracy']
        setup(data=f, target='target',pca = True, pca_method = 'incremental', silent=True)
        cm = compare_models()
        results = pull()
        Accuracy_incremental=results['Accuracy']
#         print(final4)
        final4=pd.concat([Accuracy,Accuracy_linear,Accuracy_kernel,Accuracy_incremental],axis=1)
        final4.columns=["Accuracy without PCA","Accuracy METHOD=LINEAR","Accuracy METHOD=KERNEL","Accuracy METHOD=INCREMENTAL"]
        final4.to_csv("output-101916086-PCA.csv",index=True)
#         best_model =df1["Model"].iloc[0]
        setup(data=f, target='target', silent=True)
        cm= compare_models()

        plot_model(cm, plot='confusion_matrix',save=True)
        os.rename('Confusion Matrix.png','output-101916086-ConfusionMatrix.png')
        plot_model(cm, plot='auc',save=True)
        os.rename('AUC.png','output-101916086-AUC.png')

        plot_model(cm, plot='boundary',save=True)
        os.rename('Decision Boundary.png','output-101916086-Decision Boundary.png')

        plot_model(cm, plot='feature',save=True)
        os.rename('Feature Importance.png','output-101916086-FeatureImportance.png')

        plot_model(cm, plot='learning',save=True)
        os.rename('Learning Curve.png','output-101916086-LearningCurve.png')
        
if __name__=="__main__":
    try:
        preprocess()

    except OSError:
        logging.error("File not found")
        print("File not found")
    logging.shutdown()    



 
