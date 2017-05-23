#matplotlib inline
################ Sacred place for Importing ################
from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')


#########################################
########   Utility Function     #########
#########################################


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

#####################################
######      Main Function      ######           
#####################################

def main():

### parsing and Data pre-processing

    # load the provided data
    train_features = pd.read_csv('./data/dengue_features_train.csv',index_col=[0,1,2])
    train_labels = pd.read_csv('./data/dengue_labels_train.csv',index_col=[0,1,2])
    # Seperate data for San Juan
    sj_train_features = train_features.loc['sj']
    sj_train_labels = train_labels.loc['sj']
    # Separate data for Iquitos
    iq_train_features = train_features.loc['iq']
    iq_train_labels = train_labels.loc['iq']
    # Parsing Checkpoint
    #print('San Juan')
    #print('features: ', sj_train_features.shape)
    #print('labels  : ', sj_train_labels.shape)
    #print('\nIquitos')
    #print('features: ', iq_train_features.shape)
    #print('labels  : ', iq_train_labels.shape)
    #print(sj_train_features.head())
    # Remove 'week_start_date' string.
    sj_train_features.drop('week_start_date', axis=1, inplace=True)
    iq_train_features.drop('week_start_date', axis=1, inplace=True)

    # Null CheckPoint
    #print(pd.isnull(sj_train_features).any())
    # Plot the Vegetation Index over Time 
    '''
    (sj_train_features
     .ndvi_ne
     .plot
     .line(lw=0.8))

    plt.title('Vegetation Index over Time')
    plt.xlabel('Time')
    plt.savefig('./plot/Vegetation_over_time.png', dpi = 600, format = 'png')
    '''
    #find NaN in data be unsatisfying and eliminate those ddata
    sj_train_features.fillna(method='ffill', inplace=True)
    iq_train_features.fillna(method='ffill', inplace=True)

    ### Distribution of Labels 
    #See the distribution of data (Distribution CheckPoint)
    '''
    print('San Juan')
    print('mean: ', sj_train_labels.mean()[0])
    print('var :', sj_train_labels.var()[0])

    print('\nIquitos')
    print('mean: ', iq_train_labels.mean()[0])
    print('var :', iq_train_labels.var()[0])
    '''
    # find which regression is suitable
    sj_train_labels.hist()
    iq_train_labels.hist()

    ### Correlation Mapping
    sj_train_features['total_cases'] = sj_train_labels.total_cases
    iq_train_features['total_cases'] = iq_train_labels.total_cases
    # compute the correlations
    sj_correlations = sj_train_features.corr()
    iq_correlations = iq_train_features.corr()
    # plot san juan
    sj_corr_heat = sns.heatmap(sj_correlations)
    plt.title('San Juan Variable Correlations')
    plt.savefig('./plot/SJ_var_corr.png', format = 'png')
    # plot iquitos
    iq_corr_heat = sns.heatmap(iq_correlations)
    plt.title('Iquitos Variable Correlations')
    plt.savefig('./plot/IQ_var_corr.png', format = 'png')

    # San Juan
    (sj_correlations
        .total_cases
        .drop('total_cases') # don't compare with myself
        .sort_values(ascending=False)
        .plot
        .barh())
    # Iquitos
    (iq_correlations
        .total_cases
        .drop('total_cases') # don't compare with myself
        .sort_values(ascending=False)
        .plot
        .barh())
    
    ### pre-processing data
    sj_train, iq_train = preprocess_data('./data/dengue_features_train.csv',
                                    labels_path="./data/dengue_labels_train.csv")
    #print(sj_train.describe())
    #print(iq_train.describe())
    sj_train_subtrain = sj_train.head(800)
    sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

    iq_train_subtrain = iq_train.head(400)
    iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)
    sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
    iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)

    figs, axes = plt.subplots(nrows=2, ncols=1)

    # plot sj
    sj_train['fitted'] = sj_best_model.fittedvalues
    sj_train.fitted.plot(ax=axes[0], label="Predictions")
    sj_train.total_cases.plot(ax=axes[0], label="Actual")

    # plot iq
    iq_train['fitted'] = iq_best_model.fittedvalues
    iq_train.fitted.plot(ax=axes[1], label="Predictions")
    iq_train.total_cases.plot(ax=axes[1], label="Actual")

    plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
    plt.legend()


    sj_test, iq_test = preprocess_data('./data/dengue_features_test.csv')

    sj_predictions = sj_best_model.predict(sj_test).astype(int)
    iq_predictions = iq_best_model.predict(iq_test).astype(int)

    submission = pd.read_csv("./data/submission_format.csv",
                         index_col=[0, 1, 2])

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv("./data/benchmark.csv")
if __name__ == '__main__':
    main()