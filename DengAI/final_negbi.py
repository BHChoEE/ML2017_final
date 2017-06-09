from __future__ import print_function
from __future__ import division
import os
import pandas as pd
import numpy as np
import random as rand
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#from warnings import filterwarnings
#filterwarnings('ignore')

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
weight_path = os.path.join(base_path,'weight')
rand.seed(1)
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
                 'station_min_temp_c',
                 'station_max_temp_c',
                 'reanalysis_min_air_temp_k',
                 'reanalysis_relative_humidity_percent'

                 ]
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


def get_best_model(train, test, city):

	if city == 'sj':
	    # Step 1: specify the form of the model
	    model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
	                    "station_avg_temp_c + " \
	                    "reanalysis_relative_humidity_percent "
	                    #"reanalysis_min_air_temp_k + " \
	                    
	elif city == 'iq':
		model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
	                    "station_avg_temp_c + " \
						"reanalysis_min_air_temp_k "
	grid = 10 ** np.arange(-10, -3, dtype=np.float64)
                    
	best_alpha = []
	best_score = 1000
	    
	# Step 2: Find the best hyper parameter, alpha
	for alpha in grid:
		model = smf.glm(formula=model_formula,data=train,family=sm.families.NegativeBinomial(alpha=alpha))
		results = model.fit()
		predictions = results.predict(test).astype(int)
	score = eval_measures.meanabs(predictions, test.total_cases)

	if score < best_score:
			best_alpha = alpha
			best_score = score
	#print('best alpha = ', best_alpha)
	#print('best score = ', best_score)
       
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
	train_features_path = os.path.join(data_path,'dengue_features_train.csv')
	train_labels_path = os.path.join(data_path,'dengue_labels_train.csv')
	train_features = pd.read_csv(train_features_path,index_col=[0,1,2])
	train_labels = pd.read_csv(train_labels_path,index_col=[0,1,2])
	# Seperate data for San Juan
	sj_train_features = train_features.loc['sj']
	sj_train_labels = train_labels.loc['sj']
	# Separate data for Iquitos
	iq_train_features = train_features.loc['iq']
	iq_train_labels = train_labels.loc['iq']

	# Remove 'week_start_date' string.
	sj_train_features.drop('week_start_date', axis=1, inplace=True)
	iq_train_features.drop('week_start_date', axis=1, inplace=True)

	#find NaN in data be unsatisfying and eliminate those ddata
	sj_train_features.fillna(method='ffill', inplace=True)
	iq_train_features.fillna(method='ffill', inplace=True)

	### pre-processing data
	sj_train, iq_train = preprocess_data(train_features_path, labels_path = train_labels_path)
	#print(sj_train.describe())
	#print(iq_train.describe())
	'''
	sj_train_subtrain = sj_train.head(800)
	sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)
	iq_train_subtrain = iq_train.head(400)
	iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)
	'''
	choose = rand.sample(range(0,sj_train.shape[0]-1),800)
	val = [i for i in range(sj_train.shape[0]) if i not in choose]
	sj_train_subtrain = sj_train.ix[choose]
	sj_train_subtest = sj_train.ix[val]    

	choose = rand.sample(range(0,iq_train.shape[0]-1),400)
	val = [i for i in range(iq_train.shape[0]) if i not in choose]
	iq_train_subtrain = iq_train.ix[choose]
	iq_train_subtest = iq_train.ix[val]    
	
	sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest,'sj')
	iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest,'iq')
	
	#Use K-fold to create cross validation data
	kf = KFold(n_splits=12)	
	sj_score = []
	for train_index, test_index in kf.split(sj_train):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train,X_test = sj_train.ix[train_index], sj_train.ix[test_index]
		predictions = sj_best_model.predict(X_test).astype(int)
		for i in range(predictions.shape[0]-1,3,-1):
			predictions.ix[i] = predictions.ix[i-4]
		sj_score.append(eval_measures.meanabs(predictions, X_test.total_cases))
		
	print("Mean of {} cross validation of sj_score is {} (+/- {})".format(kf.get_n_splits(sj_train)
																,np.mean(sj_score)
																,np.std(sj_score)))
	print(sj_score)
	iq_score = []
	for train_index, test_index in kf.split(iq_train):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train,X_test = iq_train.ix[train_index], iq_train.ix[test_index]
		predictions = iq_best_model.predict(X_test).astype(int)
		#print(predictions)
		for i in range(predictions.shape[0]-1,0,-1):
			predictions.ix[i] = predictions.ix[i-1]
		#print(predictions)
		iq_score.append(eval_measures.meanabs(predictions, X_test.total_cases))
	print(iq_score)
	print("Mean of {} cross validation of iq_score is {} (+/- {})".format(kf.get_n_splits(iq_train)
																,np.mean(iq_score)
																,np.std(iq_score)))
	
	
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
	plt.show()
	
	test_features_path = os.path.join(data_path,'dengue_features_test.csv')
	sj_test, iq_test = preprocess_data(test_features_path)
	sj_predictions = sj_best_model.predict(sj_test).astype(int)
	
	for i in range(sj_predictions.shape[0]-1,3,-1):
			sj_predictions.ix[i] = sj_predictions.ix[i-4]
	
	iq_predictions = iq_best_model.predict(iq_test).astype(int)
	for i in range(iq_predictions.shape[0]-1,0,-1):
			iq_predictions.ix[i] = iq_predictions.ix[i-1]
	
	sample_path = os.path.join(data_path,'submission_format.csv')
	submission = pd.read_csv(sample_path,index_col=[0, 1, 2])
	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv("./data/benchmark_shift.csv")

if __name__ == '__main__':
	main()
