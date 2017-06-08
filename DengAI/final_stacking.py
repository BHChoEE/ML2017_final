from __future__ import print_function
from __future__ import division
import os
import pandas as pd
import numpy as np
import random as rand
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as LR
from datetime import datetime
rand.seed(datetime.now())
from warnings import filterwarnings
filterwarnings('ignore')

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
    #df.drop('week_start_date', axis=1, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq


def gradient_boosting(train_data,val_data):
	params = {'n_estimators': 800, 'max_depth': 5, 'min_samples_split': 3,
	'learning_rate': 0.01, 'loss': 'ls'}
	clf = ensemble.GradientBoostingRegressor(**params)
	train_label = train_data['total_cases']
	train_feat = train_data.drop('total_cases',axis = 1)
	clf.fit(train_feat, train_label)
	predictions = clf.predict(train_feat)
	mae = eval_measures.meanabs(predictions, train_label)
	#print("Training MAE: %.4f" % mae)

	val_label = val_data['total_cases']
	val_feat = val_data.drop('total_cases',axis = 1)
	val_predictions = clf.predict(val_feat)
	mae = eval_measures.meanabs(val_predictions, val_label)
	#print("Validation MAE: %.4f" % mae)
	
	return clf


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
	
	##Use K-fold to create cross validation data
	kf = KFold(n_splits=6)	
	
	##Do the stacking by adding 2 dataframes called 'negbi' and 'gb' which store the training prediction
	sj_train = sj_train.assign(negbi = 0)
	sj_train = sj_train.assign(gb = 0)
	loop = 1
	for train_index, test_index in kf.split(sj_train):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train,X_test = sj_train.ix[train_index], sj_train.ix[test_index]
		###neg_binomial method
		sj_best_model = get_best_model(X_train, X_test,'sj')
		predictions_neg = sj_best_model.predict(X_test).astype(int)
		#Shift the prediction manually
		for i in range(predictions_neg.shape[0]-1,3,-1):
			predictions_neg.ix[i] = predictions_neg.ix[i-4]
		
		###gradient boosting method
		clf_sj = gradient_boosting(X_train.drop(['negbi','gb'],axis = 1)
			,X_test.drop(['negbi','gb'],axis = 1))	
		predictions_gb = clf_sj.predict(X_test.drop(['total_cases','negbi','gb'],axis = 1)).astype(int)
		
		###Store the result in sj_train
		print("Adding the result of the predictions to sj training data({}/{})".format(loop,6))
		for idx,index in enumerate(test_index):
			sj_train['negbi'].ix[index] = predictions_neg.ix[idx]
			sj_train['gb'].ix[index] = predictions_gb[idx]	
		loop += 1	
		
	iq_train = iq_train.assign(negbi = 0)
	iq_train = iq_train.assign(gb = 0)
	loop = 1
	for train_index, test_index in kf.split(iq_train):
		X_train,X_test = iq_train.ix[train_index], iq_train.ix[test_index]
		
		###neg_binomial method
		iq_best_model = get_best_model(X_train, X_test,'iq')
		predictions_neg = iq_best_model.predict(X_test).astype(int)
		#Shift the prediction manually
		for i in range(predictions_neg.shape[0]-1,0,-1):
			predictions_neg.ix[i] = predictions_neg.ix[i-1]
		
		###gradient boosting method
		clf_iq = gradient_boosting(X_train.drop(['negbi','gb'],axis = 1),
			X_test.drop(['negbi','gb'],axis = 1))
		predictions_gb = clf_iq.predict(X_test.drop(['total_cases','negbi','gb'],axis = 1)).astype(int)
		###Store the result in iq_train

		print("Adding the result of the predictions to iq training data({}/{})".format(loop,6))
		for idx,index in enumerate(test_index):
			iq_train['negbi'].ix[index] = predictions_neg.ix[idx]
			iq_train['gb'].ix[index] = predictions_gb[idx]	
		loop += 1
	
	
	'''
	##plotting but not update haha
	figs, axes = plt.subplots(nrows=2, ncols=1)
	
	# plot sj
	sj_train['fitted'] = sj_best_model.fittedvalues
	sj_train.fitted.plot(ax=axes[0], label="Predictions")
	SJ_predictions = sj_best_model.predict(sj_train).astype(int)
	for i in range(SJ_predictions.shape[0]-1,3,-1):
			SJ_predictions.ix[i] = SJ_predictions.ix[i-4]
	SJ_predictions.plot(ax=axes[0], label="Predictions")
	sj_train.total_cases.plot(ax=axes[0], label="Actual")

	# plot iq
	#iq_train['fitted'] = iq_best_model.fittedvalues
	#iq_train.fitted.plot(ax=axes[1], label="Predictions")
	IQ_predictions = iq_best_model.predict(iq_train).astype(int)
	for i in range(IQ_predictions.shape[0]-1,0,-1):
			IQ_predictions.ix[i] = IQ_predictions.ix[i-1]
	IQ_predictions.plot(ax=axes[1], label="Predictions")
	iq_train.total_cases.plot(ax=axes[1], label="Actual")

	plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
	plt.legend()
	plt.show()
	'''
	test_features_path = os.path.join(data_path,'dengue_features_test.csv')
	sj_test, iq_test = preprocess_data(test_features_path)
	sj_test = sj_test.assign(negbi = 0)
	sj_test = sj_test.assign(gb = 0)
	sj_predictions_neg = sj_best_model.predict(sj_test).astype(int)
	for i in range(sj_predictions_neg.shape[0]-1,3,-1):
			sj_predictions_neg.ix[i] = sj_predictions_neg.ix[i-4]
	sj_predictions_gb = clf_iq.predict(sj_test.drop(['negbi','gb'],axis = 1)).astype(int)
	print("Adding predictions as features to sj testing data...")
	for i in range(len(sj_test['negbi'])):
			sj_test['negbi'].ix[i] = sj_predictions_neg.ix[i]
			sj_test['gb'].ix[i] = sj_predictions_gb[i]	
	
	iq_test = iq_test.assign(negbi = 0)
	iq_test = iq_test.assign(gb = 0)
	iq_predictions_neg = iq_best_model.predict(iq_test).astype(int)
	for i in range(iq_predictions_neg.shape[0]-1,0,-1):
			iq_predictions_neg.ix[i] = iq_predictions_neg.ix[i-1]
	iq_predictions_gb = clf_iq.predict(iq_test.drop(['negbi','gb'],axis = 1)).astype(int)
	print("Adding predictions as features to iq testing data...")
	for i in range(len(iq_test['negbi'])):
			iq_test['negbi'].ix[i] = iq_predictions_neg.ix[i]
			iq_test['gb'].ix[i] = iq_predictions_gb[i]	

	##use new information to run a linear regression
	print("Building linear regression model...")
	
	sj_train_lr = LR()
	sj_train_lr.fit(sj_train.drop('total_cases',axis = 1),sj_train['total_cases'])
	iq_train_lr = LR()
	iq_train_lr.fit(iq_train.drop('total_cases',axis = 1),iq_train['total_cases'])
	
	sj_score = []
	for train_index, test_index in kf.split(sj_train):
		X_train,X_test = sj_train.ix[train_index], sj_train.ix[test_index]
		train_predict = np.array(sj_train_lr.predict(X_test.drop('total_cases',axis = 1))).astype(int)
		sj_score.append(eval_measures.meanabs(train_predict, X_test.total_cases))
	
	print("Mean of {} cross validation of sj_score is {} (+/- {})".format(kf.get_n_splits(sj_train)
																,np.mean(sj_score),np.std(sj_score)))
	
	iq_score = []
	for train_index, test_index in kf.split(iq_train):
		X_train,X_test = iq_train.ix[train_index], iq_train.ix[test_index]
		train_predict = np.array(iq_train_lr.predict(X_test.drop('total_cases',axis = 1))).astype(int)
		iq_score.append(eval_measures.meanabs(train_predict, X_test.total_cases))
	print("Mean of {} cross validation of iq_score is {} (+/- {})".format(kf.get_n_splits(iq_train)
																,np.mean(iq_score),np.std(iq_score)))
	
	sj_predictions = sj_train_lr.predict(sj_test)
	iq_predictions = iq_train_lr.predict(iq_test)
	sj_predictions = np.array(sj_predictions).astype(int)
	iq_predictions = np.array(iq_predictions).astype(int)
	
	print("Creating submit file...")
	sample_path = os.path.join(data_path,'submission_format.csv')
	submission = pd.read_csv(sample_path,index_col=[0, 1, 2])
	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv("./data/stacking_negbi_gb_less.csv")
	

if __name__ == '__main__':
	main()
