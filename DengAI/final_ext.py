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
from sklearn.ensemble import ExtraTreesRegressor as ETR
import xgboost as xgb
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
    '''
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'reanalysis_min_air_temp_k']
    df = df[features]
    '''
    # fill missing values
    df.drop('week_start_date', axis=1, inplace=True)
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
	                    "station_avg_temp_c " 
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
	
	### pre-processing data
	sj_train, iq_train = preprocess_data(train_features_path, labels_path = train_labels_path)
	#print(sj_train.describe())
	#print(iq_train.describe())
	
	
	kf = KFold(n_splits=6)	
	
	
	sj_model_list = []
	sj_err_list = []
	loop = 1
	for train_index, val_index in kf.split(sj_train): #The index will be split into [train_index] and [val_index] 
		X_train,X_val = sj_train.ix[train_index], sj_train.ix[val_index]
		sj_etr = ETR(n_estimators = 800,  max_depth = 4,random_state = 0, verbose = 1)
		sj_etr.fit(X_train.drop('total_cases',axis = 1),X_train['total_cases'])
		predictions = sj_etr.predict(X_val.drop('total_cases',axis = 1))
		sj_err_list.append(eval_measures.meanabs(predictions, X_val.total_cases))
		sj_model_list.append(sj_etr)
		loop += 1	
	print(sj_err_list)
	argmax = sorted(range(len(sj_err_list)), key=lambda x: sj_err_list[x])[0]
	print(argmax)
	sj_best_model = sj_model_list[argmax]

	iq_model_list = []
	iq_err_list = []
	loop = 1
	for train_index, val_index in kf.split(iq_train):
		X_train,X_val = iq_train.ix[train_index], iq_train.ix[val_index]
		iq_etr = ETR(n_estimators = 400,  max_depth = 4,random_state = 0)
		iq_etr.fit(X_train.drop('total_cases',axis = 1),X_train['total_cases'])
		predictions = iq_etr.predict(X_val.drop('total_cases',axis = 1))
		iq_err_list.append(eval_measures.meanabs(predictions, X_val.total_cases))
		iq_model_list.append(iq_etr)
				
		loop += 1
	print(iq_err_list)
	argmax = sorted(range(len(iq_err_list)), key=lambda x: iq_err_list[x])[0]
	print(argmax)
	iq_best_model = iq_model_list[argmax]

	##Accessing testing data
	test_features_path = os.path.join(data_path,'dengue_features_test.csv')
	sj_test, iq_test = preprocess_data(test_features_path)
	
	
	#Calculate the k-fold validation error
	sj_score = []
	for train_index, val_index in kf.split(sj_train):
		X_train,X_val = sj_train.ix[train_index], sj_train.ix[val_index]
		train_predict = np.array(sj_best_model.predict(X_val.drop('total_cases',axis = 1))).astype(int)
		sj_score.append(eval_measures.meanabs(train_predict, X_val.total_cases))
	print("Mean of {} cross validation of sj_score is {} (+/- {})".format(kf.get_n_splits(sj_train)
																,np.mean(sj_score),np.std(sj_score)))
	
	iq_score = []
	for train_index, val_index in kf.split(iq_train):
		X_train,X_val = iq_train.ix[train_index], iq_train.ix[val_index]
		train_predict = np.array(iq_best_model.predict(X_val.drop('total_cases',axis = 1))).astype(int)
		iq_score.append(eval_measures.meanabs(train_predict, X_val.total_cases))
	print("Mean of {} cross validation of iq_score is {} (+/- {})".format(kf.get_n_splits(iq_train)
																,np.mean(iq_score),np.std(iq_score)))
	
	##Use the model sj_lr and iq_lr trained before to predict the testing data
	print("Predicting testing data...")
	sj_predictions = sj_best_model.predict(sj_test)
	iq_predictions = iq_best_model.predict(iq_test)
	sj_predictions = np.array(sj_predictions).astype(int)
	iq_predictions = np.array(iq_predictions).astype(int)
	
	print("Creating submit file...")
	##Use submission_format as template to write the answer
	sample_path = os.path.join(data_path,'submission_format.csv')
	submission = pd.read_csv(sample_path,index_col=[0, 1, 2])
	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv("./data/ext_new.csv")
	
	'''
	##plotting but not update haha
	figs, axes = plt.subplots(nrows=2, ncols=1)
	
	# plot sj
	sj_train['fitted'] = sj_neg_model.fittedvalues
	sj_train.fitted.plot(ax=axes[0], label="Predictions")
	SJ_predictions = sj_neg_model.predict(sj_train).astype(int)
	for i in range(SJ_predictions.shape[0]-1,3,-1):
			SJ_predictions.ix[i] = SJ_predictions.ix[i-4]
	SJ_predictions.plot(ax=axes[0], label="Predictions")
	sj_train.total_cases.plot(ax=axes[0], label="Actual")

	# plot iq
	#iq_train['fitted'] = iq_neg_model.fittedvalues
	#iq_train.fitted.plot(ax=axes[1], label="Predictions")
	IQ_predictions = iq_neg_model.predict(iq_train).astype(int)
	for i in range(IQ_predictions.shape[0]-1,0,-1):
			IQ_predictions.ix[i] = IQ_predictions.ix[i-1]
	IQ_predictions.plot(ax=axes[1], label="Predictions")
	iq_train.total_cases.plot(ax=axes[1], label="Actual")

	plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
	plt.legend()
	plt.show()
	'''

if __name__ == '__main__':
	main()
