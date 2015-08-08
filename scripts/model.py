import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error, make_scorer

import matplotlib.pyplot as plt


class Model(object):

	def __init__(self, train_df, test_df):
		self.train_df = train_df
		self.test_df = test_df
		
		self.X = self.train_df[self.train_df.columns[:-1]]
		self.y = self.train_df['project_valuation']

	def remove_columns(self, cols):
		self.X = self.X[self.X.columns.drop(cols)]
			
	def build_model(self):

		self.remove_columns(['institute_latitude', 'institute_longitude'])
		
		model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1200)
		model.fit(self.X, self.y)

		return model

	def compute_importances(self, ensemble, normalize=False):
		trees_importances = [base_model.tree_.compute_feature_importances(normalize=normalize)
							 for base_model in ensemble.estimators_]

		return sum(trees_importances) / len(trees_importances)

	def plot_feature_importances(self, importances, normalize=False, color=None, alpha=0.5, label=None):
		if hasattr(importances, 'estimators_'):
			importances = self.compute_importances(importances, normalize=normalize)
		print importances
		plt.bar(range(len(importances)), importances, color=color, alpha=alpha, label=label)

	def feature_importance(self):
		extra_trees = ExtraTreesRegressor()
		self.plot_feature_importances(extra_trees.fit(self.X, self.y), label='extra')

	def eval_score(self, ytrue, ypred):
		return np.sqrt(mean_squared_error(ytrue, ypred))

	def rmse_scorer(self):
		return make_scorer(self.eval_score, greater_is_better=False)

	def split_dataset(self, test_size=0.3):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=44)

		self.Xt = X_train
		self.Xv = X_test
		self.yt = y_train
		self.yv = y_test

	def performance(self):

		self.remove_columns(['institute_state', 'institute_country', 'var10', 'var11', 
							'var12', 'var13', 'var14', 'var15', 
							'instructor_past_performance', 'instructor_association_industry_expert', 'secondary_area', 'var24'])
		self.split_dataset()

		model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1200)	
		model.fit(self.Xt, self.yt)

		yt_pred = model.predict(self.Xt)
		self.training_score = self.eval_score(self.yt, yt_pred)

		yv_pred = model.predict(self.Xv)
		self.test_score = self.eval_score(self.yv, yv_pred)

	def cross_validation(self):

		self.remove_columns(['institute_latitude', 'institute_longitude'])
		gbr = GradientBoostingRegressor()

		cv = ShuffleSplit(self.X.shape[0], n_iter=3, test_size=0.3, random_state=0)

		self.test_scores = cross_val_score(gbr, self.X, self.y, cv=cv, scoring=self.rmse_scorer(), n_jobs=1) # poor machine


	def make_predictions(self):
		model = self.build_model()

		self.test_df = self.test_df[self.test_df.columns.drop(['institute_latitude', 'institute_longitude'])]
		return model.predict(self.test_df)

	def sanitize_predictions(self, preds):
		return [x if x >= 0 else 0 for x in preds]

	def create_submission(self, filename, preds):
		ids = self.test_df.index.values
		submission = pd.read_csv('./Sample_Solution.csv')
		submission['ID'] = ids
		submission['Project_Valuation'] = preds
		submission.to_csv('./submissions/' + filename, index=False)


