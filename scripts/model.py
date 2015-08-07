import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


class Model(object):

	def __init__(self, train_df, test_df):
		self.train_df = train_df
		self.test_df = test_df
		self.y = self.train_df['project_valuation']


	def get_X_y(self):
		self.X = self.train_df[self.train_df.columns[:-1]]
		
	def build_model(self):
		self.get_X_y()

		model = GradientBoostingRegressor()
		model.fit(self.X, self.y)

		return model

	def make_predictions(self):
		model = self.build_model()
		return model.predict(self.test_df)

	def create_submission(self, filename, preds):
		ids = self.test_df.index.values
		submission = pd.read_csv('./Sample_Solution.csv')
		submission['ID'] = ids
		submission['Project_Valuation'] = preds
		submission.to_csv('./submissions/' + filename, index=False)


