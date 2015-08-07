import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Preprocessing(object):

	def __init__(self, train_df, test_df):
		self.train_df = train_df
		self.test_df = test_df

	def get_all_obj_cols(self, df):
		return [col for col in df.columns if df[col].dtype == 'object']

	def fit_transform(self):

		cols_with_obj_dtype = self.get_all_obj_cols(self.train_df)
		
		for col in cols_with_obj_dtype:
			train_feature = list(self.train_df[col])
			test_feature = list(self.test_df[col])

			feature = train_feature + test_feature

			lbl = LabelEncoder()
			lbl.fit(feature)

			self.train_df[col] = lbl.transform(train_feature)
			self.test_df[col] = lbl.transform(test_feature)

		return (self.train_df, self.test_df)

