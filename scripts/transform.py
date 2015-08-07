import pandas as pd

class Transform(object):

	def __init__(self, df):
		self.df = df


	def lowercase_columns(self):
		return [col.lower() for col in self.df.columns]

	def get_col_with_missing_values(self):
		return [col for col in self.df if self.df[col].isnull().any()]

	def fill_missing_values(self):
		missing_val_cols = self.get_col_with_missing_values()
		print missing_val_cols
		
		for col in missing_val_cols:
			if self.df[col].dtype == 'object':
				self.df[col] = self.df[col].fillna('-1')
			else:
				self.df[col] = self.df[col].fillna(-1)

		return self.df
