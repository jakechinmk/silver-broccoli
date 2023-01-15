"""
Created by Jake on 2023-01-08 (yyyy-mm-dd)
Objective: To conduct eda using only matplotlib and seaborn (dataprep is not included in this module yet)
Design: Basic OOP style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, Dict

class EDA:
    def __init__(self, df:pd.DataFrame, valid_df:pd.DataFrame, target:str):
        if valid_df is not None:
            assert all(df.columns == valid_df.columns), 'The columns is mismatch'
        self.df = df
        self.vdf = valid_df
        self.numeric_list = df.select_dtypes(include=np.number).columns.tolist()
        self.target = target

    def check_missing_values(self, valid=False):
        if valid:
            temp_series = self.vdf.isna().sum() / self.vdf.shape[0]
        else:
            temp_series = self.df.isna().sum() / self.df.shape[0]

        no_na_list = temp_series[temp_series == 0].index.tolist()
        print(f'There are columns with no missing value: {no_na_list}')

        na_series = temp_series[temp_series > 0].sort_values() * 100        
        plt.bar(x=na_series.index, heigh=na_series)
        plt.xticks(rotation=90)
        plt.title('Missing Percentage by Columns')
        plt.show()
        return na_series.index.tolist(), no_na_list
    
    def check_cardinality_values(self, valid=False, threshold=20):
        if valid:
            temp_series = self.vdf.nunique()
        else:
            temp_series = self.df.nunique()
        
        one_category_list = temp_series[temp_series == 1]
        if one_category_list != []:
            print(f'There are columns with one category: {one_category_list}')

        mask1 = temp_series > 1
        mask2 = temp_series < threshold
        mask = mask1 & mask2
        cardinality_list = temp_series[mask].index.tolist()
        print(f'Columns with at least 2 categories to {threshold}: {cardinality_list}')
        return one_category_list, cardinality_list

    def plot_histogram_by_category(self, category:str, col:str, figsize=(20, 8)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        sns.histplot(data=self.df, x=col, ax=ax1)
        ax1.set_title(f'{col} Distribution for Full Data Set')
        sns.histplot(data=self.df, x=col, hue=category, ax=ax2)
        ax2.set_title(f'{col} by {category} Distribution')
        plt.show()