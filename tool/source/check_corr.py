#coding:utf-8
import numpy as np
import pandas as pd
import os
import sys
path = sys.argv[1]
path = '/Users/shimizutoshiki/kaggle/mmp/output/'
nrows = 100000 # Set in a sufficient number
ls = os.listdir(path)
dfs = [pd.read_csv(path+csv, nrows = nrows).iloc[:,-1] for csv in ls]
df =  pd.concat(dfs, axis=1)
df.columns = ls
print (df.corr(method='pearson',))
print (df.corr(method='spearman',))
