#coding:utf-8
import numpy as np
import pandas as pd
DEBUG = True
if DEBUG:
    nrows = 10000
else:
    nrows = None
ss = pd.read_csv("../input/sample_submission.csv", nrows = nrows)
for i in range(10):
    r = np.random.rand(len(ss))
    ss[ss.columns[-1]] = r #  予測値は最後にあると仮定
    ss.to_csv('../submission/sample'+str(i)+'.csv', index=False)
