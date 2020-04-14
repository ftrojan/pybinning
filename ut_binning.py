# -*- coding: utf-8 -*-
"""
Example usage of pybinning
@author: ftrojan
"""
import pandas as pd
import pybinning as bt

# 1 load data
rdata = pd.read_csv("modeling_dataset.txt", sep="\t")

# 2 identify which columns are predictors
tgname = "T01"
hnames = [
    "H01",
    "H03",
    "H04",
    "H05",
    "H06",
    "H07",
    "H08",
    "H09",
    "T02",
    "H02"
]
pnames = list(set(rdata.columns) - {tgname} - set(hnames))
pnames.sort()

# 3 train/test SPLIT 80/20
# deterministic way which can be reproduced in other languP001s
n = len(rdata)
ntn = round(0.8*n)
itn = list(range(ntn))
its = list(set(range(n)) - set(itn))

# 4 AUTOBINNING
b0 = bt.Binning(rdata, tgname, pnames, its)
b0.autobinning(bt.binning_eq_frequency, verbose=True, nbins=3)
bb = b0.getbdf()  # binning dataframe
print("success rate = %.3f" % bb['success'][bb['flag_predictor'] == 1].mean())

# 5 MANUAL CORRECTIONS OF BINNING
b0.adjust_binning("P075", code = "8", missbin = 1)
b0.adjust_binning("P094", code = "0.7", missbin = 1)
b0.adjust_binning("P102", code = "0.71, 1.25", missbin = 2)
b0.adjust_binning("P185", code = "0, 0.36", missbin = 2)
b0.adjust_binning("P323", code = "2.3", missbin = 0)
b0.adjust_binning("P337", code = "-0.14, 0.13", missbin = 2)
b0.adjust_binning("P336", code = "-0.05", missbin = 0)
b0.adjust_binning("P070", code = "1.06", missbin = 1)
b0.adjust_binning("P074", code = "0.2", missbin = 2)
b0.adjust_binning("P152", code = "1.04", missbin = 0)
b0.adjust_binning("P202", code = "-0.01", missbin = 1)
b0.adjust_binning("P049", code = "35000, 45000", missbin = 0)
b0.adjust_binning("P182", code = "0.01", missbin = 0)
b0.adjust_binning("P339", code = "-0.14, 0.11", missbin = 0)
b0.adjust_binning("P189", code = "-0.06, 0.02", missbin = 0)
b0.adjust_binning("P124", code = "0.54, 1.09", missbin = 2)
b0.adjust_binning("P212", code = "4.07, 11.78", missbin = 2)
b0.adjust_binning("P215", code = "0, 1.22", missbin = 3)
b0.save('binning.yaml')

# 6 WOE TRANSFORMATION -> output dataset for modeling
wdata = b0.predict(rdata, verbose=True)
wdata.to_csv("woe_dataset.txt", sep="\t", index=False)

# 7 binning plots
varnames = [
    'P075',
    'P094',
    'P102',
    'P185',
    'P323',
    'P337',
    'P336',
    'P070',
    'P074',
    'P152',
    'P202',
    'P049',
    'P182',
    'P339',
    'P189',
    'P124',
    'P212',
    'P215'
]
for i in range(len(varnames)):
    varname = varnames[i]
    # b0.plot_binning(varname, '.')
    print("%d/%d" % (i+1, len(varnames)))
