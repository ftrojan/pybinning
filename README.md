# pybinning
Binning as feature engineering technique for better machine learning models

You want to do four different things around binning: autobinning, 
manual adjustments, calculate WoE and plot binning graphs.
The following example shows you how to use `pybinning` in your code.

```
import pybinning as bt # recommended alias is bt which come from original binning toolbox written in Matlab
```

## Autobinning

```
binning_object = bt.Binning(input_dataframe, target_name, predictor_names, idx_test)
binning_object.autobinning(bt.binning_eq_frequency, verbose=True, nbins=3)
```

## Manual adjustments to the binning

```
binning_object.adjust_binning("P075", code = "8", missbin = 1)
binning_object.adjust_binning("P094", code = "0.7", missbin = 1)
```

## Score new data and calculate WoE

```
woe_dataframe = binning_object.tbl_woe(another_input_dataframe, verbose = True)
```

## Plot binning graphs

```
varnames = ['P075','P094']
for i in range(len(varnames)):
    varname = varnames[i]
    binning_object.plot_binning(varname, output_directory)
    print("%d/%d\n" % (i+1, len(varnames)))
```
