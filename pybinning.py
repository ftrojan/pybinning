# -*- coding: utf-8 -*-
"""
Created at 2018-09-04
Binning toolbox
@author: com-ftrojan
"""
import logging
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import yaml

# catch all RuntimeWarning events and be able to debug them
np.seterr(all='raise')


class Binning(object):

    def __init__(self, df, target_name, predictor_names, idx_test):
        """Creates binning object with DataFrame."""
        self.df = df
        self.target_name = target_name
        self.predictor_names = predictor_names
        n = len(df)
        self.bts = np.zeros(n, dtype=bool)
        self.bts[idx_test] = True
        self.bb = []

    def binning_manual(self, predictor_name, code, missbin):
        bx = binning_manual(self.df[predictor_name], self.df[self.target_name], self.bts, code, missbin)
        return bx

    def pred_binning(self, predictor_name, bfun, **kwargs):
        bx = pred_binning(self.df[predictor_name], self.df[self.target_name], self.bts, bfun, **kwargs)
        return bx

    def autobinning(self, binning_function, verbose=True, **kwargs):
        """Performs automatic binning on given DataFrame using specified binning_function."""
        y = self.df[self.target_name]
        self.bb = []
        num_predictors = len(self.predictor_names)
        if verbose:
            yn = y[~self.bts]
            ys = y[self.bts]
            logging.debug(f"{y.sum()} positive observations in total, positive rate {100*y.mean():.1f}%.")
            logging.debug(f"{yn.sum()} positive observations in train, positive rate {100*yn.mean():.1f}%.")
            logging.debug(f"{ys.sum()} positive observations in test, positive rate {100*ys.mean():.1f}%.")
            iter_predictors = tqdm.tqdm(range(num_predictors), total=num_predictors, desc="Autobinning")
        else:
            iter_predictors = range(num_predictors)
        for i in iter_predictors:
            xname = self.predictor_names[i]
            x = self.df[xname]
            bx0 = {'ordnum': i, 'varname': xname, 'success': False}
            try:
                bx1 = pred_binning(x, y, self.bts, binning_function, **kwargs)
                bx = bx0.copy()
                bx['success'] = True
                bx.update(bx1)
            except Exception as e:
                logging.error(f"{xname}: Exception {e}")
                bx1 = predictor_profile(x)
                bx = bx0.copy()
                bx.update(bx1)
            self.bb.append(bx)
        bdf = self.getbdf()
        logging.debug(f"{num_predictors} features processed")
        bfp = bdf['flag_predictor']
        bs = bdf['success']
        logging.debug(f"{bfp.sum()} features flagged as predictor ({100*bfp.mean():.0f}% success rate).")
        logging.debug(f"{bs.sum()} features successfully binned ({100*bs.mean():.0f}% success rate).")

    def getbdf(self):
        """Returns binning results formatted as DataFrame one row per predictor."""
        cols = [
            'ordnum',
            'varname',
            'success',
            'cl',
            'flag_predictor',
            'pmiss',
            'nd',
            'nbins',
            'gini_test',
            'lift_test',
            'gini_train',
            'lift_train',
            'missbin',
            'code'
        ]
        bl = [dict2tuple(bx, cols) for bx in self.bb]
        bdf = pd.DataFrame(bl, columns=cols).sort_values('gini_train', ascending=False)
        return bdf

    def getidx(self, predictor_name):
        """Returns index of a given predictor within the binning object."""
        bbnames = [x['varname'] for x in self.bb]
        idx = bbnames.index(predictor_name)
        return idx

    def getb(self, predictor_name):
        """Returns output binning dict for the specified predictor."""
        idx = self.getidx(predictor_name)
        return self.bb[idx]

    def adjust_binning(self, predictor_name, code, missbin):
        """Changes the binning for one predictor using specified binning code."""
        idx = self.getidx(predictor_name)
        bx = binning_manual(self.df[predictor_name], self.df[self.target_name], self.bts, code, missbin)
        self.bb[idx].update(bx)
        return

    def predict(self, newdata, verbose=False):
        """Applies binning to nw data and returns WoE DataFrame.
        :param newdata: Input DataFrame.
        :param verbose: True brings tqdm progress bar.
        :return: DataFrame with WoE values instead of original values in respective columns.
        The column names are not changed.
        """
        df = newdata.copy()
        if verbose:
            bbl = tqdm.tqdm(self.bb, total=len(self.bb), desc="WoE scoring")
        else:
            bbl = self.bb
        for bx in bbl:
            xname = bx['varname']
            woe = pred_woe(newdata[xname], bx)
            df[xname] = woe  # nahrazeni puvodnich hodnot WoE hodnotami
        return df

    def plot_binning(self, predictor_name, output_path):
        """Draws binning plot for a specified predictor and saves the PNG file to the output_path."""
        bx = self.getb(predictor_name)
        plotdata = bx['summary_train']
        tr = plotdata['m'].sum() / plotdata['n'].sum()
        ftn = plot_binning_aux(
            plotdata,
            tit="Predictor: %s, Gini_train = %.2f" % (predictor_name, bx['gini_train']),
            xlab=predictor_name,
            tr=tr
        )
        ftn.savefig("%s/%s_tn.png" % (output_path, predictor_name))
        plotdata = bx['summary_test']
        tr = plotdata['m'].sum() / plotdata['n'].sum()
        fts = plot_binning_aux(
            plotdata,
            tit="Predictor: %s, Gini_train = %.2f" % (predictor_name, bx['gini_train']),
            xlab=predictor_name,
            tr=tr
        )
        fts.savefig("%s/%s_ts.png" % (output_path, predictor_name))
        return

    def plot_binning_train(self, pred_name):
        bx = self.getb(pred_name)
        plotdata = bx['summary_train']
        plotdata = plotdata.loc[plotdata['n'] > 0, :]
        tr = plotdata['m'].sum() / plotdata['n'].sum()
        ftn = plot_binning_aux(
            plotdata,
            tit="Predictor: %s, Gini_train = %.2f" % (pred_name, bx['gini_train']),
            xlab=pred_name,
            tr=tr
        )
        return ftn

    def save(self, filename: str):
        def subset(x, keys):
            y = {k: x[k] for k in keys}
            y['class_name'] = str(x['cl'])
            return y
        logging.debug(f"saving into {filename}")
        save_keys = ['code', 'missbin']
        tosave = {b['varname']: subset(b, save_keys) for b in self.bb if b['flag_predictor'] and b['success']}
        with open(filename, 'w') as out:
            yaml.dump(tosave, out, indent=2, default_flow_style=False, sort_keys=False)
        logging.debug(f"saved into {filename}")

    def load(self, filename: str):
        """Override code and missbin of the current binning with the values saved in yaml file."""
        with open(filename, 'r') as inp:
            bbload = yaml.load(inp)
        varname = [b['varname'] for b in self.bb]
        for key, value in bbload.items():
            index = varname.index(key)
            if index:
                binrec = self.bb[index]
                binrec['code'] = value['code']
                binrec['missbin'] = value['missbin']
            else:
                logging.warning(f"{key}: feature not found in the current binning")


def dict2tuple(bx, cols, missval=np.nan):
    return tuple([bx.get(col, missval) for col in cols])


def preprocess(x: pd.Series):
    classx = x.dtype
    i0 = x.isnull()
    nmiss = i0.sum()
    xv = x.values[~i0]
    return xv, classx, nmiss


def predictor_profile(x):
    n = len(x)
    (xv, classx, nmiss) = preprocess(x)
    nd = len(x.unique())  # unique contains None
    if classx == np.int64 or classx == np.float64 or classx == np.bool:
        qx = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
        qq = x.dropna().quantile(qx)
        pprofile = {
            "flag_predictor": 1 if nd > 1 else 0,
            "cl": classx,
            "n": n,
            "nmiss": nmiss,
            "pmiss": float(nmiss) / n,
            "nd": nd,
            "quantiles": qq
        }
    elif classx == np.object and isinstance(list(xv)[0], str):
        ut = x.value_counts(dropna=False)
        qq = pd.Series(ut.index, index=list(np.cumsum(ut.values) / n))
        pprofile = {
            "flag_predictor": 1 if nd > 1 else 0,
            "cl": classx,
            "n": n,
            "nmiss": nmiss,
            "pmiss": float(nmiss) / n,
            "nd": nd,
            "quantiles": qq
        }
    else:
        pprofile = {
            "flag_predictor": 0,
            "cl": classx,
            "n": n,
            "nmiss": nmiss,
            "pmiss": float(nmiss) / n,
            "nd": nd,
            "quantiles": None
        }
    return pprofile


def grid_decode(code, classx):
    # dekoduje code a vrati borders
    if classx == np.int64 or classx == np.float64 or classx == np.bool:
        # borders bude list of float
        tokens = code.split(", ")
        if len(tokens) == 1 and tokens == "NA":
            borders = np.nan
        else:
            borders = [float(ti) for ti in tokens]
    elif classx == np.object:
        # borders bude DataFrame [level, id_bin]
        l1tokens = code.split(", ")
        if len(l1tokens) > 0:
            bid = 0
            bor = []
            for li in l1tokens:
                l2tokens = li.split('" "')
                if len(l2tokens) > 0:
                    ll = [(x.replace('"', ''), bid) for x in l2tokens]
                    bi = pd.DataFrame(ll, columns=["level", "id_bin"])
                    bor.append(bi)
                    bid += 1
            borders = pd.concat(bor, axis=0).reset_index()
        else:
            borders = pd.DataFrame([("", 1)], columns=["level", "id_bin"])
    else:
        borders = None
    return borders


def strconcat(string_list):
    return ", ".join(string_list)


def grid_encode(borders, classx):
    if classx == np.int64 or classx == np.float64 or classx == np.bool:
        if all(np.isfinite(borders)):
            code = strconcat(["%g" % bi for bi in borders])
        else:
            code = "NA"
    elif classx == np.object:
        if len(borders) > 0:
            code = "\"%s\"" % borders.at[0, 'level']
            for i in range(1, len(borders)):
                if borders.at[i, 'id_bin'] != borders.at[i - 1, 'id_bin']:
                    code += ", \"%s\"" % borders.at[i, 'level']
                else:
                    code += " \"%s\"" % borders.at[i, 'level']
        else:
            code = ""
    else:
        code = ""
    return code


def interval_name(lb, ub):
    if np.isneginf(lb):
        nm1 = "(-Inf"
    else:
        nm1 = "[%g" % lb
    if np.isposinf(ub):
        nm2 = "+Inf)"
    else:
        nm2 = "%g)" % ub
    nm = "%s, %s" % (nm1, nm2)
    return nm


def create_summary(h0, missbin):
    lb = [-np.inf, *h0]
    ub = [*h0, +np.inf]
    i0 = range(len(h0) + 1)
    nm = [interval_name(lb[i], ub[i]) for i in i0]
    if {missbin} < set(i0):  # missing is merged with an existing bin
        i1 = i0
        nm[missbin] += " NA"
    else:
        i1 = [*i0, missbin]
        nm = [*nm, "NA"]
        lb = [*lb, np.nan]
        ub = [*ub, np.nan]
    su = pd.concat([pd.Series(nm), pd.Series(lb), pd.Series(ub)], axis=1)
    su.index = i1
    su.columns = ['bin', 'LB', 'UB']
    return su


def complete_summary(d, t0):
    # d: DataFrame with integer predictor and the target
    # t0: DataFrame with summary skeleton [bin, id_bin]
    nt = len(d)
    t = t0.copy().assign(n=0, m=0, r=0.0, p=0.0, woe=0.0)
    for i in range(len(t0)):
        ii = d.loc[:, 'x'] == t0.index[i]
        t.at[i, 'n'] = ii.sum()
        t.at[i, 'm'] = d.loc[ii, 'y'].sum()
        t.at[i, 'r'] = t.at[i, 'n'] / nt
        if t.at[i, 'n'] > 0:
            t.at[i, 'p'] = t.at[i, 'm'] / t.at[i, 'n']
    t['woe'] = woeagg(t)
    return t


def make_summary(j, y, t0, code, bts, pprofile, missbin):
    d = pd.DataFrame({
        'x': pd.Series(j),
        'y': y,
    })
    assert len(d) > 0, "DataFrame with group indexes is empty"
    if any(bts) and any(~bts):
        sn = complete_summary(d.loc[~bts, :], t0)
        st = complete_summary(d.loc[bts, :], t0)
    else:
        sn = complete_summary(d, t0)
        st = None
    b = pprofile
    b['flag_predictor'] = 1
    b['nbins'] = len(t0)
    b['missbin'] = missbin
    b['borders'] = sn['bin']
    b['code'] = code
    b['summary_train'] = sn
    b['summary_test'] = st
    b['gini_train'] = gini_table(sn)
    b['lift_train'] = lift_table(sn)
    b['gini_test'] = gini_table(st)
    b['lift_test'] = lift_table(st)
    return b


def group_summary(tt):
    # pro character prediktory zgrupne podle id_bin
    # vstup DataFrame[level, id_bin]
    # vystup DataFrame[id_bin, bin]
    yy = tt.groupby('id_bin').agg(strconcat)
    yy['id_bin'] = yy.index
    yy.columns = ['id_bin', 'bin']
    return yy


def gini_table(tt):
    # calculates gini out of summary table
    if tt is None:
        gg = 0
    else:
        nt = len(tt)
        if nt > 1:
            t2 = tt.sort_values(by="woe")
            n1 = t2['m'].values
            n0 = t2['n'].values - n1
            F0 = np.cumsum(n0) / n0.sum()
            F1 = np.cumsum(n1) / n1.sum()
            gi = F0[:-1] * F1[1:] - F0[1:] * F1[:-1]
            gg = sum(gi)
        else:
            gg = 0
    return gg


def lift_table(tt):
    if tt is None:
        lift = 1.0
    else:
        p0 = tt['m'].sum() / tt['n'].sum()
        lift = tt['p'].max() / p0
    return lift


def woeagg(tt):
    n1 = tt['m'].values
    n0 = tt['n'].values - n1
    ivalid = (n0 > 0) & (n1 > 0)
    sigj = np.zeros(len(tt))
    sigj[ivalid] = np.log(n1[ivalid] / n0[ivalid])
    sig = np.log(sum(n1[ivalid]) / sum(n0[ivalid]))
    woe = sigj - sig
    woe[n1 == 0] = -3.0
    woe[n0 == 0] = +3.0
    woe[(n0 == 0) & (n1 == 0)] = 0.0
    woe = np.round(woe, 3)
    return woe


def binning_manual(x, y, bts, code, missbin):
    # bts: boolean testing set indicator
    (xv, classx, nmiss) = preprocess(x)
    pprofile = predictor_profile(x)
    if code == "" or code == "NA" or pprofile['flag_predictor'] == 0:
        b = pprofile
        b['flag_predictor'] = 0
        b['nbins'] = 1
        b['missbin'] = 1
        b['borders'] = None
        b['code'] = code
        b['gini_test'] = 0
        b['lift_test'] = 1
        b['gini_train'] = 0
        b['lift_train'] = 1
    elif classx == np.int64 or classx == np.float64 or classx == np.bool:
        h0 = grid_decode(code, classx)
        j = np.digitize(x.values, bins=h0, right=False)
        j[x.isnull()] = missbin
        t0 = create_summary(h0, missbin)
        b = make_summary(j, y, t0, code, bts, pprofile, missbin)
    elif classx == np.object:
        tt = grid_decode(code, classx)  # prevodni tabulka level -> bin
        j = np.zeros((len(x),))
        for level, id_bin in zip(tt.level, tt.id_bin):
            j[x == level] = id_bin
        j[x.isnull()] = missbin
        t0 = group_summary(tt)
        b = make_summary(j, y, t0, code, bts, pprofile, missbin)
    else:
        b = pprofile
    return b


def bins_eq_frequency(x, nbins, dec):
    pb = np.linspace(start=1, stop=nbins - 1, num=nbins - 1) / nbins
    h0 = x.dropna().quantile(pb)
    h1 = np.unique(np.round(h0, dec))
    if np.nansum(x < h1[0]) == 0:  # mass probability at min value greater than 1/nbins
        if len(h1) >= 2:
            h2 = h1[1:]
        elif np.nanmax(x) == h1[0]:  # mass probability at min value equal to 1 (all observations concentrated at one value)
            h2 = [np.nan]
        else:  # mass probability at min value greater than (nbins - 1)/nbins
            h2 = [np.min(x[x > h1[0]])]
    else:
        h2 = h1
    return h2


def suggest_decimal(pprofile):
    classx = pprofile['cl']
    if classx == np.int64 or classx == np.bool:
        dec = 0
    elif classx == np.float64:
        q = pprofile['quantiles']  # Series with index
        p25 = q.at[0.25]
        p75 = q.at[0.75]
        if p75 > p25:
            dec = int(round(3 - np.log10(p75 - p25), 0))
        else:
            dec = 2
    else:
        dec = 2
    return dec


def binning_eq_frequency(x, y, bts, nbins):
    pp = predictor_profile(x)
    classx = pp['cl']
    if pp['flag_predictor'] == 1:
        if classx == np.int64 or classx == np.float64 or classx == np.bool:
            dec = suggest_decimal(pp)
            borders = bins_eq_frequency(x, nbins, dec)
            code = grid_encode(borders, classx)
            missbin = len(borders) + 1
            bx = binning_manual(x, y, bts, code, missbin)
        elif classx == np.object:
            v = x.dropna().unique()
            v.sort()
            v = [*v, 'NA']
            ids = pd.Series(range(len(v)))
            borders = pd.concat([pd.Series(v), ids], axis=1)
            borders.columns = ['level', 'id_bin']
            code = grid_encode(borders, classx)
            missbin = len(v) - 1  # posledni bin
            bx = binning_manual(x, y, bts, code, missbin)
        else:
            bx = pp
    else:
        bx = pp
    return bx


def pred_binning(x, y, bts, bfun, **kwargs):
    bx = bfun(x, y, bts, **kwargs)
    return bx


def pred_grp(x, b):
    classx = b.get('cl', np.int)
    if b.get('code', '') == "" or b.get('flag_predictor', 0) == 0:
        j = np.zeros(len(x))
    elif classx == np.int64 or classx == np.float64 or classx == np.bool:
        h0 = grid_decode(b['code'], classx)
        j = np.digitize(x.values, bins=h0, right=False)
        j[x.isnull()] = b['missbin']
    elif classx == np.object:
        tt = grid_decode(b['code'], classx)
        j = np.zeros((len(x),))
        for level, id_bin in zip(tt.level, tt.id_bin):
            j[x == level] = id_bin
        j[x.isnull()] = b['missbin']
    else:
        j = np.zeros(len(x))
    return j


def pred_woe(x, b):
    gg = pred_grp(x, b)
    woe = np.zeros(len(x))
    tt = b.get('summary_train', pd.DataFrame(index=[]))
    for i in tt.index.tolist():
        ij = (gg == i)
        woe[ij] = tt.at[i, 'woe']
    return woe


def compatible_axes(y1, y2):
    # suggests two vertical axes for plotting ranges [0,y1] and [0,y2]
    # The ylim is selected from the sequence 0.5,1,2.5,5,10,25,50 such that
    # y1 is between (40%, 100%] of ylim1 and y2 is between (40%, 100%] of ylim2.
    # each axis can then be split into five equal bins and labeled with six tick marks,
    # e.g. 0, 5, 10, 15, 20, 25.
    # result is couple (y1lim, y2lim)
    c = np.array([1, 2.5, 5, 10])  # one cycle
    lc = np.log10(c)
    y = np.array([y1, y2])
    ar = np.log10(y)  # aspect ratio
    base = np.floor(ar)
    rem = ar - base  # remainder
    ia = np.array([0, 0])
    for i in [0, 1]:
        if rem[i] == lc[0]:
            ia[i] = 0
        elif rem[i] < lc[1]:
            ia[i] = 1
        elif rem[i] < lc[2]:
            ia[i] = 2
        else:
            ia[i] = 3
    ax = np.power(10, base) * c[ia]
    return ax


def plot_binning_aux(plotdata, tit, xlab, tr):
    lp = len(plotdata)
    xdata = np.arange(lp)
    bar_width = 0.8
    xvl = plotdata['bin'].values
    yp = plotdata['p'].values
    yn = plotdata['r'].values
    ln = plotdata['n'].values
    cax = compatible_axes(np.nanmax(yn), np.nanmax(yp))
    tn = np.linspace(0, cax[0], 6)
    tp = np.linspace(0, cax[1], 6)
    plt.rcParams['figure.figsize'] = (12.0, 6.75)
    fig, ax1 = plt.subplots()
    flag_spyder = 1  # set to 0 if run from Jupyter
    ax1.bar(xdata - flag_spyder * bar_width / 2, yn, bar_width, alpha=1.0, color=[126 / 256, 192 / 256, 238 / 256])
    ax1.set_yticks(tn)
    ax1.set_yticklabels(["{:.0%}".format(x) for x in tn])
    ax1.set_ylabel('relative bin size', color='b')
    ax1.tick_params('y', colors='b')
    ax1.grid(True, axis='y')
    ax1.set_xlabel(xlab)
    plt.xticks(xdata, xvl)
    ax1.set_title(tit)
    ax2 = ax1.twinx()
    ax2.plot(xdata, yp, 'r-*')
    ax2.hlines(tr, min(xdata) - bar_width / 2, max(xdata) + bar_width / 2, colors='r', linestyles='dashed')
    ax2.set_yticks(tp)
    ax2.set_yticklabels(["{:.0%}".format(x) for x in tp])
    ax2.set_ylabel('event rate', color='r')
    ax2.tick_params('y', colors='r')
    for i in range(len(plotdata)):
        ax2.annotate("%.1f%%" % (1e2 * yp[i]), xy=(xdata[i] - 0.04 * lp, yp[i] + 0.01 * cax[1]), color='r')
        ax1.text(xdata[i], yn[i] + 0.01,
                 '%.1f%%, %d' % (100 * yn[i], ln[i]),
                 ha='center', va='bottom',
                 weight="semibold",
                 color=[126 / 256, 192 / 256, 238 / 256])
    dist_bar_tr = np.abs(yn * cax[1] / cax[0] - tr)
    dist_line_tr = np.abs(yp - tr)
    dist_min_tr = np.amin(np.column_stack((dist_bar_tr, dist_line_tr)), axis=1)
    idx_label_tr = int(np.argmax(dist_min_tr, axis=0))
    ax2.text(xdata[idx_label_tr], tr + 0.01 * cax[1], "average = %.1f%%" % (100 * tr), ha='center', va='bottom',
             color='r')
    plt.show()
    return fig


def bin_plot(predictor_name, plot_data, train_test):
    tmp = "summary_" + train_test
    bar_plot = plt.figure()
    plt.grid(zorder=0)
    xdata = np.array(range(plot_data["nbins"]))
    bar = plt.bar(xdata,
                  100 * plot_data[tmp]["r"],
                  align="center",
                  color=[126 / 256, 192 / 256, 238 / 256],
                  # edgecolor = "none",
                  zorder=3)
    plt.title("Predictor: %s\nGini_train = %.2f" % (predictor_name, plot_data["gini_train"]), loc="left")
    x_value_labels = plot_data[tmp]['bin'].values
    plt.xticks(range(plot_data["nbins"]), x_value_labels)
    plt.xlim([-0.5, plot_data["nbins"] - 0.5])
    plt.ylim([0, 100])
    plt.xlabel(predictor_name)
    plt.ylabel("Relative bin size / Employee turnover rate")

    for i in range(len(bar)):
        rect = bar[i]
        height = rect.get_height()
        plt.gca().text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                       '%.2f' % height,
                       ha='center', va='bottom',
                       weight="semibold",
                       color=[126 / 256, 192 / 256, 238 / 256])
        if abs(1.05 * height - 1.05 * plot_data[tmp]["p"][i]) > 8 or abs(
                1.05 * height - 1.05 * plot_data[tmp]["p"][i]) > abs(
                1.05 * height - 1.05 * plot_data[tmp]["p"][i] + 10):
            plt.gca().text(rect.get_x() + rect.get_width() / 2., 100 * 1.05 * plot_data[tmp]["p"][i],
                           '%.2f' % plot_data[tmp]["p"][i],
                           ha='center', va='bottom',
                           weight="semibold",
                           color="r")
        else:
            plt.gca().text(rect.get_x() + rect.get_width() / 2., 100 * 1.05 * plot_data[tmp]["p"][i] + 10,
                           '%.2f' % plot_data[tmp]["p"][i],
                           ha='center', va='bottom',
                           weight="semibold",
                           color="r")
    vals = bar_plot.get_axes()[0].get_yticks()
    bar_plot.get_axes()[0].set_yticklabels(["{:.0%}".format(x) for x in vals / 100])
    plt.plot(xdata, 100 * plot_data[tmp]["p"],
             zorder=4,
             lw=1.5,
             color="r",
             marker=".")
    tr = sum(plot_data[tmp]["m"]) / sum(plot_data[tmp]["n"]) * 100
    plt.plot([-10, 10],
             [tr, tr],
             color="grey",
             linestyle="--",
             zorder=5,
             lw=1.5)
    return bar_plot
