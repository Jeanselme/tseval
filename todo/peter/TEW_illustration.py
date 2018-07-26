#!/usr/bin/env python
"""
create a figure showing the median TEW curve for a random Bernoulli classifier

there might be an unrealistic hack in here, see comments in code
"""


import plt_savers

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.style.use('ggplot')
# mpl.style.use('seaborn-darkgrid')


mpl.rcParams.update({
    'font.size':16,
    'xtick.labelsize':16,
    'ytick.labelsize':16
})


# number of repeats from which to calculate median at each FPR
n_ep = 1000

prodromal_duration = 7
tew_censor_value = -3

# total length of the warning period
total_duration = prodromal_duration - tew_censor_value + 1


def index_to_TEW(index):
    """ translate index to TEW value"""
    return range(prodromal_duration, tew_censor_value-1, -1)[index]


fprs = np.arange(0, 1, 0.001)

mean_TEWs = np.zeros(len(fprs))

median_TEWs = np.zeros(len(fprs))
quant_1_TEWs = np.zeros(len(fprs))
quant_3_TEWs = np.zeros(len(fprs))


for ii, FPR in enumerate(fprs):
    simulated = np.random.binomial(1, FPR, size=(n_ep, total_duration))

    TEWs = []
    for jj, row in enumerate(simulated):

        try:
            first_index = list(row).index(1)
            TEW = index_to_TEW(first_index)
        except ValueError:
            TEW = tew_censor_value

        TEWs.append(TEW)

    median_TEWs[ii] = np.median(TEWs)


# make 'median_TEWs' convex, this is a hack since we are not doing it
# correctly (see comment below)
lastmax = tew_censor_value
for ii, thing in enumerate(median_TEWs):
    if thing > lastmax:
        lastmax = thing
    elif thing < lastmax:
        median_TEWs[ii] = lastmax
    elif thing == lastmax:
        # that's fine, no need to do anything
        pass
    else:
        raise Exception('impossible')


# NOTE: since for a real model we have the same set of novelty
# scores for each FPR, what we're doing here isn't entirely
# consistent, but we can fix that later...essentially we should be
# outputting a ton of continuous random numbers for each episode,
# include some for non-episode, and then calculate FPR and TEW
# based on that...that's pretty long-winded though, so we just
# hack it for now
msg = """WARNING: the plot produced is independent from one FPR to the
next FPR, which is inconsistent with reality"""

print(msg)


# plotting results
fig = plt.figure(figsize=(10,6))

plt.xlim([0,1])
plt.ylim([tew_censor_value - 1,
          prodromal_duration + 1])

plt.xlabel('FPR (1 - specificity)')
plt.ylabel('TEW')

plt.plot(fprs, median_TEWs)

fname = '~/Dropbox/phd/figures_thesis/ch4/from_script_TEW_illustration'
plt_savers.png(fig, fname)
plt.close(fig)
