import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import multivariate_normal

n = 1000
x1 = stats.norm(0, 1).rvs(n)
x2 = stats.norm(0, 1).rvs(n)
x4 = stats.norm(0, 1).rvs(n)
X = np.stack((x1, x2, 2 * x1, x4), 1)

# heatmapy korelacji
df = pd.DataFrame(X)
# sns.pairplot(df)  # , kind="reg"
# plt.show()

sns.set_theme(style="darkgrid")

corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    square=True,
    xticklabels=5,
    yticklabels=5,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)
plt.show()
corr
