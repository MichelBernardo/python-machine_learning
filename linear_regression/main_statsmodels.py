import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns


base = pd.read_csv('data/mtcars.csv')

# print(base.shape)
# print(base.head)
base = base.drop(['model'], axis=1)  # Remove the car's model column


# ========================= Correlation Analysis
# corr = base.corr()
# sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
# plt.show()


# ========================= Scatter plot Analysis
# column_pairs = [('mpg','cyl'),('mpg','disp'),('mpg','hp'),('mpg','wt'),('mpg','drat'),('mpg','vs')]
# n_plots = len(column_pairs)
# fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6,4 * n_plots))

# for i, pair in enumerate(column_pairs):
#     x_col, y_col = pair
#     sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
#     axes[i].set_title(f'{x_col} vs {y_col}')

# plt.tight_layout()
# plt.show()


# ========================= Model
# AIC: 156.6 ~ BIC: 162.5
# model = sm.ols(formula='mpg ~ wt + disp + hp', data=base)

# AIC: 166.6 ~ BIC: 171.0 - The model has gotten worse 
# model = sm.ols(formula='mpg ~ disp + hp', data=base)

# AIC: 179.1 ~ BIC: 183.5 - The model has gotten worse
model = sm.ols(formula='mpg ~ drat + vs', data=base)
model = model.fit()
print(model.summary())

# Resid
resids = model.resid
plt.hist(resids, bins=20)
plt.xlabel('Waste')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.show()

# Q-Q Plot
stats.probplot(resids, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# Teste de Shapiro
stat, pval = stats.shapiro(resids)
print(f'Shapiro-Wiki: {stat:.3f}, p-value: {pval:.3f}')

