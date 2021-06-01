# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Programming Assignment II: Explainability
#
# In this assignment you will train machine learning models and experiment with techniques discussed in the lectures.
# This assignment makes use of existing Python libraries. We have provided links to tutorials/examples if you're not familiar with them yet.
#
#
# All code that you write should be in this notebook. You should submit:
# * This notebook with your code added. Make sure to add enough documentation.
# * A short report, max 3 pages including any figures and/or tables. Use this [template](https://www.overleaf.com/read/mvskntycrckw).
# * Zip the notebook .ipynb and report .pdf files in a file with name format 'Prog_Explainability_Group_[X].zip', where X is your programming group ID (e.g. Prog_Explainability_Group_10.zip). The .ipynb and .pdf files should also have the same name as the zip file.
#
#
# Important notes:
# * Deadline for this assignment is **Monday June 7, 17:00**.
# * Send it to both Heysem Kaya (h.kaya@uu.nl) and Yupei Du (y.du@uu.nl), CCing your programming partner.
# * Title of the email: [INFOMHCML] Explainability programming assignment submission [X], with X the number of your group.
# * There will be a lab session to assist you with the assignment on **Tuesday, June 1, between 9:00-12:45 over Lab Channel in Teams**.
#
#
# %% [markdown]
# ## Installation
#
# For this assignment, we are going to use the following Python packages:
#
# matplotlib, pandas, statsmodels, interpret, scikit-learn, openpyxl and graphviz

# %%
# Installing packages
# get_ipython().system('conda install python-graphviz -y')
# get_ipython().system('pip install matplotlib pandas statsmodels sklearn')
# get_ipython().system('pip install interpret openpyxl')

# %% [markdown]
# ## Downloading the data
# We are going to use the combined cycle power plant dataset. This dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. We have the following features: hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V). We will train ML models to predict the net hourly electrical energy output (EP) of the plant.
#
# For a detailed description, see: [[Description](https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant)]
#
# We first need to download and prepare data.
#

# %%
# Download and unzip data
# get_ipython().system('curl https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip --output CCPP.zip')
# get_ipython().system('unzip CCPP.zip')

# %% [markdown]
# ## Loading and preprocessing the data
# We split the data into training (first 5000 instances) and validation (the subsequent 2000) and test (the last 2568) sets. We will use the training set to train a model, and validation set to optimize the model hyper-parameters.
#

# %%
# Load and prepare data
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# global variables
DATA_FILENAME = 'CCPP/Folds5x2_pp.xlsx'
FEATURE_NAMES = ['AT', 'V', 'AP', 'RH']
LABEL_NAME = 'PE'


# Load the data from the excel file
def load_data():
    def split_feature_label(data_set):
        features = data_set[FEATURE_NAMES]
        labels = data_set[LABEL_NAME]
        return features, labels

    data = pd.read_excel(DATA_FILENAME, engine='openpyxl')
    train_set, dev_set, test_set = data[:5000], data[5000:7000], data[7000:]

    train_features, train_labels = split_feature_label(train_set)
    dev_features, dev_labels = split_feature_label(dev_set)
    test_features, test_labels = split_feature_label(test_set)

    return train_features, train_labels, dev_features, dev_labels, test_features, test_labels


# preprocess (by z-normalization) the data for the regression task
# return the normalized feature sets and corresponding target variables
def prepare_load_regression_data():
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = load_data()

    scaler = StandardScaler()
    scaler = scaler.fit(train_features)
    train_features = pd.DataFrame(data=scaler.transform(train_features), columns=FEATURE_NAMES)
    dev_features = pd.DataFrame(data=scaler.transform(dev_features), columns=FEATURE_NAMES)
    test_features = pd.DataFrame(data=scaler.transform(test_features), columns=FEATURE_NAMES)

    return train_features, train_labels, dev_features, dev_labels, test_features, test_labels


# binarize the data for the classification task
# return the discretized feature sets and corresponding target variables
def prepare_load_classification_data():
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = load_data()
    feature_mean, label_mean = train_features.mean(axis=0), train_labels.mean(axis=0)

    train_features = pd.DataFrame(data=np.where(train_features > feature_mean, 1, 0), columns=FEATURE_NAMES)
    dev_features = pd.DataFrame(data=np.where(dev_features > feature_mean, 1, 0), columns=FEATURE_NAMES)
    test_features = pd.DataFrame(data=np.where(test_features > feature_mean, 1, 0), columns=FEATURE_NAMES)
    train_labels = pd.DataFrame(data=np.where(train_labels > label_mean, 1, 0), columns=[LABEL_NAME])
    dev_labels = pd.DataFrame(data=np.where(dev_labels > label_mean, 1, 0), columns=[LABEL_NAME])
    test_labels = pd.DataFrame(data=np.where(test_labels > label_mean, 1, 0), columns=[LABEL_NAME])

    return train_features, train_labels, dev_features, dev_labels, test_features, test_labels


# %% [markdown]
# ## Training and Interpreting a Linear Regression Model
#
# **Q1**. Train a linear regression model (we recommend the statsmodels package) and report $R^2$ (goodness of fit) statistic.
#
# For model interpretability, provide for each feature (+ the bias variable) the following in tabular format:
# * Weight estimates
# * SE (standard error of estimates)
# * T statistics
#
#
# Further Questions regarding the linear model (to be included in the report):
#
# **Q2**. Which three features are the most important?
#
# **Q3**. How does the gas turbine energy yield (EP) change with unit (one degree C) increase of the ambient temperature given that all other feature values remain the same?
#
# **Q4**. Visualize the weight estimates using 95% confidence intervals.
#
# **Q5**. Show bar graph illustrations of the feature effects for the first two validation set instances.

# %%
# We recommend the statsmodels package
# Linear regression
import io
import statsmodels.api as sm
import statsmodels.formula.api as smf
# %%
train_features, train_labels, dev_features, dev_labels, test_features, test_labels = prepare_load_regression_data()
md = smf.ols(f"{LABEL_NAME} ~ AT+V+AP+RH", data=train_features.join(train_labels))
mdf = md.fit()
print(mdf.summary())
# %%
results_df = pd.read_csv(io.StringIO(mdf.summary().tables[1].as_csv()), index_col=0, header=0)
cap = "This table shows the regression results for a Linear Regression on the CCPP"
header = "Coefficients SE T P [2.5% 97.5%]".split()
print("")
print(
    results_df.to_latex(
        caption=cap,
        label=LABEL_NAME,
        header=header,
        float_format="%.3f",
    ).replace("{tabular}", "{tabular}"))
#

# %%
offset = 1
barWidth = .5
coeff_values = results_df.iloc[offset:, 0]
lower_bound = results_df.iloc[offset:, 4]
upper_bound = results_df.iloc[offset:, 5]
label_names = coeff_values.keys()
bar_position = list(range(len(coeff_values)))
cnf_intvs = (upper_bound - lower_bound) / 2
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.barh(
    bar_position,
    coeff_values,
    # width=barWidth,
    # color='blue',
    edgecolor='black',
    xerr=cnf_intvs,
    capsize=7,
    # label='poacee',
)
ax.set_yticks([r for r in bar_position])
ax.set_yticklabels(label_names)
fig.suptitle(f"Confidence intervals for the coefficients - Intercept at {results_df.iloc[0, 0]}")
fig.tight_layout()
plt.savefig("figures/confidence_intervalls.png")
plt.show()

# %%
index = ["Instance 0", "Instance 1"]
cols = dev_features[:2].columns
FE_tmp = dev_features[:2].values * results_df.iloc[1:, 0].values
FE = pd.DataFrame(FE_tmp, columns=cols, index=index)
fig = plt.figure(figsize=(10, 5))
ax = plt.gca()
FE.plot.barh(
    ax=ax,
    edgecolor='black',
    capsize=7,
)
fig.suptitle("Feature effects")
plt.savefig("figures/feature_effects.png")
plt.show()
# %% [markdown]
# **Q6.** Reflection: why would training a regression tree not work well for this dataset?
import seaborn as sns
sns.pairplot(data=train_features.join(train_labels), kind="kde", y_vars=["PE"])
plt.savefig("figures/PE_vs_Others_kde.png")
plt.show()
# %%
import seaborn as sns
sns.pairplot(data=train_features.join(train_labels), kind="hist", y_vars=["PE"])
plt.savefig("figures/PE_vs_Others_hist.png")
plt.show()
# %%
import seaborn as sns
sns.pairplot(data=train_features.join(train_labels), kind="scatter", y_vars=["PE"])
plt.savefig("figures/PE_vs_Others_scatter.png")
plt.show()

# %%
import seaborn as sns
plotting_data = train_features.join(train_labels)
plotting_data_melted = plotting_data.melt(id_vars="PE",
                                          value_vars="AT V AP RH".split(),
                                          var_name="Variable",
                                          value_name="X")

sns.lmplot(
    x='X',
    y='PE',
    col='Variable',  # the column by which you need to split - needs to be categorical
    data=plotting_data_melted,
    col_wrap=2,  # number of columns per row
    
    sharex=False,
    sharey=True,  # will repeat ticks, coords for each plot
    line_kws={'color': 'red', "linewidth":3}  # symbol for regression line
)
plt.savefig("figures/PE_vs_Others_reg.png")
plt.show()

# %% [markdown]
# ## Training and Interpreting Classification Models
# Using the preprocessing function implemented above to prepare the dataset for  the classification task. This function simply binarizes all variables including the target variable (EP) using the respective training set mean as threshold. A value of 1 means a high value vs 0 a low(er than average) value. Note that we do the feature binarization to ease interpretation of the models, normally that is not necessary for classification models.
#
#
train_features, train_labels, dev_features, dev_labels, test_features, test_labels = prepare_load_classification_data()
md = smf.ols(f"{LABEL_NAME} ~ AT+V+AP+RH", data=train_features.join(train_labels))
mdf = md.fit()
print(mdf.summary())

# %% [markdown]
# ### Training and Interpreting EBM
# Train a Explainable Boosting Machine (with [interpret.ml](https://github.com/interpretml/interpret/))
#
# For a tutorial see: [[Tutorial](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Interpretable%20Classification%20Methods.ipynb)]
#
# **Q7**. Report (global) feature importances for EBM as a table or figure. What are the most important three features in EBM? Are they the same as in the linear model?
#
# w_1X + w_2Y + w_3(XY) = Z
# %%
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

train_features, train_labels, dev_features, dev_labels, test_features, test_labels = prepare_load_classification_data()
ebm = ExplainableBoostingClassifier(n_jobs=-1)
ebm.fit(train_features, train_labels)
# EBM
#%% # Global Explanation
ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)
#%% # Local Explanation
ebm_local = ebm.explain_local(dev_features[:5], dev_labels[:5], name='EBM')
show(ebm_local)
#%% # Performance
from interpret.perf import ROC
ebm_perf = ROC(ebm.predict_proba).explain_perf(dev_features, dev_labels, name='EBM')
show(ebm_perf)
# %% [markdown]
# ### Training and Explaining Neural Networks
# Train two Neural Networks:
# 1. One-layer MLP (ReLU activation function + 50 hidden neurons)
# 2. Two-layer MLP (ReLU activation function + (20, 20) hidden neurons)
#
# We recommend to use the Adam optimizer. Fine-tune the learning rate and any other hyper-parameters you find necessary.
#
# For a tutorial see: [[Tutorial](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)]

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# One-layer MLP
one_layer_nn = MLPClassifier((50, ))
one_layer_nn.fit(train_features, train_labels)
pred_ = one_layer_nn.predict(dev_features)
score = accuracy_score(dev_labels, pred_)
print(f"Train 1 layer neural network achieved an accuarcy of {score}")
# Two-layer MLP
two_layer_nn = MLPClassifier((20, 20))
two_layer_nn.fit(train_features, train_labels)
pred_ = two_layer_nn.predict(dev_features)
score = accuracy_score(dev_labels, pred_)
print(f"Train 2 layer neural network achieved an accuarcy of {score}")
# %% [markdown]
# You can check the tutorials for SHAP and LIME explanations for neural networks
# [[LIME Tutorial](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)]
# [[SHAP Tutorial](https://nbviewer.jupyter.org/github/interpretml/interpret/blob/master/examples/python/notebooks/Explaining%20Blackbox%20Classifiers.ipynb)]
#
#
# **Q8**. Provide explanations for randomly selected three test set instances using two explanation methods (LIME and SHAP) with two NN models  (namely the single-hidden layer NN model and the two-hidden-layer NN model: 2 x 2 x 3 = 12 explanations in total).

# %%
# Global explanations
import graphviz
from interpret import show, preserve
import random
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel
# Local explanations (SHAP and LIME)
np.random.seed(42)
indices_to_pick = np.random.randint(0, len(dev_features), 3)
feature_names = list(train_features.columns)
background_val = np.median(train_features, axis=0).reshape(1, -1)
print(f"Picked indices: {indices_to_pick}")
X, y = dev_features.loc[indices_to_pick], dev_labels.loc[indices_to_pick]
# %% LIME
#Blackbox explainers need a predict function, and optionally a dataset
lime = LimeTabular(predict_fn=one_layer_nn.predict_proba, data=train_features, random_state=1)
lime_local = lime.explain_local(X, y, name='LIME')
show(lime_local)
# %% LIME
lime = LimeTabular(predict_fn=two_layer_nn.predict_proba, data=train_features, random_state=1)
lime_local = lime.explain_local(X, y, name='LIME')
show(lime_local)
# %% SHAP
shap = ShapKernel(predict_fn=one_layer_nn.predict_proba, data=background_val, feature_names=feature_names)
shap_local = shap.explain_local(X, y, name='SHAP')
show(shap_local)
# %% SHAP
shap = ShapKernel(predict_fn=two_layer_nn.predict_proba, data=background_val, feature_names=feature_names)
shap_local = shap.explain_local(X, y, name='SHAP')
show(shap_local)
# %%
