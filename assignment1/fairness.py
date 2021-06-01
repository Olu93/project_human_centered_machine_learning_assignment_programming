# %% [markdown]
# # Submission instructions
#
# All code that you write should be in this notebook.
# Submit:
#
# * This notebook with your code added. Make sure to add enough documentation.
# * A short report, max 2 pages including any figures and/or tables (it is likely that you won't need the full 2 pages). Use [this template](https://www.overleaf.com/read/mvskntycrckw).
# * The deadline is Monday 17th of May, 17.00.
#
# For questions, make use of the "Lab" session (see schedule).
# Questions can also be posted to the MS teams channel called "Lab".
#

# %% [markdown]
# # Installing AIF360
# %% [markdown]
# In this assignment, we're going to use the AIF360 library.
# For documentation, take a look at:
#
#     * https://aif360.mybluemix.net/
#     * https://aif360.readthedocs.io/en/latest/ (API documentation)
#     * https://github.com/Trusted-AI/AIF360 Installation instructions
#
# We recommend using a dedicated Python environment for this assignment, for example
# by using Conda (https://docs.conda.io/en/latest/).
# You could also use Google Colab (https://colab.research.google.com/).
#
# When installing AIF360, you only need to install the stable, basic version (e.g., pip install aif360)
# You don't need to install the additional optional dependencies.
#
# The library itself provides some examples in the GitHub repository, see:
# https://github.com/Trusted-AI/AIF360/tree/master/examples.
#
# **Notes**
# * The lines below starting with ! can be used in Google Colab by commenting them out, or in your console
# * The first time you're running the import statements, you may get a warning "No module named tensorflow".
#   This can be ignored--we don't need it for this assignment. Just run the code block again, and it should disappear

# %%
# !pip install aif360
# !pip install fairlearn

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.datasets import Dataset
from aif360.datasets.structured_dataset import StructuredDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from IPython import get_ipython
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Exploring the data
# %% [markdown]
# **COMPAS dataset**
#
# In this assignment we're going to use the COMPAS dataset.
#
# If you haven't done so already, take a look at this article: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing.
# For background on the dataset, see https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
#
# **Reading in the COMPAS dataset**
#
# The AIF360 library has already built in code to read in this dataset.
# However, you'll first need to manually download the COMPAS dataset
# and put it into a specified directory.
# See: https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/compas/README.md.
# If you try to load in the dataset for the first time, the library will give you instructions on the steps to download the data.
#
# The protected attributes in this dataset are 'sex' and 'race'.
# For this assignment, we'll only focus on race.
#
# The label codes recidivism, which they defined as a new arrest within 2 years.
# Note that in this dataset, the label is coded with 1 being the favorable label.

# %%
get_ipython().system(
    'curl https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv --output compas-scores-two-years.csv'
)
get_ipython().system(
    'mv compas-scores-two-years.csv C:\\Users\\ohund\\Anaconda3\\envs\\fairness\\lib\\site-packages\\aif360\\datasets\\..\\data\\raw\\compas\\compas-scores-two-years.csv'
)

compas_data = load_preproc_data_compas(protected_attributes=['race'])

# %% [markdown]
# Now let's take a look at the data:

# %%
compas_data

# %% [markdown]
# **Creating a train and test split**
#
# We'll create a train (80%) and test split (20%).
#
# Note: *Usually when carrying out machine learning experiments,
# we also need a dev set for developing and selecting our models (incl. tuning of hyper-parameters).
# However, in this assignment, the goal is not to optimize
# the performance of models so we'll only use a train and test split.*
#
# Note: *due to random division of train/test sets, the actual output in your runs may slightly differ with statistics showing in the rest of this notebook.*

# %%
train_data, test_data = compas_data.split([0.5], shuffle=True)
train_data_1, test_data_1, train_data_2, test_data_2 = train_data.copy(deepcopy=True), test_data.copy(
    deepcopy=True), train_data.copy(deepcopy=True), test_data.copy(deepcopy=True)
# %% [markdown]
# In this assignment, we'll focus on protected attribute: race.
# This is coded as a binary variable with "Caucasian" coded as 1 and "African-American" coded as 0.

# %%
priv_group = [{'race': 1}]  # Caucasian
unpriv_group = [{'race': 0}]  # African-American

# %% [markdown]
# Now let's look at some statistics:

# %%
print("Training set shape: %s, %s" % train_data.features.shape)
print("Favorable (not recid) and unfavorable (recid) labels: %s; %s" %
      (train_data.favorable_label, train_data.unfavorable_label))
print("Protected attribute names: %s" % train_data.protected_attribute_names)
# labels of privileged (1) and unprovileged groups (0)
print("Privileged (Caucasian) and unprivileged (African-American) protected attribute values: %s, %s" %
      (train_data.privileged_protected_attributes, train_data.unprivileged_protected_attributes))
print("Feature names: %s" % train_data.feature_names)

# %% [markdown]
# Now, let's take a look at the test data and compute the following difference:
#
# $$𝑃(𝑌=favorable|𝐷=unprivileged)−𝑃(𝑌=favorable|𝐷=privileged)$$
#

# %%
metric_test_data = BinaryLabelDatasetMetric(test_data, unprivileged_groups=unpriv_group, privileged_groups=priv_group)
print("Mean difference (statistical parity difference) = %f" % metric_test_data.statistical_parity_difference())


def compute_statistical_parity(data, unpriv_group, priv_group):
    if isinstance(data, pd.DataFrame):
        transformed_data = BinaryLabelDataset(df=data,
                                              label_names=["two_year_recid"],
                                              protected_attribute_names=["race"],
                                              favorable_label=0,
                                              unfavorable_label=1)
    else:
        transformed_data = data

    metric_test_data = BinaryLabelDatasetMetric(transformed_data,
                                                unprivileged_groups=unpriv_group,
                                                privileged_groups=priv_group)
    parity_difference = metric_test_data.statistical_parity_difference()
    print(f"Mean difference (statistical parity difference) = {parity_difference}")
    return parity_difference


def compute_metrics(data, predictions, unpriv_group, priv_group):
    transformed_data = BinaryLabelDataset(df=data,
                                          label_names=["two_year_recid"],
                                          protected_attribute_names=["race"],
                                          favorable_label=0,
                                          unfavorable_label=1) if isinstance(data, pd.DataFrame) else data
    t_data_train_true = transformed_data.copy(deepcopy=True)
    t_data_train_pred = transformed_data.copy(deepcopy=True)
    t_data_train_pred.labels = predictions.reshape(-1, 1)
    metric_test_data = ClassificationMetric(
        t_data_train_true,
        t_data_train_pred,
        unprivileged_groups=unpriv_group,
        privileged_groups=priv_group,
    )
    tpr_difference = metric_test_data.true_positive_rate_difference()
    tpr_priviledged = metric_test_data.true_positive_rate(True)
    tpr_unpriviledged = metric_test_data.true_positive_rate(False)
    return tpr_difference, tpr_priviledged, tpr_unpriviledged


compute_statistical_parity(test_data.convert_to_dataframe()[0], unpriv_group, priv_group)
# %% [markdown]
# To be clear, because we're looking at the original label distribution this is the base rate difference between the two groups

# %%
metric_test_data.base_rate(False)  # Base rate of the unprivileged group

# %%
metric_test_data.base_rate(True)  # Base rate of the privileged group

# %% [markdown]
# To explore the data, it can also help to convert it to a dataframe.
# Note that we get the same numbers as the reported base rates above,
# but because when calculating base rates the favorable label is taken (which is actually 0),  it's 1-...

# %% [markdown]
# **Report**
#
# Report basic statistics in your report, such as the size of the training and test set.
#
# Now let's explore the *training* data further.
# In your report include a short analysis of the training data. Look at the base rates of the outcome variable (two year recidivism) for the combination of both race and sex categories. What do you see?
# %%
test_data.convert_to_dataframe()[0].groupby(['race'])['two_year_recid'].describe()
# %%
test_data.convert_to_dataframe()[0].groupby(['sex'])['two_year_recid'].describe()
# %%
test_data.convert_to_dataframe()[0].groupby(['race', 'sex'])['two_year_recid'].describe()
# %%
train_data.convert_to_dataframe()[0].groupby(['race'])['two_year_recid'].describe()
# %%
train_data.convert_to_dataframe()[0].groupby(['sex'])['two_year_recid'].describe()
# %%
train_data.convert_to_dataframe()[0].groupby(['race', 'sex'])['two_year_recid'].describe()

# %% [markdown]
# # Classifiers
#
# **Training classifiers**
#
# Now, train the following classifiers:
#
# 1. A logistic regression classifier making use of all features
# 2. A logistic regression classifier without the race feature
# 3. A classifier after reweighting instances in the training set https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.Reweighing.html.
#     * Report the weights that are used for reweighing and a short interpretation/discussion.
# 4. A classifier after post-processing
# https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing
#
# For training the classifier we recommend using scikit-learn (https://scikit-learn.org/stable/).
# AIF360 contains a sklearn wrapper, however that one is in development and not complete.
# We recommend using the base AIF360 library, and not their sklearn wrapper.
#
# **Report**
#
# For each of these classifiers, report the following:
# * Overall precision, recall, F1 and accuracy.
# * The statistical parity difference. Does this classifier satisfy statistical parity? How does this difference compare to the original dataset?
# * Difference of true positive rates between the two groups. Does the classifier satisfy the equal opportunity criterion?
#
#
# %%

all_results = []

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

data = train_data.convert_to_dataframe()[0]
X, y = data.drop("two_year_recid", axis=1), data["two_year_recid"]
log_reg_all = LogisticRegression().fit(X, y)

t_data = test_data.convert_to_dataframe()[0]
X_, y_ = t_data.drop("two_year_recid", axis=1), t_data["two_year_recid"]
pred_ = log_reg_all.predict(X_)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)
tmp = t_data.copy()
tmp["two_year_recid"] = pred_

parity_diff = compute_statistical_parity(tmp, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data, pred_, unpriv_group, priv_group)
all_results.append(("All Features", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")

# %%
data = train_data.convert_to_dataframe()[0]
X, y = data.drop(["two_year_recid", "race"], axis=1), data["two_year_recid"]
log_reg_all = LogisticRegression().fit(X, y)

t_data = test_data.convert_to_dataframe()[0]
X_, y_ = t_data.drop(["two_year_recid", "race"], axis=1), t_data["two_year_recid"]
pred_ = log_reg_all.predict(X_)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)
tmp = t_data.copy()
tmp["two_year_recid"] = pred_

parity_diff = compute_statistical_parity(tmp, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data, pred_, unpriv_group, priv_group)
all_results.append(("Without Race", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")

bag = [X_, y_, pred_]
# %%
log_reg_RW = Reweighing(unpriv_group, priv_group).fit(train_data)
transformed_data = log_reg_RW.transform(train_data)
display(train_data.instance_weights.mean(), train_data.instance_weights.std())
display(transformed_data.instance_weights.mean(), transformed_data.instance_weights.std())
t_data = train_data.convert_to_dataframe()[0]
t_data["weights"] = transformed_data.instance_weights
t_data_blacks = t_data[t_data.race == 0]
t_data_whites = t_data[t_data.race == 1]
print(t_data_blacks.weights.describe())
print(t_data_whites.weights.describe())
t_data.boxplot(["weights"], by="race", figsize=(10, 5))
plt.show()

# %%
data = train_data.convert_to_dataframe()[0]
X, y = data.drop(["two_year_recid"], axis=1), data["two_year_recid"]
log_reg_all = LogisticRegression().fit(X, y, sample_weight=transformed_data.instance_weights)

t_data = test_data.convert_to_dataframe()[0]
X_, y_ = t_data.drop(["two_year_recid"], axis=1), t_data["two_year_recid"]
pred_ = log_reg_all.predict(X_)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)
tmp = t_data.copy()
tmp["two_year_recid"] = pred_

parity_diff = compute_statistical_parity(tmp, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data, pred_, unpriv_group, priv_group)
all_results.append(("Reweighting", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")
bag += [X_, y_, pred_]
# %%
# tmp = BinaryLabelDatasetMetric()
log_reg_RW = Reweighing(unpriv_group, priv_group).fit(train_data)
transformed_data = log_reg_RW.transform(train_data)

data = train_data.convert_to_dataframe()[0]
X, y = data.drop(["two_year_recid", "race"], axis=1), data["two_year_recid"]
log_reg_all = LogisticRegression().fit(X, y, sample_weight=transformed_data.instance_weights)

t_data = test_data.convert_to_dataframe()[0]
X_, y_ = t_data.drop(["two_year_recid", "race"], axis=1), t_data["two_year_recid"]
pred_ = log_reg_all.predict(X_)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)
tmp = t_data.copy()
tmp["two_year_recid"] = pred_

parity_diff = compute_statistical_parity(tmp, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data, pred_, unpriv_group, priv_group)
all_results.append(("Reweighting without Race", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")
bag += [X_, y_, pred_]
#  %%
data = train_data.convert_to_dataframe()[0]
X, y = data.drop(["two_year_recid"], axis=1), data["two_year_recid"]
log_reg_all = LogisticRegression().fit(X, y)

# =======================================
t_data_train = train_data.convert_to_dataframe()[0]
X_, y_ = t_data_train.drop(["two_year_recid"], axis=1), t_data_train["two_year_recid"]
pred_ = log_reg_all.predict(X_)

t_data_train_true = train_data.copy(deepcopy=True)
t_data_train_pred = train_data.copy(deepcopy=True)
t_data_train_pred.labels = pred_.reshape(-1, 1)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)

parity_diff = compute_statistical_parity(t_data_train_pred, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data_train_true, pred_, unpriv_group, priv_group)
all_results.append(("Metrics on Training Data", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")

# =======================================
t_data_test = test_data.convert_to_dataframe()[0]
X_, y_ = t_data_test.drop(["two_year_recid"], axis=1), t_data_test["two_year_recid"]
pred_ = log_reg_all.predict(X_)

t_data_test_true = test_data.copy(deepcopy=True)
t_data_test_pred = test_data.copy(deepcopy=True)
t_data_test_pred.labels = pred_.reshape(-1, 1)

ps = precision_score(y_, pred_)
rs = recall_score(y_, pred_)
fs = f1_score(y_, pred_)
as_ = accuracy_score(y_, pred_)

parity_diff = compute_statistical_parity(t_data_test_pred, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data_test_true, pred_, unpriv_group, priv_group)
all_results.append(("Metrics on Test Data", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")

# %%

y_valid_pred = np.zeros_like(test_data.labels)

odd_equalizer_EQ = EqOddsPostprocessing(unpriv_group, priv_group).fit(t_data_train_true, t_data_train_pred)
transformed_data = odd_equalizer_EQ.predict(t_data_test_pred)

ps = precision_score(t_data_test_true.labels, transformed_data.labels)
rs = recall_score(t_data_test_true.labels, transformed_data.labels)
fs = f1_score(t_data_test_true.labels, transformed_data.labels)
as_ = accuracy_score(t_data_test_true.labels, transformed_data.labels)

parity_diff = compute_statistical_parity(transformed_data, unpriv_group, priv_group)
tpr_diff, tpr_priv, tpr_unpriv = compute_metrics(t_data_test_pred, transformed_data.labels, unpriv_group, priv_group)
all_results.append(("EqOdds Postprocessing", ps, rs, fs, as_, parity_diff, tpr_diff, tpr_priv, tpr_unpriv))

print(f"The precision is {ps}.\nThe recall is {rs}.\nThe F1 is {fs}.\nThe accuracy is {as_}.")

# %%

col_names = "Configuration Precision Recall F1 Accuracy Stat.-Parity TPR-Diff TPR_1 TPR_0".split()
results_df = pd.DataFrame(all_results, columns=col_names)

display(results_df)
display("")
cap = "This table shows the metric results for various Logisitic Regression configurations on the Compas Dataset"
print(
    results_df.to_latex(caption=cap, label="results", header=col_names,
                        float_format="%.5f").replace("{tabular}", "{tabularx}"))

# %% [markdown]
# # Discussion
#
# **Report**
# * Shortly discuss your results. For example, how do the different classifiers compare against each other?
# * Also include a short ethical discussion (1 or 2 paragraphs) reflecting on these two aspects: 1) The use of a ML system to try to predict recidivism; 2) The public release of a dataset like this.
#

# %%
