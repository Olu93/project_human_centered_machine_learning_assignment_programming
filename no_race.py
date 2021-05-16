


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