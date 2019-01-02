import xgboost as xgb
from matplotlib import pyplot
params = {'learning_rate': 0.1, 
          'max_depth':5, 
          'n_estimator':100,
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':3,
          'objective': 'binary:logistic', 
          'random_state': 99, 
          'scale_pos_weight': 150,
          'silent': True}

clf = xgb.XGBClassifier(**params)
clf.fit(NB_train_x, NB_train_y, eval_set = [(NB_train_x,NB_train_y),(NB_valid_x, NB_valid_y)],
        eval_metric= 'auc', verbose =True, early_stopping_rounds=10
         )

y_pred = clf.predict_proba(NB_valid_x)
fpr, tpr, _ = roc_curve(NB_valid_y, y_pred[:,1])

plt.plot(fpr, tpr, 'o-', label="XGB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)


pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()




import xgboost as xgb

variables = ['ip', 'app', 'device', 'os', 'channel','click_hour']
for v in variables:
    NB_train_x[v] = NB_train_x[v].astype('int64')
    NB_train_x[v]=NB_train_x[v].astype('int64')
    NB_valid_x[v] = NB_valid_x[v].astype('int64')
    NB_valid_x[v]=NB_valid_x[v].astype('int64')
NB_train_x.info()
params = {'eta': 0.15, 
          'max_depth': 5, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':3,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'scale_pos_weight': 99,
          'silent': True}
          
NB_train_y = talkingData_train['is_attributed']
watchlist = [ (xgb.DMatrix(NB_train_x, NB_train_y), 'train'), (xgb.DMatrix(NB_valid_x, NB_valid_y), 'valid')]
model = xgb.train(params, xgb.DMatrix(NB_train_x, NB_train_y),  200, watchlist, verbose_eval=2, early_stopping_rounds = 10)



y_pred = model.predict(xgb.DMatrix(NB_valid_x, NB_valid_y))
fpr, tpr, _ = roc_curve(NB_valid_y, y_pred)

plt.plot(fpr, tpr, 'o-', label="XGB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)
