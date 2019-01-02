import lightgbm as lgb 


NB_train_x['ip']=NB_train_x['ip'].astype('uint32')
NB_train_x['app']=NB_train_x['app'].astype('uint16')
NB_train_x['device']=NB_train_x['device'].astype('uint16')
NB_train_x['os']=NB_train_x['os'].astype('uint16')
NB_train_x['channel']=NB_train_x['channel'].astype('uint16')
NB_train_x['click_hour']=NB_train_x['click_hour'].astype('uint16')
    
NB_valid_x['ip']=NB_train_x['ip'].astype('uint32')
NB_valid_x['app']=NB_train_x['app'].astype('uint16')
NB_valid_x['device']=NB_train_x['device'].astype('uint16')
NB_valid_x['os']=NB_train_x['os'].astype('uint16')
NB_valid_x['channel']=NB_train_x['channel'].astype('uint16')
NB_valid_x['click_hour']=NB_train_x['click_hour'].astype('uint16')



categorical = ['app', 'device', 'os', 'channel', 'click_hour']
fit_params={"early_stopping_rounds":20, 
            "eval_metric" : 'auc', 
            "eval_set" : [(NB_valid_x,NB_valid_y)],
            'eval_names': ['valid'],
            'verbose': 1,
            'feature_name': 'auto', # that's actually the default
           
            'categorical_feature': 'auto' # that's actually the default
            
            
           }

clf = lgb.LGBMClassifier(num_leaves= 15, max_depth=4, 
                         random_state=314, 
                         silent=True, 
                          metric = 'auc',
                         objective='binary',
                         n_jobs=1, 
                         n_estimators=200,
                         colsample_bytree=0.9,
                         scale_pos_weight =  150,
                         subsample_freq= 1,
                         subsample=0.7,
                         min_child_samples=100,
                         min_child_weight= 0,
                         learning_rate=0.2)
clf.fit(NB_train_x, NB_train_y, **fit_params)

y_pred = clf.predict_proba(NB_valid_x)
fpr, tpr, _ = roc_curve(NB_valid_y, y_pred[:,1])

plt.plot(fpr, tpr, 'o-', label="LightGBM")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)






train_data=lgb.Dataset(NB_train_x,label=NB_train_y)
valid_data=lgb.Dataset(NB_valid_x,label=NB_valid_y)



lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 30,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 1000,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 3,
        'verbose': 1,
        'scale_pos_weight':99
    }
categorical = ['ip','app', 'device', 'os', 'channel', 'click_hour']
evals_results = {}
bst1 = lgb.train(lgb_params, 
                     train_data, 
                 evals_result = evals_results,
                     valid_sets=[ valid_data], 
                     valid_names=['valid'], 
                      
                     num_boost_round=500,
                   
                     early_stopping_rounds=30,
                     verbose_eval=True, 
                     feval=None,
                   categorical_feature = 'auto'
                  )
