from sklearn.ensemble import AdaBoostClassifier
estimator = list(range(100,301,50))
learning = np.arange(0.5,1.5,0.1)
best_auc = 0
best_estimator =0
best_learning_rate =0
for i in estimator:
    for j in learning:
        
        abc = AdaBoostClassifier(n_estimators = i, learning_rate = j)
        model = abc.fit(NB_train_x, NB_train_y)
        y_pred = model.predict_proba(NB_valid_x)
        fpr, tpr, _ = roc_curve(NB_valid_y, y_pred[:,1])
        auc_tmp = auc(fpr, tpr)
        if auc_tmp > best_auc:
            best_auc = auc_tmp
            best_estimator = i
            best_learning_rate = j
        
print(best_auc, best_estimator, best_learning_rate)


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators = 100, learning_rate = 0.5)
model = abc.fit(NB_train_x, NB_train_y)
y_pred = model.predict_proba(NB_valid_x)
fpr, tpr, _ = roc_curve(NB_valid_y, y_pred[:,1])
plt.plot(fpr, tpr, 'o-', label="AdaBoost")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of AdaBoost')
plt.show()
auc(fpr, tpr)
