

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
talkingData_train['click_hour'] = talkingData_train['click_time'].dt.strftime('%H').astype('category')
talkingData_test['click_hour']= talkingData_test['click_time'].dt.strftime('%H').astype('category')
talkingData_valid['click_hour']= talkingData_valid['click_time'].dt.strftime('%H').astype('category')

NB_train_x = talkingData_train.iloc[:,0:5].join(talkingData_train.iloc[:,8:])
NB_train_x.head()
NB_train_y = talkingData_train['is_attributed']


NB = model.fit(NB_train_x, NB_train_y)
#validset import
validset = talkingData_valid.iloc[:,:]

NB_valid_x = validset.iloc[:,0:5].join(validset.iloc[:,8:])
NB_valid_y = validset['is_attributed']

pred_valid = NB.predict_proba(NB_valid_x)

#sk_metric.confusion_matrix(pred_valid, NB_valid_y)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(NB_valid_y,pred_valid[:,1])
fpr, tpr, thresholds
plt.plot(fpr, tpr, 'o-', label="Gaussian NB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver operating characteristic example')
plt.show()

from sklearn.metrics import auc
auc(fpr, tpr)
