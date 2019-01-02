
#각 변수의 category별 실제 다운로드 확률

talkingData_train['click_hour'] = talkingData_train['click_time'].dt.strftime('%H').astype('category')
talkingData_test['click_hour']= talkingData_test['click_time'].dt.strftime('%H').astype('category')
talkingData_valid['click_hour']= talkingData_valid['click_time'].dt.strftime('%H').astype('category')


feature_names = ['ip', 'app', 'device', 'os','channel', 'click_hour']
talkingData_train['is_attributed'] = talkingData_train['is_attributed'].astype('int64')
talkingData_valid['is_attributed'] = talkingData_valid['is_attributed'].astype('int64')

for cols in feature_names:
    col_count = talkingData_train.groupby([cols])
    
    def rate_calculation(x):
        rate = (x.sum()+1)/(x.count()+len(col_count))
    
        return rate

    talkingData_train = talkingData_train.merge(col_count['is_attributed'].apply(rate_calculation).
                            reset_index().rename(columns = {'is_attributed':'{}_rate'.format(cols)}), how = 'left')

for cols in feature_names:
    col_count = talkingData_valid.groupby([cols])
    
    def rate_calculation(x):
        rate = (x.sum()+1)/(x.count()+len(col_count))
    
        return rate

    talkingData_valid = talkingData_valid.merge(col_count['is_attributed'].apply(rate_calculation).
                            reset_index().rename(columns = {'is_attributed':'{}_rate'.format(cols)}), how = 'left')

    
talkingData_train['is_attributed'] = talkingData_train['is_attributed'].astype('category')
talkingData_valid['is_attributed'] = talkingData_valid['is_attributed'].astype('category')

talkingData_train.head()
