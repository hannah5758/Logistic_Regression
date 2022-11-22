import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data
datasets = pd.read_csv('train.csv')
# print(datasets)

test_datasets = pd.read_csv('test.csv')

# null 값 제거
#inplace=True 는 이걸 나중에도 저장하는 거
datasets.drop(columns = ['Cabin'], inplace=True)
datasets.dropna(inplace=True, axis=0)

print(datasets.isnull().sum())

test_datasets.drop(columns=['Cabin'], inplace=True)
test_datasets.dropna(inplace=True, axis=0)



# print(datasets)
# 사용하지 않을 컬럼 제거
datasets.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked'], inplace=True)
print(datasets)

test_datasets.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked'], inplace=True)


# 성별 데이터 수치화
gen_dict = {'female':0, 'male':1}
datasets.replace(gen_dict, inplace=True)
print(datasets)

test_datasets.replace(gen_dict, inplace=True)

# x, y 데이터로 split
tr_x = datasets.drop(columns=['Survived'])
tr_y = datasets['Survived']

print(tr_x)
print(tr_y)

te_x = test_datasets
# te_y = test_datasets['Survived']

# tr_np = np.array(datasets).astype(np.float32)
# print(tr_

# 학습
model = LogisticRegression()
model.fit(tr_x, tr_y)

#y predict
te_predict = model.predict(te_x)
print(te_predict)

# training score 확인
tr_score = model.score(tr_x, tr_y)
print(f'training score : {tr_score}')

# te_score = model.score(te_x, te_y)
# print(f'test score : {te_score}')
