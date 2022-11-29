from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

#데이터 불러오기
datasets = datasets.load_breast_cancer()
print(datasets)
cancer_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(cancer_df)

x = datasets['data']
print(type(x))
x_slice = x[:, :20]
print(x_slice)
y = datasets['target']

#데이터 표준화 (평균이 0 이고, 표준 편차가 1인 데이터로 만들기)
scaler = StandardScaler()
x = scaler.fit_transform(x_slice)

#로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(x, y)

#예측 결과 확인하기
y_predicted = model.predict(x)
print(y_predicted)

# 정확도 평가하기
score = model.score(x,y)
print(score)

