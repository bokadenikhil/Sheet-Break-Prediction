import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv(r'data.csv')
df = df.rename(columns={df.columns[-1]:'Target'})
df = df.dropna()

encoded = pd.get_dummies(df['Target'])
df = pd.concat([df,encoded],axis='columns')
df = df.drop(['Target','NORMAL   '],axis=1)
columns = [col.strip() for col in df.columns]
df.columns = columns

X,y = df.drop('BREAK',axis=1),df['BREAK']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test,y_pred)
