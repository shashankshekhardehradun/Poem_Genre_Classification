import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Poems.csv')
vect = TfidfVectorizer()
X=vect.fit_transform(df['content'])
y=vect.fit_transform(df['type'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train=y_train.nonzero()[1]
y_test=y_test.nonzero()[1]
clf = RandomForestClassifier(n_estimators=400, random_state= 20)
fit1=clf.fit(X_train, y_train)
pred=fit1.predict(X_test)
print(accuracy_score(y_test,pred))
