import numpy as np #imported numpy and Assigned as np
import pandas as pd #imported pandas and Assigned as pd
import pickle
df = pd.read_csv('mtsamples.csv')

col = ['medical_specialty','keywords']
df = df[col]
df.columns = ['medical_specialty','keywords']

df['category_id'] = df['medical_specialty'].factorize()[0]
from io import StringIO
category_id_df = df[['medical_specialty', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'medical_specialty']].values)

df.head()

#import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(8,6))
#df.groupby('medical_specialty').description.count().plot.bar(ylim=0)
#plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.keywords).toarray()
labels = df.category_id
features.shape



pickle.dump(tfidf,open('tfidf.pkl','wb'))


from sklearn.feature_selection import chi2
import numpy as np

#N = 2

  
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

####MultinomialNB MODEL PREDICTION
print("Running Multinomial NB Model Training & Testing")
X_train, X_test, y_train, y_test = train_test_split(df['keywords'], df['medical_specialty'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

multinb = MultinomialNB().fit(X_train_tfidf,y_train)
print("Creating Multinomial NB Model Pickle Object Code")
pickle.dump(multinb,open('multinb.pkl','wb'))

pickle.dump(count_vect,open('vectorizer.pkl', 'wb'))



####LINEAR SVC MODEL PREDICTION
print("Running LINEAR SVC Model Training & Testing")
from sklearn.model_selection import train_test_split

linear = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=3)
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)


print("Creating LINEAR SVC Model Pickle Object Code")
pickle.dump(linear,open('linearsvc.pkl', 'wb'))

####Random Forest Classifier MODEL PREDICTION 
print("Running Random Forest Classifier Model Training & Testing")
from sklearn.model_selection import train_test_split

random = RandomForestClassifier()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=3)
random.fit(X_train, y_train)
y_pred = random.predict(X_test)


print("Creating Random Forest Classifier Model Pickle Object Code")
pickle.dump(random,open('random.pkl', 'wb'))
pickle.dump(id_to_category,open('id_to_category.pkl','wb'))


###LogisticRegression MODEL PREDICTION 
print("Running  LogisticRegression Model Training & Testing")
from sklearn.model_selection import train_test_split

logic = RandomForestClassifier()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=3)
logic.fit(X_train, y_train)
y_pred = logic.predict(X_test)

print("Creating LogisticRegression Model Pickle Object Code")
pickle.dump(logic,open('logic.pkl', 'wb'))

print("Model.py Executed and Created Respective Pickle Files")
print("Now Run App")
