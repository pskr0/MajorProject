import numpy as np #imported numpy and Assigned as np
import pandas as pd #imported pandas and Assigned as pd
import pickle
df = pd.read_csv('mtsamples.csv')

col = ['medical_specialty', 'description']
df = df[col]
df.columns = ['medical_specialty', 'description']

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

features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
features.shape



pickle.dump(tfidf,open('tfidf.pkl','wb'))


from sklearn.feature_selection import chi2
import numpy as np

N = 2

  
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(df['description'], df['medical_specialty'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf = MultinomialNB().fit(X_train_tfidf,y_train)

pickle.dump(clf,open('model.pkl','wb'))

pickle.dump(count_vect,open('vectorizer.pkl', 'wb'))



####LINEAR SVC MODEL PREDICTION
from sklearn.model_selection import train_test_split

model3 = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=3)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)

texts = ["Consult for laparoscopic gastric bypass.",
		 "Cerebral Angiogram - moyamoya disease."]
text_features = tfidf.transform(texts)
predictions = model3.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


pickle.dump(model3,open('linearsvc.pkl', 'wb'))


####


####random forest MODEL PREDICTION
from sklearn.model_selection import train_test_split




#pickle.dump(random,open('random.pkl','wb'))







model4 = RandomForestClassifier()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=3)
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)

#texts = ["Consult for laparoscopic gastric bypass.",
	#	 "Cerebral Angiogram - moyamoya disease."]
text_features = tfidf.transform(texts)
predictions = model4.predict(text_features)
#for text, predicted in zip(texts, predictions):
 # print('"{}"'.format(text))
  #print("  - Predicted as: '{}'".format(id_to_category[predicted]))
 # print("")


pickle.dump(model4,open('random.pkl', 'wb'))
pickle.dump(id_to_category,open('id_to_category.pkl','wb'))

####


#print("RESULT@@@")

print(clf.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
print(clf.predict(count_vect.transform(["  Whole body PET scanning."])))

print("end_end_end")
