import pandas as pd
df = pd.read_csv('mtsamples.csv')
#df.head()
col = ['medical_specialty', 'description']
df = df[col]
df.columns = ['medical_specialty', 'description']

df['category_id'] = df['medical_specialty'].factorize()[0]
from io import StringIO
category_id_df = df[['medical_specialty', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'medical_specialty']].values)

df.head()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('medical_specialty').description.count().plot.bar(ylim=0)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
features.shape

from sklearn.feature_selection import chi2
import numpy as np

#N = 2
#for medical_specialty, category_id in sorted(category_to_id.items()):
  #features_chi2 = chi2(features, labels == category_id)
  #indices = np.argsort(features_chi2[0])
  #feature_names = np.array(tfidf.get_feature_names())[indices]
  #unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  #bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  #print("# '{}':".format(medical_specialty))
  #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
 # print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['medical_specialty'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


print("RESULT@@@")

print(clf.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
print(clf.predict(count_vect.transform(["  Whole body PET scanning."])))

#df[df['description'] == "Fertile male with completed family.  Elective male sterilization via bilateral vasectomy."]

print("End REsult")
#TESTING DIFFERENT MODELS
#@#
rom sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])




#PLOTING MODELS GRAPH
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

#MODELS ACCURACY
#cv_df.groupby('model_name').accuracy.mean()

#MODEL LINEARSVC SELECTED
from sklearn.model_selection import train_test_split

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_df.medical_specialty.values, yticklabels=category_id_df.medical_specialty.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#ACTUAL VS PREDICTED DATA

#from IPython.display import display

#for predicted in category_id_df.category_id:
  #for actual in category_id_df.category_id:
    #if predicted != actual and conf_mat[actual, predicted] >= 6:
     # print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
     # display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['medical_specialty', 'description']])
     # print('')
	  
#CHACTEGORIZING DATA MODEL
from sklearn.feature_selection import chi2

#N = 2
#for medical_specialty, category_id in sorted(category_to_id.items()):
#  indices = np.argsort(model.coef_[category_id])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
#  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
#  print("# '{}':".format(medical_specialty))
#  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
#  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))	  
#  
####INPUT DATA & PREDICTION DATA

texts = ["Consult for laparoscopic gastric bypass.",
		 "Cerebral Angiogram - moyamoya disease."]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

#OVERALL ACURACY & REPORT

#clf1 = RandomForestClassifier().fit(X_train_tfidf, y_train)
#print(clf1.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
#print(clf1.predict(count_vect.transform(["  Whole body PET scanning."])))




#clf2 = LogisticRegression().fit(X_train_tfidf, y_train)
#print(clf2.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
#print(clf2.predict(count_vect.transform(["  Whole body PET scanning."])))




#clf3 = LinearSVC().fit(X_train_tfidf, y_train)
#print(clf3.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
#print(clf3.predict(count_vect.transform(["  Whole body PET scanning."])))






#from sklearn import metrics
#print(metrics.classification_report(y_test, y_pred))

