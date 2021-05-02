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

from sklearn.feature_selection import chi2
import numpy as np

N = 2
#for medical_specialty, category_id in sorted(category_to_id.items()):
 # features_chi2 = chi2(features, labels == category_id)
  #indices = np.argsort(features_chi2[0])
  #feature_names = np.array(tfidf.get_feature_names())[indices]
  #unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  #bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  #print("# '{}':".format(medical_specialty))
  #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  #print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  
  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['medical_specialty'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, y_train)
clf = MultinomialNB().fit(X_train_tfidf,y_train)

pickle.dump(clf,open('model.pkl','wb'))

pickle.dump(count_vect,open('vectorizer.pkl', 'wb'))




#print("RESULT@@@")

print(clf.predict(count_vect.transform([" Nasal endoscopy and partial rhinectomy due to squamous cell carcinoma, left nasal cavity."])))
print(clf.predict(count_vect.transform(["  Whole body PET scanning."])))

print("end_end_end")
#df[df['description'] == "Fertile male with completed family.  Elective male sterilization via bilateral vasectomy."]













