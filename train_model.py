import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load data
df = pd.read_csv('news.csv')
df = df[['text', 'label']]

# Split data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_vec, y_train)

# Save files
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained and saved!")

