#  Import Required Libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
import string
import re
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore

#  Load the Dataset
df = pd.read_csv('Email spam detection.py\\spamhamdata.csv', encoding='latin-1', on_bad_lines='skip', delimiter='\t', header=None)#[['v1', 'v2']]
df.columns = ['label', 'message']  # Renaming columns
print(df)

#  Preprocess the Text
def clean_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()  # Convert to lowercase
        # Add other cleaning steps here (e.g., remove punctuation, numbers, etc.)
    else:
        text = ""  # or use 'text = str(text)' to convert non-strings to empty strings
    return text

# Apply text cleaning
df['cleaned_message'] = df['message'].fillna('').apply(clean_text)

# Convert 'ham' to 0 and 'spam' to 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

#  Remove rows with missing target labels
df = df.dropna(subset=['label_num'])

#  Vectorize Text Data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label_num'].values

#  Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

#  Evaluate the Model
y_pred = model.predict(X)

print("🔹 Accuracy:", accuracy_score(y, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y, y_pred))

#  Predict Custom Email
def predict_email(text):
    """
    Predict whether the input email text is 'Spam' or 'Ham'.
    - Clean the input text
    - Vectorize the cleaned text
    - Predict using the trained model
    """
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Test Predictions
print("\n🔎 Prediction 1:", predict_email("Congratulations! You’ve won a free iPhone. Click here to claim now!"))
print("🔎 Prediction 2:", predict_email("Hey, I’ll see you at the meeting tomorrow at 10."))
