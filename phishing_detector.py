import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully.")
print("Total emails:", len(data))


# Separate text and labels
X = data["text"]
y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert email text into numbers
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Create machine learning model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)


# Test model
prediction = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)

print("\n==============================")
print("PHISHING EMAIL DETECTION")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")


# Confusion matrix
cm = confusion_matrix(
    y_test,
    prediction,
    labels=["Safe", "Phishing"]
)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

display.plot()

plt.title("Phishing Email Detection - Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()


# Test a new email
print("\n==============================")
print("TEST YOUR OWN EMAIL")
print("==============================")

email = input("Enter email text: ")

email_vector = vectorizer.transform([email])

result = model.predict(email_vector)[0]

print("\nPrediction:", result)

if result == "Phishing":
    print("Warning: This email may be a phishing email.")
else:
    print("This email appears to be safe.")