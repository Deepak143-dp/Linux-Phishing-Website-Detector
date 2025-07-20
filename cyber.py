import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import tldextract
import joblib


def extract_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    url_length = len(url)
    has_at = int('@' in url)
    has_https = int(parsed.scheme == 'https')
    num_dots = url.count('.')
    subdomain_length = len(ext.subdomain)
    
    return [url_length, has_at, has_https, num_dots, subdomain_length]

df = pd.read_csv("database.csv")

X = df[['url_length', 'has_at', 'has_https', 'num_dots', 'subdomain_length']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump(model, "phishing_model.pkl")

model = joblib.load("phishing_model.pkl")


userURL = input("\nEnter a URL to check that is phishing URL or not : ")

if not userURL.startswith(("http://", "https://")):
    print("Please enter a correct URL that starts with  like https....ect ")
else:
    features = extract_features(userURL)
    prediction = model.predict([features])[0]

    if prediction == 1:
        print(" \n\n  This is look like a phishing website.\n\n")
    else:
        print(" \n\n  This website is real.\n\n")


