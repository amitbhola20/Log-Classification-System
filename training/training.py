import pandas as pd

df = pd.read_csv(r'C:\Users\hp\Log-Classification-System\training\dataset\synthetic_logs.csv')
#print(df.head())
print(df.source.unique())
print(df.target_label.unique())

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for log messages
embeddings = model.encode(df['log_message'].tolist())
print(embeddings[:2])

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=1, metric='cosine')
clusters = dbscan.fit_predict(embeddings)

# Add cluster label to the DataFrame
df['cluster'] = clusters
print(df.head())
print(df[df.cluster==5])

cluster_counts = df['cluster'].value_counts()
large_clusters = cluster_counts[cluster_counts > 10].index

for cluster in large_clusters:
    print(f"Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['log_message'].head(5).to_string(index=False))
    print()

    import re


def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message):
            return label
    return None


print(classify_with_regex("User User123 logged in."))
print(classify_with_regex("System reboot initiated by user User179."))
print(classify_with_regex("Hey you, chill bro"))

# Apply regex classification
df['regex_label'] = df['log_message'].apply(classify_with_regex)
print(df[df['regex_label'].notnull()])
print(df[df['regex_label'].isnull()].head(5))

df_non_regex = df[df['regex_label'].isnull()].copy()
print(df_non_regex.shape)

df_legacy = df_non_regex[df_non_regex.source=="LegacyCRM"]
print(df_legacy)

df_non_legacy = df_non_regex[df_non_regex.source!="LegacyCRM"]
print(df_non_legacy)

print(df_non_legacy.shape)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
embeddings_filtered = model.encode(df_non_legacy['log_message'].tolist())

len(embeddings_filtered)

X = embeddings_filtered
y = df_non_legacy['target_label'].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

import joblib
joblib.dump(clf, r'C:\Users\hp\Log-Classification-System\models\log_classifier.joblib')