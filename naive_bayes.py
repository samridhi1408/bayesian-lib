import pandas as pd
import numpy as np
from foundation.multiGaussian import computeLOG 

# 1. Load data
df = pd.read_csv('spambase.csv', header=None)
label_col = df.columns[-1]

# 2. TRAINING (Do this once outside the loop)
spam_df = df[df[label_col] == 1]
not_spam_df = df[df[label_col] == 0]

# Calculate stats for Spam
spam_means = spam_df.iloc[:, :-1].mean().values
spam_vars = spam_df.iloc[:, :-1].var().clip(lower=1e-6).values
cov_spam = np.diag(spam_vars)
log_prior_spam = np.log(len(spam_df) / len(df))

# Calculate stats for Not-Spam
not_spam_means = not_spam_df.iloc[:, :-1].mean().values
not_spam_vars = not_spam_df.iloc[:, :-1].var().clip(lower=1e-6).values
cov_not_spam = np.diag(not_spam_vars)
log_prior_not_spam = np.log(len(not_spam_df) / len(df))

# 3. PREDICTION LOOP
predictions = []
actual_labels = df[label_col].values

print("Analyzing 4,601 emails...")
for i in range(len(df)):
    # USE 'i' TO GET THE CURRENT EMAIL
    sample = df.iloc[i, :-1].values
    
    # Calculate Log-Scores
    score_spam = log_prior_spam + computeLOG(sample, spam_means, cov_spam)
    score_not_spam = log_prior_not_spam + computeLOG(sample, not_spam_means, cov_not_spam)
    
    if score_spam > score_not_spam:
        predictions.append(1)
    else:
        predictions.append(0)

# 4. FINAL RESULTS
accuracy = np.mean(predictions == actual_labels) * 100
print(f"Final Accuracy: {accuracy:.2f}%")


