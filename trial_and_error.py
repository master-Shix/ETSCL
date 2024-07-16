from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

true_labels = [0, 1, 0, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 2, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1, 1, 0, 1, 2, 0, 0, 2, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0]

predicted_labels = [0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0]


# Calculate Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy:.4f}')

# Calculate Kappa
kappa = cohen_kappa_score(true_labels, predicted_labels)
print(f'Kappa: {kappa:.4f}')

# Calculate F1 Score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f'F1 Score: {f1:.4f}')
