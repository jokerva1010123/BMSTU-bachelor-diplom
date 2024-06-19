import cudf
from cuml import SVC
from cuml.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Convert data to cuDF DataFrame
X_cudf = cudf.DataFrame(X)
y_cudf = cudf.Series(y)

# Split the data into training and testing sets
X_train, X_test = X_cudf[:800], X_cudf[800:]
y_train, y_test = y_cudf[:800], y_cudf[800:]

# Create and train SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)

# Evaluate the model
accuracy = (predictions == y_test).sum() / len(y_test)
print(f"Accuracy: {accuracy}")
