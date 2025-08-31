## KNN Classifier on Iris Dataset

### 📋 Project Overview
Implementation of a K-Nearest Neighbors (KNN) classifier to classify iris flowers into three species (setosa, versicolor, virginica) based on their sepal and petal measurements.

### 🎯 Key Features
- Data loading and exploration of the famous Iris dataset
- KNN model implementation with customizable neighbors parameter
- Model evaluation using accuracy score and confusion matrix
- Visualization of results with heatmaps
- Hyperparameter tuning to find optimal k-value

### 📊 Dataset Information
The Iris dataset contains 150 samples with 4 features each:
- Sepal length (cm)
- Sepal width (cm) 
- Petal length (cm)
- Petal width (cm)

Three target classes:
- Setosa
- Versicolor
- Virginica

### 🛠️ Technologies Used
- Python 3.x
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

### 🚀 Code Explanation

```python
# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier implementation
from sklearn.model_selection import train_test_split  # Data splitting utility
from sklearn.datasets import load_iris  # Load iris dataset
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluation metrics
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization

# Load the iris dataset
iris = load_iris()
x = iris.data  # Feature matrix (150 samples × 4 features)
y = iris.target  # Target vector (150 labels)

# Explore dataset structure
print(x.shape)  # Output: (150, 4) - 150 samples, 4 features
print(y.shape)  # Output: (150,) - 150 target labels
print(iris.feature_names)  # Display feature names
print(iris.target_names)  # Display class names

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Initialize KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)  # Train the model

# Make predictions on test data
y_pred = knn.predict(x_test)

# Evaluate model performance
print(confusion_matrix(y_test, y_pred))  # Display confusion matrix
print(accuracy_score(y_test, y_pred)*100)  # Calculate accuracy percentage

# Make predictions on custom samples
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]  # Custom feature vectors
preds = knn.predict(sample)  # Predict classes
preds_species = [iris.target_names[p] for p in preds]  # Convert to species names

# Calculate training and testing accuracy
print(knn.score(x_train, y_train))  # Training accuracy
print(knn.score(x_test, y_test))  # Testing accuracy

# Hyperparameter tuning - test different k values
for j in range(5, 120, 5):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(j, knn.score(x_test, y_test))  # Print k value and corresponding accuracy

# Create confusion matrix visualization with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

### 📈 Results
The model achieves approximately 93.33% accuracy with k=3 neighbors and reaches 100% accuracy with k=10, 15, and 20 neighbors on the test set. The confusion matrix visualization provides clear insights into classification performance across all three iris species.

---
