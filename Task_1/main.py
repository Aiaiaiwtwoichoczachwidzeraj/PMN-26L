import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

# 1. Ładowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Klasyfikacja KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 4. Metryki
print(f"Indeks: 119 118, Grupa: 3")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nRaport:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Wizualizacja t-SNE (2D)
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))
plt.title("Wizualizacja t-SNE zbioru Iris")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
