from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn2rknn

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=500, alpha=0.0001, solver='adam')
clf.fit(X_train, y_train)

def accuracy(pred, y):
    return sum(pred == y) / len(y)

pred = clf.predict(X_test)
print(f"Accuracy: {accuracy(pred, y_test)}")

sklearn2rknn.convert(clf, "digits.rknn", 'rk3588')
