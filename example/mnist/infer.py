from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import scirknn

def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

model = scirknn.MLPClassifier("digits.rknn")
pred = model.predict(x_test)
print("Accuracy: ", accuracy(y_test, pred))
