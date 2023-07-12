import scirknn

clf = scirknn.MLPClassifier('xor.rknn')

x = [1, 0]
y = clf.predict(x)
print(y)
