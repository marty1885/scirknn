import scirknn

clf = scirknn.MLPClassifer("test.rknn")
pred = clf.predict([0, 0])
print(pred)
