import scirknn
import numpy as np

clf = scirknn.MLPClassifier('xor.rknn')

x = np.array([[1, 1], [0, 1]], dtype=np.float32)
y = clf.predict(x)
print(y)
