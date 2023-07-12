import sklearn2rknn
from sklearn.neural_network import MLPClassifier

x = [[0., 0.], [1., 1.], [1., 0.], [0., 1.]]
y = [0, 0, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x, y)

sklearn2rknn.convert(clf, 'xor.rknn', 'rk3588', batch_size=1)
