import sklearn2rknn
from sklearn.neural_network import MLPClassifier

x = [[0., 0.], [1., 1.], [1., 0.], [0., 1.]]
y = [0, 0, 1, 1]

clf = MLPClassifier(solver='adam', activation='tanh', alpha=0.03, hidden_layer_sizes=(5, 5), random_state=1, max_iter=10000)
clf.fit(x, y)

sklearn2rknn.convert(clf, 'xor.rknn', 'rk3588', batch_size=1)
