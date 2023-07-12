from sklearn.neural_network import MLPClassifier, MLPRegressor
import sklearn2rknn
import os

def test_classifier():
    x = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    y = [0, 1, 2, 3]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1)
    clf.fit(x, y)

    sklearn2rknn.convert(clf, 'test.rknn', 'rk3588')
    # Check test.rknn and test.rknn.json exist
    assert os.path.exists('test.rknn')
    assert os.path.exists('test.rknn.json')

    # test with quantization on
    sklearn2rknn.convert(clf, 'test.rknn', 'rk3588', quantization=True, example_input=x)
    assert os.path.exists('test.rknn')
    assert os.path.exists('test.rknn.json')

def test_regresser():
    x = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    y = [0, 1, 2, 3]

    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1)
    clf.fit(x, y)

    sklearn2rknn.convert(clf, 'test.rknn', 'rk3588')

    # Check test.rknn and test.rknn.json exist
    assert os.path.exists('test.rknn')
    assert os.path.exists('test.rknn.json')


