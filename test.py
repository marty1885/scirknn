from sklearn.neural_network import MLPClassifier, MLPRegressor
import sklearn2rknn
import os

# Create test_model_store/ if not exist and clean it
if not os.path.exists('test_model_store'):
    os.mkdir('test_model_store')
for file in os.listdir('test_model_store'):
    os.remove(os.path.join('test_model_store', file))

def test_classifier():
    x = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    y = [0, 1, 2, 3]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1)
    clf.fit(x, y)

    sklearn2rknn.convert(clf, 'test_model_store/test.rknn', 'rk3588', batch_size=1)
    assert os.path.exists('test_model_store/test.rknn')
    assert os.path.exists('test_model_store/test.rknn.json')

    # test with quantization on
    sklearn2rknn.convert(clf, 'test_model_store/test_quantized.rknn', 'rk3588', quantization=True, example_input=x)
    assert os.path.exists('test_model_store/test_quantized.rknn')
    assert os.path.exists('test_model_store/test_quantized.rknn.json')

def test_regresser():
    x = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    y = [0, 1, 2, 3]

    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1)
    clf.fit(x, y)

    sklearn2rknn.convert(clf, 'test_model_store/test_regresser.rknn', 'rk3588')

    assert os.path.exists('test_model_store/test_regresser.rknn')
    assert os.path.exists('test_model_store/test_regresser.rknn.json')


