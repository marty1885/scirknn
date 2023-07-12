from rknnlite.api import RKNNLite
import json
import numpy as np

class MLPRegresser:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        with open(model_path + '.json', 'r') as f:
            self.meta = json.load(f)

    def predict(self, x):
        x = np.array(x, dtype=np.float32)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if not self._shape_check(x.shape):
            accept_shape = self.meta['input_shape']
            raise ValueError(f'Input shape {x.shape} is not acceptible to the model. Accept shape: {accept_shape}')
        return np.array(self.rknn.inference(inputs=[x]))[0]

    def __del__(self):
        self.rknn.release()

    def _shape_check(self, x):
        accept_shape = tuple(self.meta['input_shape'])
        x = tuple(x)
        same = x == accept_shape
        if same: return True

        if accept_shape[0] == -1 or accept_shape[0] == None:
            return accept_shape[1:] == x[1:]


class MLPClassifer:
    def __init__(self, model_path):
        self.regresser = MLPRegresser(model_path)
        self.classes = self.regresser.meta['classes']

    def predict(self, x):
        probs = self.predict_proba(x)
        pred = np.argmax(probs, axis=1)
        return [self.classes[n] for n in pred]

    def predict_proba(self, x):
        prob = self.regresser.predict(x)
        if prob.shape[1] == 1:
            return np.concatenate([1 - prob, prob], axis=1)
        return prob
