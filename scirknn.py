from rknn.api import RKNN
import json
import numpy as np

class MLPRegresser:
    def __init__(self, model_path):
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()

        with open(model_path + '.json', 'r') as f:
            self.meta = json.load(f)

    def predict(self, x):
        x = np.array(x)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        if not self._shape_check(x.shape):
            accept_shape = self.meta['input_shape']
            raise ValueError(f'Input shape {x.shape} is not acceptible to the model. Accept shape: {accept_shape}')
        return self.rknn.inference(inputs=[x])

    def __del__(self):
        self.rknn.release()

    def _shape_check(self, x):
        same = x == self.meta['input_shape']
        if same: return True

        if self.meta['input_shape'][0] == -1 || self.meta['input_shape'][1] == None:
            return self.meta['input_shape'][1:] == x[1:]


class MLPClassifer:
    def __init__(self, model_path):
        self.regresser = MLPRegresser(model_path)
        self.classes = self.regresser.meta['classes']

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes[np.argmax(probs, axis=1))]

    def predict_proba(self, x):
        return self.regresser.predict(x)
