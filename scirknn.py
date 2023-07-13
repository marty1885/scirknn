from rknnlite.api import RKNNLite
import json
import numpy as np

class MLPRegressor:
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

        self._shape_check(x.shape)

        single_infer_out_shape = self.meta['output_shape']
        infer_batch_size = self.meta['output_shape'][0]
        assert len(x) % infer_batch_size == 0 # Should be guarenteed by the shape check above
        nruns = len(x) // infer_batch_size
        out_shape = single_infer_out_shape
        out_shape[0] = out_shape[0]*nruns

        out = np.zeros(out_shape, dtype=np.float32)
        for i in range(0, len(x), infer_batch_size):
            xp = x[i:i+infer_batch_size]
            out[i:i+infer_batch_size] = np.array(self.rknn.inference(inputs=[xp]), dtype=np.float32)
        return out

    def __del__(self):
        self.rknn.release()

    def _shape_check(self, x) -> None:
        accept_shape = tuple(self.meta['input_shape'])
        x = tuple(x)
        if x == accept_shape: return

        batch_size = accept_shape[0]
        if batch_size != -1 and batch_size is not None and x[0] % batch_size != 0:
            raise ValueError(f"Input has batch size of {x[0]}. Which is not a multiple of {batch_size}, the model's expected size.")
        if accept_shape[1:] != x[1:]:
            raise ValueError(f'Input shape {x} mismatch with model. Expecting shape: {accept_shape}')


class MLPClassifier:
    def __init__(self, model_path):
        self.regressor = MLPRegressor(model_path)
        self.classes = self.regressor.meta['classes']

    def predict(self, x):
        probs = self.predict_proba(x)
        pred = np.argmax(probs, axis=1)
        return [self.classes[n] for n in pred]

    def predict_proba(self, x):
        prob = self.regressor.predict(x)
        if prob.shape[1] == 1:
            return np.concatenate([1 - prob, prob], axis=1)
        return prob
