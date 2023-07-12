# scirknn

Unholy but necessary code that converts scikit-learn's MLP classifer/regresser into RKNN2 files in order to run on Rockchip NPUs. Because `rknn-toolkit2` does not support the operations `sklearn-onnx` spits out. So you can put your busness logic on the edge (or other things you want to do). Also works around a ton of stuff that ONNX and rknn-toolkit2 expects the other party to solve.

## How it works

* Grabs weights from the sklearn model
* Build ONNX graph with nodes that RKNN accepts
* Move operations RKNN does not support into Python (these are lite operations)
* Convert ONNX into RKNN
* Provides a lite Python wrapper with the same API as scikit-learn if you need it

### Example

```python
from sklearn.neural_network import MLPClassifier
import sklearn2rknn # Convers scikit models into RKNN 

# obtain a MLP model 
x = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
y = [0, 1, 2, 3]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 32), random_state=1)
clf.fit(x, y)

# Convert and save the model targeting the RK3588 chip. If successful, this step produces
# two files. `example.rknn` and `example.rknn.json`. The latter is metadata that is used
# by wrappers later
# !!IMPORTANT: You WANT to enable quantization. As of rknn-toolkit2 1.5.0. It
# Only uses the NPU for fp16 operations and when all input tensor size is a multiple
# of 32 elements (if I am not mistaken)
sklearn2rknn.convert(clf, "example.rknn", "rk3588", quantization=True, example_input=x)

```
Or, use the commandline.

```bash
python -m sklearn2rknn model.pkl model.rknn --quantization --example_input /path/to/data.npy

```

Now, copy `example.rknn` and `example.rknn.json` to your development board.

```python
import scirknn # Wrapper to provide easy to use API
model = scirknn.MLPClassifer("example.krnn")
pred = model.predict([0, 0])
print(pred) # [0]
```

## Install

* Install [rknn-toolkit2][rknn-toolkit-whl] from the official `.whl` file
* Install the remaining dependencies `pip install -r requirments`
  * only needed for model conversion

[rknn-toolkit-whl]: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/packages

## Running the examples

The examples are stored in the `examples` folder. Every example has 2 files. `train.py` which trains a MLP model and coverts it into RKNN format. By default the training scripts converts for RK3588. And `infer.py` which shold run on the terget device. To run them, invoke them as a module.

```bash
> python -m example.xor.train
```

This prduces `xor.rknn` and `xor.rknn.json` in the current working directory. Copy them to your target device. Then run the following command to infer

```bash
> python -m example.xor.infer
I RKNN: [14:45:42.681] RKNN Runtime Information: librknnrt version: 1.5.0 (e6fe0c678@2023-05-25T08:09:20)
...
[0]
```
The final line is the output!
