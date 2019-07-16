import mxnet as mx
import numpy as np
from sklearn.preprocessing import normalize


class FaceRecognitionWithArcFace:
    def __init__(self, ctx=mx.cpu(), model_path='./models/fr_model_r100_ii/model'):
        prefix = model_path
        epoch = 0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        # noinspection PyTypeChecker
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding
