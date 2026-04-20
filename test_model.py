import tensorflow as tf
import numpy as np

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x):
        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_": self.lambda_})
        return config

# load model
model = tf.keras.models.load_model(
    "brain_tumor_model.keras",
    custom_objects={'GradientReversalLayer': GradientReversalLayer}
)

# dummy input
x = np.random.rand(1, 224, 224, 3)

preds = model.predict(x)

print("PRED SHAPES:")
for p in preds:
    print(np.array(p).shape)