import tensorflow as tf
from tensorflow import keras

from transformer_block import TransformerBlock

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, num_layers, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.transformer_blocks = [TransformerBlock(embed_size, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]

    def call(self, inputs, training=False):
        for transformer_block in self.transformer_blocks:
            inputs = transformer_block(inputs, training=training)

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)