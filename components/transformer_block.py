import tensorflow as tf
from tensorflow import keras

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_size // num_heads,
            dropout=dropout_rate,
        )

        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dense(embed_size),
        ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        norm_inputs = self.layernorm1(inputs)
        attn_output = self.attention(norm_inputs, norm_inputs)
        attn_output = self.dropout1(attn_output, training=training)
        inputs = inputs + attn_output

        norm_out1 = self.layernorm2(inputs)
        ffn_output = self.ffn(norm_out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        inputs = inputs + ffn_output
        
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)