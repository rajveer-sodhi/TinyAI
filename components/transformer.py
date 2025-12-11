import tensorflow as tf
from tensorflow import keras

from embeddings import EmbeddingLayer
from transformer_encoder import TransformerEncoder

class Transformer(keras.Model):
    def __init__(self,
                input_vocab_size,
                ans_vocab_size,
                d_model,
                max_seq_length,
                num_layers,
                num_heads,
                ff_dim,
                dropout_rate=0.1,
                **kwargs):
        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate


        self.embedding = EmbeddingLayer(input_vocab_size, d_model, max_seq_length, dropout_rate)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, ff_dim, num_layers, dropout_rate)

        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout_rate)

        self.output_head = keras.layers.Dense(ans_vocab_size)

    def call(self, inputs, training=False):
        embeddings = self.embedding(inputs, training=training)
        encoded = self.transformer_encoder(embeddings, training=training)

        outputs = self.layernorm(encoded)
        outputs = self.dropout(outputs, training=training)
        outputs = self.output_head(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_vocab_size': self.input_vocab_size,
            'ans_vocab_size': self.ans_vocab_size,
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

