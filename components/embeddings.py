import tensorflow as tf
from tensorflow import keras


class EmbeddingLayer(keras.layers.Layer):    
    def __init__(self, vocab_size, d_model, max_seq_length, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        
        # Token embedding
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=True,
            name='token_embedding'
        )
        
        # Learned positional embeddings
        self.position_embedding = keras.layers.Embedding(
            input_dim=max_seq_length,
            output_dim=d_model,
            name='position_embedding'
        )
        
        # Layer normalization
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout for regularization
        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        seq_length = tf.shape(inputs)[1]
        
        # Get token embeddings
        # Shape: (batch_size, seq_length, d_model)
        token_embeds = self.token_embedding(inputs)
        
        # Get positional embeddings
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = tf.expand_dims(positions, 0)

        batch_size = tf.shape(inputs)[0]
        positions = tf.tile(positions, [batch_size, 1])

        # Shape: (batch_size, seq_length, d_model)
        position_embeds = self.position_embedding(positions)
        
        # Combine token and positional embeddings
        embeddings = token_embeds + position_embeds
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length,
            'dropout_rate': self.dropout_rate,
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)