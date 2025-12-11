import tensorflow as tf
from tensorflow import keras

from embeddings import EmbeddingLayer
from transformer_encoder import TransformerEncoder

class RecursiveTransformer(keras.Model):
    def __init__(self,
                vocab_size,
                d_model,
                max_seq_length,
                num_layers,
                num_heads,
                ff_dim,
                dropout_rate = 0.1,
                deep_rec_cycles = 3,
                deep_sup_steps = 4,
                act_loss_weight = 0.1,
                num_l_steps = 6,
                halt_exploration_prob = 0.1,
                halt_max_steps = 16,
                step_penalty_weight = 0.01,
                **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.deep_rec_cycles = deep_rec_cycles
        self.deep_sup_steps = deep_sup_steps
        self.act_loss_weight = act_loss_weight
        self.num_l_steps = num_l_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.halt_max_steps = halt_max_steps
        self.step_penalty_weight = step_penalty_weight

        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_length, dropout_rate)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, ff_dim, num_layers, dropout_rate)

        self.layernorm = keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout = keras.layers.Dropout(dropout_rate)

        self.output_head = keras.layers.Dense(vocab_size)

        self.q_head = keras.layers.Dense(1)
        self.y0 = self.add_weight(shape = (1, 1, d_model), initializer = "zeros", trainable = True, name = "y0")
        self.z0 = self.add_weight(shape = (1, 1, d_model), initializer = "zeros", trainable = True, name = "z0")

    def f_L(self, embedding, y, z, training):
        z_input = z + y + embedding
        z_hidden = self.transformer_encoder(z_input, training = training)
        z_hidden = self.layernorm(z_hidden)
        z_hidden = self.dropout(z_hidden, training = training)
        return z_hidden

    def f_H(self, y, z, training):
        y_input = y + z
        y_hidden = self.transformer_encoder(y_input, training = training)
        y_hidden = self.layernorm(y_hidden)
        y_hidden = self.dropout(y_hidden, training=training)
        return y_hidden

    def full_recursion(self, embedding, y, z, training):
        # n times f_L
        for _ in range(self.num_l_steps):
            z = self.f_L(embedding, y, z, training = training)
        # 1 times f_H
        y = self.f_H(y, z, training = training)
        return y, z

    def recursive_reasoning(self, embedding, y, z, training = False):
        # for i in range(max(self.deep_rec_cycles - 1, 0)):
        #     y_tmp, z_tmp = self.full_recursion(embedding, y, z, training = False)
        #     y = tf.stop_gradient(y_tmp)
        #     z = tf.stop_gradient(z_tmp)
        y = tf.stop_gradient(y)
        z = tf.stop_gradient(z)

        y, z = self.full_recursion(embedding, y, z, training = training)

        logits = self.output_head(y)
        q_logit = tf.squeeze(self.q_head(y[:, 0, :]), axis=-1)
        return y, z, logits, q_logit

    def train_step(self, data):
        inputs, labels = data

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        with tf.GradientTape() as tape:
            # Initialize y and z inside GradientTape so gradients can flow to y0 and z0
            y = tf.tile(self.y0, [batch_size, seq_len, 1])
            z = tf.tile(self.z0, [batch_size, seq_len, 1])
            
            # Compute embedding inside GradientTape so gradients flow to embedding weights
            embedding = self.embedding(inputs, training = True)
            
            total_ce = 0.0
            total_act = 0.0
            total_step_penalty = 0.0

            # Use more lenient halting during training to encourage more refinement steps
            train_halt_prob = self.halt_exploration_prob * 0.5  # More lenient during training

            for i in range(self.deep_sup_steps):
                # run recursive reasoning once
                y, z, logits, q_logit = self.recursive_reasoning(embedding, y, z, training = True)

                # get ce loss
                ce = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)
                total_ce += tf.reduce_mean(ce)

                # ACT logic - more lenient during training
                preds = tf.argmax(logits, output_type = labels.dtype, axis = -1)
                seq_correct = tf.reduce_mean(tf.cast(preds == labels, tf.float32), axis=-1)

                # Binary cross-entropy for halting decision (want to halt when correct)
                act_loss = keras.losses.binary_crossentropy(seq_correct, q_logit, from_logits = True)
                total_act += tf.reduce_mean(act_loss)

                # Step penalty: encourage taking more steps (penalize halting too early)
                # Higher penalty for early steps, lower for later steps
                step_penalty = self.step_penalty_weight * tf.reduce_mean(-q_logit) / (i + 1)
                total_step_penalty += step_penalty

                # remove latent state AFTER computing loss
                # Don't stop gradient on first iteration to ensure gradients flow to y0/z0
                # Stop gradient on subsequent iterations to prevent gradient flow between supervision steps
                if i > 0:
                    y = tf.stop_gradient(y)
                    z = tf.stop_gradient(z)

            ce_mean = total_ce / tf.cast(self.deep_sup_steps, tf.float32)
            act_mean = total_act / tf.cast(self.deep_sup_steps, tf.float32)
            step_penalty_mean = total_step_penalty / tf.cast(self.deep_sup_steps, tf.float32)

            loss = ce_mean + self.act_loss_weight * act_mean + step_penalty_mean

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def call(self, inputs, training = False):
        embeddings = self.embedding(inputs, training = training)
        
        batch_sz = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        y = tf.tile(self.y0, [batch_sz, seq_len, 1])
        z = tf.tile(self.z0, [batch_sz, seq_len, 1])
        
        if training:
            y, z, logits, q_logit = self.recursive_reasoning(embeddings, y, z, training = True)
            return logits
        else:
            # Use tf.while_loop for graph compatibility
            def condition(steps, halted, y, z, prev_logits):
                # Continue if not all halted and under max steps
                not_all_halted = tf.logical_not(tf.reduce_all(halted))
                under_max_steps = tf.less(steps, self.halt_max_steps)
                return tf.logical_and(not_all_halted, under_max_steps)
            
            def body(steps, halted, y, z, prev_logits):
                y_new, z_new, logits, q_logit = self.recursive_reasoning(embeddings, y, z, training = False)
                halt_prob = tf.sigmoid(q_logit)
                halted_new = tf.logical_or(halted, halt_prob > self.halt_exploration_prob)
                return [steps + 1, halted_new, y_new, z_new, logits]
            
            # Initialize loop variables
            steps = tf.constant(0)
            halted = tf.zeros((batch_sz,), dtype=tf.bool)
            # Initialize prev_logits with zeros of correct shape
            vocab_size = self.vocab_size
            prev_logits = tf.zeros((batch_sz, seq_len, vocab_size), dtype=tf.float32)
            
            # Run the loop
            _, _, _, _, final_logits = tf.while_loop(
                condition, body,
                [steps, halted, y, z, prev_logits],
                maximum_iterations=self.halt_max_steps
            )
            
            return final_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'deep_rec_cycles': self.deep_rec_cycles,
            'deep_sup_steps': self.deep_sup_steps,
            'act_loss_weight': self.act_loss_weight,
            'num_l_steps': self.num_l_steps,
            'halt_exploration_prob': self.halt_exploration_prob,
            'halt_max_steps': self.halt_max_steps,
            'step_penalty_weight': self.step_penalty_weight,
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

