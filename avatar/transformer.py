from constants import TRANSFORMER_MODEL

from dataclasses import dataclass

import tensorflow as tf


class SharedEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, model_dim, pad_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.pad_id = pad_id

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.vocab_size, self.model_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.model_dim ** -0.5),
            trainable=True,
        )
        self.weight.assign(
            tf.tensor_scatter_nd_update(
                self.weight,
                [[self.pad_id]],
                tf.zeros((1, self.model_dim), dtype=self.weight.dtype),
            )
        )

    def call(self, token_ids):
        return tf.gather(self.weight, token_ids)

    def logits(self, features):
        return tf.matmul(tf.cast(features, self.weight.dtype), self.weight, transpose_b=True)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim, num_heads, dropout, attention_dropout, activation_dropout):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attention_dropout,
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn1 = tf.keras.layers.Dense(ffn_dim, activation="relu")
        self.ffn_dropout = tf.keras.layers.Dropout(activation_dropout)
        self.ffn2 = tf.keras.layers.Dense(model_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, mask, training=False):
        attn = self.self_attn(x, x, attention_mask=mask, training=training)
        x = self.norm1(x + self.dropout1(attn, training=training))
        ffn = self.ffn2(self.ffn_dropout(self.ffn1(x), training=training))
        x = self.norm2(x + self.dropout2(ffn, training=training))
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim, num_heads, dropout, attention_dropout, activation_dropout):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attention_dropout,
        )
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attention_dropout,
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn1 = tf.keras.layers.Dense(ffn_dim, activation="relu")
        self.ffn_dropout = tf.keras.layers.Dropout(activation_dropout)
        self.ffn2 = tf.keras.layers.Dense(model_dim)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, encoder_outputs, self_mask, cross_mask, training=False):
        attn = self.self_attn(x, x, attention_mask=self_mask, training=training)
        x = self.norm1(x + self.dropout1(attn, training=training))
        cross = self.cross_attn(x, encoder_outputs, attention_mask=cross_mask, training=training)
        x = self.norm2(x + self.dropout2(cross, training=training))
        ffn = self.ffn2(self.ffn_dropout(self.ffn1(x), training=training))
        x = self.norm3(x + self.dropout3(ffn, training=training))
        return x


@dataclass
class TransformerConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    model_dim: int = TRANSFORMER_MODEL["model_dim"]
    ffn_dim: int = TRANSFORMER_MODEL["ffn_dim"]
    num_heads: int = TRANSFORMER_MODEL["num_heads"]
    num_layers: int = TRANSFORMER_MODEL["num_layers"]
    dropout: float = TRANSFORMER_MODEL["dropout"]
    attention_dropout: float = TRANSFORMER_MODEL["attention_dropout"]
    activation_dropout: float = TRANSFORMER_MODEL["activation_dropout"]
    max_source_length: int = 512
    max_target_length: int = 512


class Transformer(tf.keras.Model):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = SharedEmbedding(config.vocab_size, config.model_dim, config.pad_id)
        self.source_positions = tf.keras.layers.Embedding(config.max_source_length, config.model_dim)
        self.target_positions = tf.keras.layers.Embedding(config.max_target_length, config.model_dim)
        self.encoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.decoder_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.encoder_dropout = tf.keras.layers.Dropout(config.dropout)
        self.decoder_dropout = tf.keras.layers.Dropout(config.dropout)
        self.encoder_layers = [
            EncoderLayer(
                config.model_dim,
                config.ffn_dim,
                config.num_heads,
                config.dropout,
                config.attention_dropout,
                config.activation_dropout,
            )
            for _ in range(config.num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(
                config.model_dim,
                config.ffn_dim,
                config.num_heads,
                config.dropout,
                config.attention_dropout,
                config.activation_dropout,
            )
            for _ in range(config.num_layers)
        ]

    def source_mask(self, token_ids):
        return tf.not_equal(token_ids, self.config.pad_id)

    def target_mask(self, token_ids):
        return tf.not_equal(token_ids, self.config.pad_id)

    def encoder_attention_mask(self, token_ids):
        mask = self.source_mask(token_ids)
        return tf.logical_and(mask[:, :, None], mask[:, None, :])

    def decoder_self_attention_mask(self, token_ids):
        mask = self.target_mask(token_ids)
        seq_len = tf.shape(token_ids)[1]
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        causal = causal[None, :, :]
        return tf.logical_and(tf.logical_and(mask[:, :, None], mask[:, None, :]), causal)

    def decoder_cross_attention_mask(self, decoder_ids, encoder_ids):
        decoder_mask = self.target_mask(decoder_ids)
        encoder_mask = self.source_mask(encoder_ids)
        return tf.logical_and(decoder_mask[:, :, None], encoder_mask[:, None, :])

    def add_positions(self, token_ids, position_embedding):
        seq_len = tf.shape(token_ids)[1]
        position_ids = tf.range(seq_len)[None, :]
        return self.embedding(token_ids) + position_embedding(position_ids)

    def encode(self, encoder_inputs, training=False):
        x = self.add_positions(encoder_inputs, self.source_positions)
        x = self.encoder_dropout(self.encoder_norm(x), training=training)
        mask = self.encoder_attention_mask(encoder_inputs)
        source_mask = self.source_mask(encoder_inputs)
        for layer in self.encoder_layers:
            x = layer(x, mask, training=training)
            x = x * tf.cast(source_mask[:, :, None], x.dtype)
        return x

    def decode(self, decoder_inputs, encoder_inputs, encoder_outputs, training=False):
        x = self.add_positions(decoder_inputs, self.target_positions)
        x = self.decoder_dropout(self.decoder_norm(x), training=training)
        self_mask = self.decoder_self_attention_mask(decoder_inputs)
        cross_mask = self.decoder_cross_attention_mask(decoder_inputs, encoder_inputs)
        target_mask = self.target_mask(decoder_inputs)
        for layer in self.decoder_layers:
            x = layer(x, encoder_outputs, self_mask, cross_mask, training=training)
            x = x * tf.cast(target_mask[:, :, None], x.dtype)
        return x

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs = self.encode(encoder_inputs, training=training)
        decoder_outputs = self.decode(decoder_inputs, encoder_inputs, encoder_outputs, training=training)
        return self.embedding.logits(decoder_outputs)

    def warmup(self):
        encoder_inputs = tf.constant([[self.config.eos_id]], dtype=tf.int32)
        decoder_inputs = tf.constant([[self.config.bos_id]], dtype=tf.int32)
        self((encoder_inputs, decoder_inputs), training=False)

    def greedy_decode(self, encoder_inputs, max_length):
        encoder_outputs = self.encode(encoder_inputs, training=False)
        tokens = tf.fill([tf.shape(encoder_inputs)[0], 1], self.config.bos_id)
        finished = tf.zeros([tf.shape(encoder_inputs)[0]], dtype=tf.bool)
        outputs = tf.TensorArray(tf.int32, size=max_length)
        max_length = tf.convert_to_tensor(max_length, dtype=tf.int32)
        t = tf.constant(0, dtype=tf.int32)

        def cond(t, tokens, finished, outputs):
            return t < max_length

        def body(t, tokens, finished, outputs):
            decoder_outputs = self.decode(tokens, encoder_inputs, encoder_outputs, training=False)
            logits = self.embedding.logits(decoder_outputs[:, -1, :])
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            next_token = tf.where(finished, self.config.eos_id, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.config.eos_id))
            tokens = tf.concat([tokens, next_token[:, None]], axis=1)
            return t + 1, tokens, finished, outputs.write(t, next_token)

        _, _, _, outputs = tf.while_loop(
            cond,
            body,
            [t, tokens, finished, outputs],
            parallel_iterations=1,
        )
        return tf.transpose(outputs.stack(), [1, 0])

    def beam_decode(self, encoder_input, max_length, beam_size=5, length_penalty=0.6):
        if len(encoder_input.shape) == 1:
            encoder_input = encoder_input[None, :]
        encoder_outputs = self.encode(encoder_input, training=False)
        beams = [([self.config.bos_id], 0.0, False)]
        for _ in range(max_length):
            candidates = []
            for tokens, score, done in beams:
                if done:
                    candidates.append((tokens, score, done))
                    continue
                decoder_inputs = tf.constant([tokens], dtype=tf.int32)
                decoder_outputs = self.decode(decoder_inputs, encoder_input, encoder_outputs, training=False)
                logits = self.embedding.logits(decoder_outputs[:, -1, :])[0]
                log_probs = tf.nn.log_softmax(logits).numpy()
                top_ids = log_probs.argsort()[-beam_size:][::-1]
                for token_id in top_ids:
                    token_id = int(token_id)
                    candidates.append(
                        (
                            tokens + [token_id],
                            score + float(log_probs[token_id]),
                            token_id == self.config.eos_id,
                        )
                    )
            beams = sorted(
                candidates,
                key=lambda item: item[1] / ((len(item[0]) ** length_penalty) or 1.0),
                reverse=True,
            )[:beam_size]
            if all(done for _, _, done in beams):
                break
        best_tokens = max(
            beams,
            key=lambda item: item[1] / ((len(item[0]) ** length_penalty) or 1.0),
        )[0]
        return best_tokens[1:]
