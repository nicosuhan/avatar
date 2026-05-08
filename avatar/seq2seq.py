from constants import SEQ2SEQ_MODEL

from dataclasses import dataclass

import tensorflow as tf


def uniform_init():
    return tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)


class SharedEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, pad_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pad_id = pad_id

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.vocab_size, self.embed_size),
            initializer=uniform_init(),
            trainable=True,
        )
        self.weight.assign(
            tf.tensor_scatter_nd_update(
                self.weight,
                [[self.pad_id]],
                tf.zeros((1, self.embed_size), dtype=self.weight.dtype),
            )
        )

    def call(self, token_ids):
        return tf.gather(self.weight, token_ids)

    def logits(self, features):
        return tf.matmul(tf.cast(features, self.weight.dtype), self.weight, transpose_b=True)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, decoder_hidden_size, encoder_output_size):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(
            encoder_output_size,
            use_bias=False,
            kernel_initializer=uniform_init(),
        )
        self.output_proj = tf.keras.layers.Dense(
            decoder_hidden_size,
            use_bias=False,
            kernel_initializer=uniform_init(),
        )

    def call(self, hidden, encoder_outputs, encoder_mask):
        query = self.input_proj(hidden)
        scores = tf.reduce_sum(encoder_outputs * query[:, None, :], axis=-1)
        scores = tf.where(encoder_mask, scores, tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype)))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.reduce_sum(encoder_outputs * weights[:, :, None], axis=1)
        output = tf.nn.tanh(self.output_proj(tf.concat([context, hidden], axis=-1)))
        return output, weights


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    embed_size: int = SEQ2SEQ_MODEL["embed_size"]
    encoder_hidden_size: int = SEQ2SEQ_MODEL["encoder_hidden_size"]
    decoder_hidden_size: int = SEQ2SEQ_MODEL["decoder_hidden_size"]
    encoder_layers: int = SEQ2SEQ_MODEL["encoder_layers"]
    decoder_layers: int = SEQ2SEQ_MODEL["decoder_layers"]
    dropout: float = SEQ2SEQ_MODEL["dropout"]


ModelConfig = Seq2SeqConfig


class Seq2Seq(tf.keras.Model):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.embedding = SharedEmbedding(config.vocab_size, config.embed_size, config.pad_id)
        self.encoder_dropout_in = tf.keras.layers.Dropout(config.dropout)
        self.decoder_dropout_in = tf.keras.layers.Dropout(config.dropout)
        self.decoder_dropout_out = tf.keras.layers.Dropout(config.dropout)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                config.encoder_hidden_size,
                return_sequences=True,
                return_state=True,
                kernel_initializer=uniform_init(),
                recurrent_initializer=uniform_init(),
                bias_initializer=uniform_init(),
            ),
            merge_mode="concat",
        )
        self.encoder_hidden_proj = tf.keras.layers.Dense(
            config.decoder_hidden_size,
            kernel_initializer=uniform_init(),
            bias_initializer=uniform_init(),
        )
        self.encoder_cell_proj = tf.keras.layers.Dense(
            config.decoder_hidden_size,
            kernel_initializer=uniform_init(),
            bias_initializer=uniform_init(),
        )
        self.decoder_cell = tf.keras.layers.LSTMCell(
            config.decoder_hidden_size,
            kernel_initializer=uniform_init(),
            recurrent_initializer=uniform_init(),
            bias_initializer=uniform_init(),
        )
        self.attention = AttentionLayer(config.decoder_hidden_size, config.encoder_hidden_size * 2)

    def encode(self, encoder_inputs, training=False):
        encoder_mask = tf.not_equal(encoder_inputs, self.config.pad_id)
        embedded = self.encoder_dropout_in(self.embedding(encoder_inputs), training=training)
        encoder_outputs, f_h, f_c, b_h, b_c = self.encoder(embedded, mask=encoder_mask, training=training)
        hidden = self.encoder_hidden_proj(tf.concat([f_h, b_h], axis=-1))
        cell = self.encoder_cell_proj(tf.concat([f_c, b_c], axis=-1))
        return encoder_outputs, encoder_mask, hidden, cell

    def output_logits(self, features):
        return self.embedding.logits(features)

    def decode_step(self, token, encoder_outputs, encoder_mask, hidden, cell, input_feed, training=False):
        token_emb = self.decoder_dropout_in(self.embedding(token), training=training)
        rnn_input = tf.concat([token_emb, input_feed], axis=-1)
        hidden, [hidden, cell] = self.decoder_cell(rnn_input, [hidden, cell], training=training)
        hidden = self.decoder_dropout_out(hidden, training=training)
        output, weights = self.attention(hidden, encoder_outputs, encoder_mask)
        output = self.decoder_dropout_out(output, training=training)
        logits = self.output_logits(output)
        return logits, hidden, cell, output, weights

    def decode_teacher(self, encoder_outputs, encoder_mask, hidden, cell, decoder_inputs, training=False):
        steps = tf.shape(decoder_inputs)[1]
        batch_size = tf.shape(decoder_inputs)[0]
        input_feed = tf.zeros((batch_size, self.config.decoder_hidden_size), dtype=tf.float32)
        tokens = tf.transpose(decoder_inputs, [1, 0])
        outputs = tf.TensorArray(tf.float32, size=steps)
        t = tf.constant(0, dtype=tf.int32)

        def cond(t, hidden, cell, input_feed, outputs):
            return t < steps

        def body(t, hidden, cell, input_feed, outputs):
            _, hidden, cell, output, _ = self.decode_step(
                tokens[t],
                encoder_outputs,
                encoder_mask,
                hidden,
                cell,
                input_feed,
                training=training,
            )
            return t + 1, hidden, cell, output, outputs.write(t, output)

        _, _, _, _, outputs = tf.while_loop(
            cond,
            body,
            [t, hidden, cell, input_feed, outputs],
            parallel_iterations=1,
        )
        features = tf.transpose(outputs.stack(), [1, 0, 2])
        return self.output_logits(features)

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs, encoder_mask, hidden, cell = self.encode(encoder_inputs, training=training)
        return self.decode_teacher(encoder_outputs, encoder_mask, hidden, cell, decoder_inputs, training=training)

    def warmup(self):
        encoder_inputs = tf.constant([[self.config.eos_id]], dtype=tf.int32)
        decoder_inputs = tf.constant([[self.config.bos_id]], dtype=tf.int32)
        self((encoder_inputs, decoder_inputs), training=False)

    def greedy_decode(self, encoder_inputs, max_length):
        encoder_outputs, encoder_mask, hidden, cell = self.encode(encoder_inputs, training=False)
        batch_size = tf.shape(encoder_inputs)[0]
        token = tf.fill([batch_size], self.config.bos_id)
        input_feed = tf.zeros((batch_size, self.config.decoder_hidden_size), dtype=tf.float32)
        finished = tf.zeros([batch_size], dtype=tf.bool)
        outputs = tf.TensorArray(tf.int32, size=max_length)
        max_length = tf.convert_to_tensor(max_length, dtype=tf.int32)
        t = tf.constant(0, dtype=tf.int32)

        def cond(t, token, hidden, cell, input_feed, finished, outputs):
            return t < max_length

        def body(t, token, hidden, cell, input_feed, finished, outputs):
            logits, hidden, cell, input_feed, _ = self.decode_step(
                token,
                encoder_outputs,
                encoder_mask,
                hidden,
                cell,
                input_feed,
                training=False,
            )
            token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            token = tf.where(finished, self.config.eos_id, token)
            finished = tf.logical_or(finished, tf.equal(token, self.config.eos_id))
            return t + 1, token, hidden, cell, input_feed, finished, outputs.write(t, token)

        _, _, _, _, _, _, outputs = tf.while_loop(
            cond,
            body,
            [t, token, hidden, cell, input_feed, finished, outputs],
            parallel_iterations=1,
        )
        return tf.transpose(outputs.stack(), [1, 0])

    def beam_decode(self, encoder_input, max_length, beam_size=5, length_penalty=0.6):
        if len(encoder_input.shape) == 1:
            encoder_input = encoder_input[None, :]
        encoder_outputs, encoder_mask, hidden, cell = self.encode(encoder_input, training=False)
        input_feed = tf.zeros((1, self.config.decoder_hidden_size), dtype=tf.float32)
        beams = [([self.config.bos_id], 0.0, hidden, cell, input_feed, False)]
        for _ in range(max_length):
            candidates = []
            for tokens, score, beam_hidden, beam_cell, beam_input_feed, done in beams:
                if done:
                    candidates.append((tokens, score, beam_hidden, beam_cell, beam_input_feed, done))
                    continue
                logits, next_hidden, next_cell, next_input_feed, _ = self.decode_step(
                    tf.constant([tokens[-1]], dtype=tf.int32),
                    encoder_outputs,
                    encoder_mask,
                    beam_hidden,
                    beam_cell,
                    beam_input_feed,
                    training=False,
                )
                log_probs = tf.nn.log_softmax(logits[0]).numpy()
                top_ids = log_probs.argsort()[-beam_size:][::-1]
                for token_id in top_ids:
                    token_id = int(token_id)
                    candidates.append(
                        (
                            tokens + [token_id],
                            score + float(log_probs[token_id]),
                            next_hidden,
                            next_cell,
                            next_input_feed,
                            token_id == self.config.eos_id,
                        )
                    )
            beams = sorted(
                candidates,
                key=lambda item: item[1] / ((len(item[0]) ** length_penalty) or 1.0),
                reverse=True,
            )[:beam_size]
            if all(done for _, _, _, _, _, done in beams):
                break
        best_tokens = max(
            beams,
            key=lambda item: item[1] / ((len(item[0]) ** length_penalty) or 1.0),
        )[0]
        return best_tokens[1:]
