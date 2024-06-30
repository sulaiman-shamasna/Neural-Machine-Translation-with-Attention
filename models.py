import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf
import pathlib
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from check_shape import ShapeChecker

from preprocessing import (train_ds,
                           val_ds,
                           context_text_processor,
                           target_text_processor,
                           target_raw, context_raw)

from preprocessing import (ex_context_tok,
                           ex_tar_in,
                           ex_tar_out)

UNITS = 256

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor: tf_text.BertTokenizer, units: int):

        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(units, return_sequences=True, recurrent_initializer='glorot_uniform')
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
   
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, texts: List[str]) -> tf.Tensor:

        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context

# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):

        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
 
        shape_checker = ShapeChecker()

        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(query=x, value=context, return_attention_scores=True)

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

attention_layer = CrossAttention(UNITS)

# Attend to the encoded tokens
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(), output_dim=UNITS, mask_zero=True)
ex_tar_embed = embed(ex_tar_in)

result = attention_layer(ex_tar_embed, ex_context)

print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
print(f'Attention result, shape (batch, t, units): {result.shape}')
print(f'Attention weights, shape (batch, t, s): {attention_layer.last_attention_weights.shape}')

attention_layer.last_attention_weights[0].numpy().sum(axis=-1)

attention_weights = attention_layer.last_attention_weights
mask = (ex_context_tok != 0).numpy()

# plt.subplot(1, 2, 1)
# plt.pcolormesh(mask * attention_weights[:, 0, :])
# plt.title('Attention weights')

# plt.subplot(1, 2, 2)
# plt.pcolormesh(mask)
# plt.title('Mask')
# plt.show()

class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        """
        Add a method to the (Specified) Decoder class.
        This enables adding methods to the class while compiling!

        Args:
            fun: Function to add.

        Returns:
            Function added to the class.
        """
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor: tf_text.BertTokenizer, units: int):
 
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(),
                                                       mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(),
                                                       mask_token='', oov_token='[UNK]', invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.attention = CrossAttention(units)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

@Decoder.add_method
def call(self, context: tf.Tensor, x: tf.Tensor, state: tf.Tensor = None, return_state: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:

    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')

    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')

    if return_state:
        return logits, state
    else:
        return logits

decoder = Decoder(target_text_processor, UNITS)
logits = decoder(ex_context, ex_tar_in)

print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')

@Decoder.add_method
def get_initial_state(self, context: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

@Decoder.add_method
def tokens_to_text(self, tokens: tf.Tensor) -> tf.Tensor:

    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result

@Decoder.add_method
def get_next_token(self, context: tf.Tensor, next_token: tf.Tensor, done: tf.Tensor, state: tf.Tensor, temperature: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    logits, state = self(context, next_token, state=state, return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    done = done | (next_token == self.end_token)
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state

next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []

for n in range(10):
    next_token, done, state = decoder.get_next_token(ex_context, next_token, done, state, temperature=1.0)
    tokens.append(next_token)

tokens = tf.concat(tokens, axis=-1)  # (batch, t)

result = decoder.tokens_to_text(tokens)
result[:3].numpy()

class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):

        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units: int, context_text_processor: tf_text.BertTokenizer, target_text_processor: tf_text.BertTokenizer):

        super().__init__()
        self.encoder = Encoder(context_text_processor, units)
        self.decoder = Decoder(target_text_processor, units)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:

        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        # TODO: remove this
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

import einops
@Translator.add_method
def translate(self,
              texts,
              *,
              max_length=500,
              temperature=tf.constant(0.0)):
  shape_checker = ShapeChecker()
  context = self.encoder.convert_input(texts)
  batch_size = tf.shape(context)[0]
  shape_checker(context, 'batch s units')

  next_token, done, state = self.decoder.get_initial_state(context)

  tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)

  for t in tf.range(max_length):
    next_token, done, state = self.decoder.get_next_token(
        context, next_token, done, state, temperature)
    shape_checker(next_token, 'batch t1')

    tokens = tokens.write(t, next_token)

    if tf.reduce_all(done):
      break

  tokens = tokens.stack()
  shape_checker(tokens, 't batch t1')
  tokens = einops.rearrange(tokens, 't batch 1 -> batch t')
  shape_checker(tokens, 'batch t')

  text = self.decoder.tokens_to_text(tokens)
  shape_checker(text, 'batch')

  return text

model = Translator(UNITS, context_text_processor, target_text_processor)
# model.summary()
logits = model((ex_context_tok, ex_tar_in))

print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')

class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, inputs):
    return self.model.translate(inputs)
  

export = Export(model)

# _ = export.translate(inputs)

tf.saved_model.save(export, 'dynamic_translator',
                    signatures={'serving_default': export.translate})