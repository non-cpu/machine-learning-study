# Transformer seq2seq

import os
import io
import re
import time
import pickle
import random
import pathlib
import unicodedata

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocessSentence(s: str):
    s = ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'([.?!,¿])', r' \1 ', s)
    s = re.sub(r'[" "]+', ' ', s)
    s = re.sub(r'[^a-z.?!,¿]+', ' ', s)
    s = '<start> ' + s.strip() + ' <end>'
    return s

def tokenize(lang):
    lang_tokenizer = Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)  
    tensor = pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

path = os.path.dirname(__file__)

try:
    with open(f'{path}/spa-eng.pickle', 'rb') as f:
        input_tensor, inp_lang_tokenizer, target_tensor, targ_lang_tokenizer = pickle.load(f)
except:
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True
    )

    path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

    lines = io.open(path_to_file, encoding='UTF-8').read().split('\n')[:-1]
    wordPairs = [[preprocessSentence(s) for s in line.split('\t')] for line in lines]

    en, sp = zip(*wordPairs)

    input_tensor, inp_lang_tokenizer = tokenize(sp)
    target_tensor, targ_lang_tokenizer = tokenize(en)

    dataset = (input_tensor, inp_lang_tokenizer, target_tensor, targ_lang_tokenizer)

    with open(f'{path}/spa-eng.pickle', 'wb') as f:
        pickle.dump(dataset, f)

max_input_len, max_target_len = input_tensor.shape[1], target_tensor.shape[1]
input_vocab_size, target_vocab_size = len(inp_lang_tokenizer.word_index) + 1, len(targ_lang_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))

train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)

train_val_size = train_size + val_size

train_val_dataset = dataset.take(train_val_size)
test_dataset = dataset.skip(train_val_size)

train_val_dataset = train_val_dataset.shuffle(buffer_size=train_val_size)
train_val_dataset = train_val_dataset.shuffle(buffer_size=train_val_size)

train_dataset = train_val_dataset.take(train_size)
val_dataset = train_val_dataset.skip(train_size)

print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))

train_batch_size = 64
val_batch_size = 32
test_batch_size = 64

train_dataloader = train_dataset.batch(train_batch_size, drop_remainder=True)
val_dataloader = val_dataset.batch(val_batch_size, drop_remainder=True)
test_dataloader = test_dataset.batch(test_batch_size, drop_remainder=True)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_size, vocab_size, embedding_dim):
        super().__init__()
        self.sequence_size = sequence_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.token_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.positional_embedding = tf.keras.layers.Embedding(self.sequence_size, self.embedding_dim)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(length)

        embedded_tokens = self.token_embedding(x)
        embedded_positions = self.positional_embedding(positions)
        
        return embedded_tokens + embedded_positions

    def compute_mask(self, x, mask=None):
        return tf.math.not_equal(x, 0)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, sequence_size, vocab_size, embedding_dim, dense_dim, num_heads, supports_masking):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.supports_masking = supports_masking

        self.attention = tf.keras.layers.MultiHeadAttention(self.num_heads, self.embedding_dim)
        self.dense_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_dim, activation='relu'),
            tf.keras.layers.Dense(self.embedding_dim)
        ])

        self.layerNorm1 = tf.keras.layers.LayerNormalization()
        self.layerNorm2 = tf.keras.layers.LayerNormalization()

        self.positional_embedding = PositionalEmbedding(sequence_size, vocab_size, embedding_dim)

    def call(self, x, padding_mask=None, training=False):
        x = self.positional_embedding(x)
        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype='int32')

        attention_output = self.attention(x, x, x, padding_mask, training=training)
        
        proj_x = self.layerNorm1(x + attention_output)
        proj_output = self.dense_proj(proj_x)

        return self.layerNorm2(proj_x + proj_output)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, sequence_size, vocab_size, embedding_dim, latent_dim, num_heads, supports_masking, dropout_prob):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.supports_masking = supports_masking
        self.dropout_prob = dropout_prob

        self.attention1 = tf.keras.layers.MultiHeadAttention(self.num_heads, self.embedding_dim)
        self.attention2 = tf.keras.layers.MultiHeadAttention(self.num_heads, self.embedding_dim)

        self.dense_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_dim, activation='relu'),
            tf.keras.layers.Dense(self.embedding_dim)
        ])

        self.layerNorm1 = tf.keras.layers.LayerNormalization()
        self.layerNorm2 = tf.keras.layers.LayerNormalization()
        self.layerNorm3 = tf.keras.layers.LayerNormalization()

        self.positional_embedding = PositionalEmbedding(sequence_size, vocab_size, embedding_dim)

        self.dropout = tf.keras.layers.Dropout(self.dropout_prob)

        self.out_dense = tf.keras.layers.Dense(vocab_size)

    def get_causal_attention_mask(self, x):
        input_shape = tf.shape(x)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, x, encoder_outputs, padding_mask=None, training=False):
        x = self.positional_embedding(x)
        causal_mask = self.get_causal_attention_mask(x)

        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype='int32')
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output1 = self.attention1(x, x, x, causal_mask, training=training)
        out1 = self.layerNorm1(x + attention_output1)

        attention_output2 = self.attention2(out1, encoder_outputs, encoder_outputs, padding_mask, training=training)
        out2 = self.layerNorm2(out1 + attention_output2)

        proj_output = self.dense_proj(out2)

        out3 = self.layerNorm3(out2 + proj_output)
        out3 = self.dropout(out3, training=training)
        return self.out_dense(out3)

max_epoch = 20
learning_rate = 1e-3

rnn_units = 2048
embedding_dim = 256

loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

encoder = TransformerEncoder(max_input_len, input_vocab_size, embedding_dim, rnn_units, 8, True)
decoder = TransformerDecoder(max_target_len, target_vocab_size, embedding_dim, rnn_units, 8, True, 0.5)

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

encoder_inputs = tf.keras.Input(shape=(None,), dtype='int64')
encoder_outputs = encoder(encoder_inputs)

decoder_inputs = tf.keras.Input((None,), dtype='int64')
decoder_outputs = decoder(decoder_inputs, encoder_outputs)

transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

transformer.summary()

def decode_tr_sequence(sentence):
    result = str()
    sentence = preprocessSentence(sentence)

    inputs = inp_lang_tokenizer.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs, maxlen=max_input_len, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    dec_input = tf.cast(tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 1), dtype=tf.int64)

    for t in range(max_target_len):
        dec_outputs = transformer([inputs, dec_input])

        predicted_id = tf.expand_dims(tf.argmax(tf.math.softmax(dec_outputs[:, t, :]), axis=1), 0)

        dec_input = tf.concat([dec_input, predicted_id], 1)
        sampled_token = targ_lang_tokenizer.index_word[predicted_id.numpy()[0, 0]]

        if sampled_token == '<end>':
            result += '<end>'

            break

        result += sampled_token + ' '

    return result

print(decode_tr_sequence('¿Todavía están en casa?'))

def format_dataset(_src, _tar):
    # return ({
    #     'encoder_inputs': _src,
    #     'decoder_inputs': _tar[:, :-1]
    # }, _tar[:, 1:])

    return ({
        'input_1': _src,
        'input_2': _tar[:, :-1]
    }, _tar[:, 1:])

transformer.compile(
    optimizer=optimizer,
    loss=loss_function,
    # metrics=['accuracy']
)

# tf.config.run_functions_eagerly(True) # DEBUG

transformer.fit(
    train_dataloader.map(format_dataset),
    epochs=max_epoch,
    # callbacks=None,
    validation_data=val_dataloader.map(format_dataset)
)

print(decode_tr_sequence('¿Todavía están en casa?'))
