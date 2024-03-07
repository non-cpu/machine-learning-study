# RNN seq2seq

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

    lines = io.open(path_to_file, encoding='UTF-8').read().split('\n')[:32768]
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

train_dataset = train_val_dataset.take(train_size)
val_dataset = train_val_dataset.skip(train_size)

# print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))

train_batch_size = 64
val_batch_size = 32
test_batch_size = 64

train_dataloader = train_dataset.batch(train_batch_size, drop_remainder=True)
val_dataloader = val_dataset.batch(val_batch_size, drop_remainder=True)
test_dataloader = test_dataset.batch(test_batch_size, drop_remainder=True)

class GRUEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state

class GRUDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.fc(output)

        attention = None

        return output, state, attention

max_epoch = 40
learning_rate = 1e-3

rnn_units = 1024
embedding_dim = 256
teacher_forcing_ratio = 0.5

loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

encoder = GRUEncoder(input_vocab_size, embedding_dim, rnn_units)
decoder = GRUDecoder(target_vocab_size, embedding_dim, rnn_units)

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

@tf.function
def _step(input, target, teacher_forcing_ratio):
    loss = 0
    output = list()
    
    enc_output, dec_hidden = encoder(input)
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * target.shape[0], 1)

    for t in range(1, target.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden)

        ground_truth = tf.expand_dims(target[:, t], 1)
        loss += loss_function(ground_truth, predictions)

        predictions = tf.argmax(predictions, 2)
        output.append(predictions)

        if random.random() < teacher_forcing_ratio:
            dec_input = ground_truth
        else:
            dec_input = predictions

    return loss, output

@tf.function
def train_step(input, target, teacher_forcing_ratio):
    with tf.GradientTape() as tape:
        loss, output = _step(input, target, teacher_forcing_ratio)

    batch_loss = loss / target.shape[1]

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, output

@tf.function
def test_step(input, target):
    loss, output = _step(input, target, 0.0)
    batch_loss = loss / target.shape[1]
    return batch_loss, output

checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, f'{path}/checkpoints', max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

tensorboard_writer = tf.summary.create_file_writer(f'{path}/tensorboard')

step = 0

val_dataloader = iter(val_dataloader)

for epoch in range(max_epoch):
    start = time.time()

    total_loss = 0

    for (batch, (input, target)) in enumerate(train_dataloader):
        batch_loss, output = train_step(input, target, teacher_forcing_ratio)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch {epoch} Batch {batch} Loss {batch_loss.numpy():.4f}')

        step += 1

    train_loss = total_loss / len(train_dataloader)

    input, target = next(val_dataloader)
    val_loss, output = test_step(input, target)
    
    prediction = tf.squeeze(tf.stack(output, axis=1))
    
    input_text = inp_lang_tokenizer.sequences_to_texts(input.numpy())
    target_text = targ_lang_tokenizer.sequences_to_texts(target.numpy())
    prediction_text = targ_lang_tokenizer.sequences_to_texts(prediction.numpy())
    
    with tensorboard_writer.as_default(step=step):
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('val_loss', val_loss)

        tf.summary.text('val_input_loss', input_text[0])
        tf.summary.text('val_target_loss', target_text[0])
        tf.summary.text('val_prediction_loss', prediction_text[0])

    if (epoch) % 2 == 0:
        manager.save()

    print(f'Epoch {epoch} Train_loss {train_loss:.4f} Val_loss {val_loss:.4f} Sec {time.time() - start:.2f}')

# without attention, 0.5 : Epoch 39 Train_loss 0.0746 Val_loss 0.0305
# without attention, 1.0 : Epoch 39 Train_loss 0.0587 Val_loss 0.3219
