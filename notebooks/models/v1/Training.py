# %%
import logging
import tensorflow as tf

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

import keras

# %%
from pathlib import Path

from tensorflow.core.protobuf import rewriter_config_pb2
config = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization  = off
tf_session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(tf_session)

from keras.callbacks import ModelCheckpoint,  CSVLogger
from keras.layers import Add, Dense, Input, LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

import numpy as np
import pandas as pd
import joblib
# Local library with model definitions for training and generating
from models import Generator, create_training_model

# %%
"""
# Load Input
"""

# %%
# Settings

# Percent of samples to use for training, might be necessary if you're running out of memory
sample_size = 1

# The latent dimension of the LSTM
latent_dim = 2048

# Number of epochs to train for
epochs = 20

root_path = Path('../../..')
input_path = root_path / 'input'
poem_path = input_path / 'poems'
haiku_path = poem_path / 'haikus.csv'

name = 'all_data_test_2'
output_dir = Path('output_%s' % name)
output_dir.mkdir()

# %%
df = pd.read_csv(str(haiku_path))
df = df.sample(frac=sample_size)
df

# %%
"""
# Format Input for Training
"""

# %%
# Duplicate lines with ambiguous syllable counts
# (syllable counts where there is a comma because
# multiple pronounciations are acceptable)

lines = set([0, 1, 2])

for i in range(3):
    lines.remove(i)
    df = df[[
        '0', '1', '2',
        #'1_syllables', '2_syllables'
    ] + ['%s_syllables' % j for j in lines]].join(
        df['%s_syllables' % i].str.split(
            ',', expand=True
        ).stack(-1).reset_index(
            level=1, drop=True
        ).rename('%s_syllables' % i)
    ).drop_duplicates()
    lines.add(i)

df

# %%
# Drop samples that are longer that the 99th percentile of length

max_line_length = int(max([df['%s' % i].str.len().quantile(.99) for i in range(3)]))
df = df[
    (df['0'].str.len() <= max_line_length) & 
    (df['1'].str.len() <= max_line_length) & 
    (df['2'].str.len() <= max_line_length)
].copy()
df

# %%
# Pad the lines to the max line length with new lines
for i in range(3):
    # For input, duplicate the first character
    # TODO - Why?
    df['%s_in' % i] = (df[str(i)].str[0] + df[str(i)]).str.pad(max_line_length+2, 'right', '\n')
    
    # 
    #df['%s_out' % i] = df[str(i)].str.pad(max_line_len, 'right', '\n') + ('\n' if i == 2 else df[str(i+1)].str[0])
    
    # TODO - trying to add the next line's first character before the line breaks
    if i == 2: # If it's the last line
        df['%s_out' % i] = df[str(i)].str.pad(max_line_length+2, 'right', '\n')
    else: 
        # If it's the first or second line, add the first character of the next line to the end of this line.
        # This helps with training so that the next RNN has a better chance of getting the first character right.
        df['%s_out' % i] = (df[str(i)] + '\n' + df[str(i+1)].str[0]).str.pad(max_line_length+2, 'right', '\n')
    
max_line_length += 2

df

# %%
inputs = df[['0_in', '1_in', '2_in']].values

tokenizer = Tokenizer(filters='', char_level=True)
tokenizer.fit_on_texts(inputs.flatten())
n_tokens = len(tokenizer.word_counts) + 1

# X is the input for each line in sequences of one-hot-encoded values
X = np_utils.to_categorical([
    tokenizer.texts_to_sequences(inputs[:,i]) for i in range(3)
], num_classes=n_tokens)

outputs = df[['0_out', '1_out', '2_out']].values

# Y is the output for each line in sequences of one-hot-encoded values
Y = np_utils.to_categorical([
    tokenizer.texts_to_sequences(outputs[:,i]) for i in range(3)
], num_classes=n_tokens)

# X_syllables is the count of syllables for each line
X_syllables = df[['0_syllables', '1_syllables', '2_syllables']].values

# %%
joblib.dump([latent_dim, n_tokens, max_line_length, tokenizer], str(output_dir / 'metadata.pkl'))

# %%


# %%
"""
# Training Model
"""

# %%
training_model, lstm, lines, inputs, outputs = create_training_model(latent_dim, n_tokens)

filepath = str(output_dir / ("%s-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5" % latent_dim))
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

csv_logger = CSVLogger(str(output_dir / 'training_log.csv'), append=True, separator=',')

callbacks_list = [checkpoint, csv_logger]

training_model.fit([
    X[0], X_syllables[:,0], 
    X[1], X_syllables[:,1], 
    X[2], X_syllables[:,2]
], [Y[0], Y[1], Y[2]], batch_size=64, epochs=epochs, validation_split=.1, callbacks=callbacks_list)

# %%
"""
# Test Model
"""

# %%
generator = Generator(lstm, lines, tf_session, tokenizer, n_tokens, max_line_length)

# %%
generator.generate_haiku()

# %%
