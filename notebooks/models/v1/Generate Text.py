# %%
import keras

# %%
from pathlib import Path

import tensorflow as tf
tf_session = tf.Session()
from keras import backend as K
K.set_session(tf_session)

from sklearn.externals import joblib

# Local library with model definitions for training and generating
from models import create_training_model, Generator

# %%
# Load a copy of the training model from disk by creating a new model using the same parameters and then loading the weights.

output_dir = Path('minus_twaiku')

# Get the parameters used for creating the model
latent_dim, n_tokens, max_line_length, tokenizer = joblib.load(output_dir / 'metadata.pkl')

# Create the new placeholder model
training_model, lstm, lines, inputs, outputs = create_training_model(latent_dim, n_tokens)

# Load the specified weights
training_model.load_weights(output_dir / '2048-05-1.72.hdf5')

# %%
# Create a generator using the training model as the template

generator = Generator(lstm, lines, tf_session, tokenizer, n_tokens, max_line_length)

# %%
for i in range(50):
    generator.generate_haiku()
    print()

# %%
for i in range(50):
    generator.generate_haiku([3, 5, 3])
    print()

# %%
for i in range(50):
    generator.generate_haiku([10, 10, 10])
    print()

# %%
for i in range(50):
    generator.generate_haiku(temperature=.3)
    print()

# %%
for i in range(50):
    generator.generate_haiku(temperature=.5)
    print()

# %%
for i in range(50):
    generator.generate_haiku(temperature=1)
    print()

# %%
for i in range(50):
    generator.generate_haiku([3, 3, 3], temperature=.3)
    print()

# %%
for i in range(50):
    generator.generate_haiku([3, 3, 3], temperature=.75)
    print()

# %%
