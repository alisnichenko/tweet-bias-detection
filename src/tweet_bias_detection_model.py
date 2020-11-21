"""
The main file that contains the model definitions and word
processing methods, including embeddings, vectorization, etc.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import (
    TextVectorization
)


def get_model_data(csv_path: str, val_split: int) -> tuple():
    """
    Returns a tuple of size 4, where the training samples, validation
    samples, training labels, and validation labels are stored in
    respective lists. The data is shuffled and split according to
    parameters above.
    Args:
        csv_path - a string for the relative path of the csv data file.
        val_split - double value that indicated the validation split.
    Returns:
        a tuple, where the respective elements are list and:
        tuple[0] - training sample tweets.
        tuple[1] - validation sample tweets.
        tuple[2] - training labels of tweets.
        tuple[3] - validation labels of tweets.
    """
    # Get csv file into pandas dataframe.
    tweets_df = pd.read_csv(csv_path)
    # Convert columns to lists for shuffling.
    biases = tweets_df['biases'].to_list()
    tweets = tweets_df['tweets'].to_list()

    # Shuffle the data.
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(tweets)
    rng = np.random.RandomState(seed)
    rng.shuffle(biases)

    # Extract training and validation splits.
    num_val_tweets = int(val_split * len(tweets))
    train_samples = tweets[:-num_val_tweets]
    val_samples = tweets[-num_val_tweets:]
    train_labels = biases[:-num_val_tweets]
    val_labels = biases[-num_val_tweets:]

    return (train_samples, val_samples, train_labels, val_labels)


def get_embedding_layer(path_words: str, dim: int, tokens: int,
                        seq_len: int) -> tuple():
    """
    Returns tf.keras.layers.Embedding layer trained on stanford's glove
    dataset. This layer will be used for the bias detection model.
    Args:
        path_words: string path to the glove words file.
        dim: the dimensionality of the glove dataset.
        tokens: integer that shows the top number of words considered.
        seq_len: maximum sequence length. For tweets 140.
    Returns:
        Pretrained embedding layer from keras on glove dataset.
    """
    train_samples, val_samples, train_labels, val_labels = get_model_data(
        '../data/tweets_biases.csv', 0.2)
    # Create vocabulary index.
    vectorizer = TextVectorization(max_tokens=tokens,
                                   output_sequence_length=seq_len)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
    vectorizer.adapt(text_ds)

    # Incorporate glove embeddings.
    embeddings_idx = {}
    with open(path_words, "r", encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_idx[word] = coefs
    print("[INFO] Finished embeddings. Found {0} word vecs."
          .format(len(embeddings_idx)))

    # Variables for embedding matrix.
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc) + 2
    embedding_dim = dim
    hits, misses = 0, 0

    # Prepare embedding matrix.
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_idx.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    # Print info about the word preprocessing success.
    print("[INFO] Converted {0} words ({1} misses).".format(hits, misses))

    # Build the embedding layer.
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        # Pretrained layer on glove.
        trainable=False,
    )

    print("[INFO] Built the embedding layer ({0} dim).".format(embedding_dim))
    return (embedding_layer, vectorizer)


def get_model(embedding_layer: Embedding,
              vectorizer: TextVectorization) -> keras.Model():
    """
    Returns a keras model with the following architecture: (...).
    Model features embedding layer pretrained on glove dataset.
    Args:
        embedding_layer: embedding layer using glove dataset.
    Returns:
        keras.Model() with simple conv nets.
    """
    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(128, 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(embedded_sequences)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(128, 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(128, 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(int_sequences_input, preds)

    # Print model summary in the console.
    print("[INFO] Model summary displayed below.")
    print(model.summary())

    # Train the model.
    train_samples, val_samples, train_labels, val_labels = get_model_data(
        '../data/tweets_biases.csv', 0.2)
    x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
    x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()
    y_train = np.array(train_labels) / 10
    y_val = np.array(val_labels) / 10

    # Reduce learning rate over time to find optimal learning rate
    batch_size = 128
    steps_per_epoch = len(train_samples) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=steps_per_epoch * 1000,
        decay_rate=1,
        staircase=False)

    # Set early stopping callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Compile the model.
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  metrics=["acc"])
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_data=
    (x_val, y_val), callbacks=[callback])

    # Export the model.
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    predictions = model(x)
    end_to_end_model = keras.Model(string_input, predictions)

    # Sample prediction.
    prob = end_to_end_model.predict(
        [["Unbelievable move by the senate, it is impressive how quickly they ruin the country."]]
    )
    print("Probability for given input is {0}.".format(prob[0][np.argmax(prob[0])]))
    print("[INFO] Trained the model. Exporting the model.")
    end_to_end_model.save('../data/')

    return end_to_end_model


if __name__ == '__main__':
    # Get the model with pretrained embedding layer with glove.
    embedding, vectorizer = get_embedding_layer('../data/glove.6B.100d.txt',
                                                100, 20000, 140)
    model = get_model(embedding, vectorizer)