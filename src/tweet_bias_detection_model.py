"""
The main file that contains the model definitions and word
processing methods, including embeddings, vectorization, etc.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
    seq_len: int) -> Embedding:
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
    with open(path_words) as f:
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
    return embedding_layer

if __name__ == '__main__':
    # Get the pretrained embedding layer with glove.
    embedding = get_embedding_layer('../data/glove.6B.100d.txt',
        100, 20000, 140)

