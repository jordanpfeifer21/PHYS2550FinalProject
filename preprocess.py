import numpy as np
import tensorflow as tf
import pandas as pd
from functools import partial

############## Preprocessing on Oscar / local machine ##########################

def compute_stats(data):
    # Compute the mean and standard deviation of features in a TensorFlow dataset.
    total_sum = tf.constant(0, dtype=tf.float32)
    num_elements = tf.constant(0, dtype=tf.float32)
    
    # Calculate mean.
    for item in data:
        total_sum += item
        num_elements += 1
    mean = total_sum / num_elements

    # Calculate std dev.
    total_squared_error = tf.constant(0, dtype=tf.float32)
    for item in data:
        total_squared_error += (item - mean) ** 2
    variance = total_squared_error / num_elements
    std_dev = tf.sqrt(variance)
    return tf.data.Dataset.from_tensors(mean), tf.data.Dataset.from_tensors(std_dev)


def parse_csv_line(line, num_features, one_channel=False):
    # Function taking one CSV line and parsing it.
    def_values = [.0] * num_features # To replace missing values.
    fields = tf.io.decode_csv(line, record_defaults=def_values) # Each column maps to one tensor.
    columns = tf.stack(fields) # Stack tensors together to make a row.
    if one_channel == False:
        return tf.reshape(columns, [3, 700])
    else:
        return columns


def dataset_split_train_test(dataset, test_ratio, instances, shuffle_buffer_size, seed=42):
    # Split a TensorFlow dataset into training and test ones.
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    test_size = int(test_ratio * instances)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(instances))
    X_train = dataset.skip(test_size)
    X_test = dataset.take(test_size)
    return X_train, X_test


def standardization(data, X_train_mean, X_train_std_dev):
    # Perform standardization.
    data_len = tf.data.experimental.cardinality(data).numpy()
    X_train_mean = X_train_mean.repeat(data_len)
    X_train_std_dev = X_train_std_dev.repeat(data_len)
    data_scaled = tf.data.Dataset.zip((data, X_train_mean, X_train_std_dev))
    data_scaled = data_scaled.map(lambda x, y, z: tf.where(z == 0., tf.zeros_like(x), (x - y) / z)) # Remove NaN when divide by 0.
    # Make sure there are no NaNs.
    data_scaled = data_scaled.map(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))
    return data_scaled

def give_label(data):
    # Give "label" to unlabelled data: (inst, target) in our case is (inst, inst).
    return data.map(lambda x: (x, x))


def csv_preprocces(filepaths, test_ratio=0.2, valid_ratio=0.1, num_features=2100, 
                     shuffle_buffer_size=10000, batch_size=32, seed=42, one_channel=False,
                     training=True):
    '''
    Read and preprocess single or multiple .csv data files. First, we create a 
    Tensorflow dataset containing the list of filepaths we want to read. Then, we
    read from all the files at a time and interleave their lines. The dataset
    will contain strings, so we convert them into float type variables. Then we 
    split our dataset into three: train, test, and validation. Using the training
    dataset, we find X_mean and X_std_dev that we use to perform standardization.
    The function returns three batched datasets.

    :param filepaths: A single string or a list of strings containing file paths to data.
    :param: test_ratio: Proportion of the dataset to be used for testing.
    :param: valid_ratio: Proportion of the training dataset to be used for testing.
    :param num_features: Number of features. The standard value is 2100 (700 particles + 3 coordinates).
    :param shuffle_buffer_size: Shuffle the dataset using a buffer of size. The default value 10000 could be higher if the system doesn't crush.
    :param batch_size: Size of a batch the final datasets are split into.
    :param seed: Random seed. The default value is 42 to ensure reproducibility of the results.
    :param one_channel: Determines whether the returned tensors are split into three channels (pT, eta, phi) when set to False, or if they're combined into one channel when set to True.
    :param training: 'True' if we preprocess the data to train the model, 'False' - to make predictions.

    :return: TensorFlow datasets split into batches, ready to be for training or evaluation.
    '''
    # List of filepaths.
    dataset_filepaths = tf.data.Dataset.list_files(
        filepaths, seed=seed # Shuffle filepaths.
    )
    
    # To process lines from files.
    dataset = dataset_filepaths.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        num_parallel_calls=tf.data.AUTOTUNE # Read files in parallel.
    )
    
    # Convert strings into floats.
    dataset = dataset.map(partial(parse_csv_line, num_features=num_features, one_channel=one_channel), 
                          num_parallel_calls=tf.data.AUTOTUNE) # To do it in parallel.

    # Split into train, test, and valid datasets.
    ## Not the best way to do it, but this way we don't waste any resources.
    ## tf.data.experimental.cardinality(dataset) returns unknown number because the dataset is too
    ## big to easy calculate its length. I introduced `instances" instead to make it faster.
    print('----------------- Data split ----------------------------')
    print(f'test = {test_ratio * 100}%, validation = {valid_ratio * 100}% of the train dataset.')
    print('---------------------------------------------------------')
    
    # TODO: Find more efficient way to calculate the size of large datasets.
    print(f'Calculating the size of the original dataset...')
    instances = tf.data.experimental.cardinality(dataset)
    if instances == -2: # Means the dataset is too large to give its size efficiently.
        count = 0
        for item in dataset:
            count += 1
        instances = count
    print(f'its size is {instances}\n')
    
    print(f'Spliting the dataset...\n')
    X_train_full, X_test = dataset_split_train_test(dataset=dataset, test_ratio=test_ratio,
                                                    shuffle_buffer_size=shuffle_buffer_size,
                                                    instances=instances)
    train_instances = tf.data.experimental.cardinality(X_train_full).numpy()
    X_train, X_valid = dataset_split_train_test(dataset=X_train_full, test_ratio=0.1,
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                instances=train_instances)
    # Standardization.
    print('Standardization...')
    X_train_mean, X_train_std_dev = compute_stats(X_train)
    X_train_std_scaled = standardization(X_train, X_train_mean=X_train_mean, X_train_std_dev=X_train_std_dev)
    X_test_std_scaled = standardization(X_test, X_train_mean=X_train_mean, X_train_std_dev=X_train_std_dev)
    X_valid_std_scaled = standardization(X_valid, X_train_mean=X_train_mean, X_train_std_dev=X_train_std_dev)

    # Split into batches.
    X_train_std_scaled = X_train_std_scaled.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    X_test_std_scaled = X_test_std_scaled.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    X_valid_std_scaled = X_valid_std_scaled.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    train_len = tf.data.experimental.cardinality(X_train_std_scaled).numpy()
    test_len = tf.data.experimental.cardinality(X_test_std_scaled).numpy()
    valid_len = tf.data.experimental.cardinality(X_valid_std_scaled).numpy()

    # Print results.
    print('----------------- Preprocessing is done -----------------')
    print(f'The data is loaded as TensorFlow datasets, shuffled, split, standardized, and batched (batch = {batch_size}).')
    print(f'Train dataset: {train_len} batched elements.')
    print(f'Test dataset {test_len} batched elements.')
    print(f'Validation dataset {valid_len} bathced elements.')
    print('----------------- one_channel parameter -----------------')
    print(f'one_channel = {one_channel}')
    print(f'Determines whether the returned tensors are split into three channels (pT, eta, phi) when set to False, or if they are combined into one channel when set to True.')
    
    if training == True:
        return give_label(X_train_std_scaled), give_label(X_test_std_scaled), give_label(X_valid_std_scaled)
    else:
        return X_train_std_scaled, X_test_std_scaled, X_valid_std_scaled





############## Functions below might be useful for Oscar #######################

def get_data_h5(file_path, load_option='full'):
    '''
    Load a single h5 file containing 1M either LHC events or the background
    simulated events.

    :param file_path: Path to the h5 data file.
    :param load_option: 
        'full' - load everything into memory,
        'random' - load random 10k events.
    :return: 
       Single pandas DataFrame.

    Warning: the h5 file with events is too huge to open on the local machine,
    but Oscar should not have any problems.
    '''
    # Process the file based on the chosen load option.
    if load_option == 'full':
        data = pd.read_hdf(file_path)
    elif load_option == 'random':
        np.random.seed(42)
        data = pd.read_hdf(file_path, skiprows=np.random.randint(0, 1e6-10000),
                           nrows=10000)
    else:
        raise ValueError("load_option parameter accepts only two strings:"
                         "'full' or 'random'.")

    # Print information about dataset size and resources used.
    print(f'Shape of the loaded dataset: {data.shape}')
    print(f'Memory usage: '
            f'{(data.memory_usage(index=False, deep=True) / 2**30).sum().round(3)} GB')
    
    return data



def get_data_chunks_h5(file_path, chunksize=10000):
    '''
    Load a single h5 file one chunk at a time.
    Function acts as a generator.

    :param file_path: Path to the h5 data file.
    :param chunksize: Size of one chunk. 
    yield: One chunk of the data at a time.

    The same warning as above: probably it will just crush because of the size of
    the original h5 file. 
    '''
    
    count = 0
    with pd.read_hdf(file_path, chunksize=chunksize) as reader:
        for data in reader:
            count += 1
            yield data          # Generator.
    print(f'The dataset was split into: {count} chunks.')