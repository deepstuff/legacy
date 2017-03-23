import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset
dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename='../data/legacy-vectors.csv',
                                                                 target_dtype=np.float32,
                                                                 features_dtype=np.float32)

# split in to train and test sets
train_data, test_data, train_target, test_target = \
    train_test_split(dataset.data, dataset.target, test_size=0.33, stratify=dataset.target, random_state=1)

# construct fake fNNN feature names - we have 585 real-valued columns (as opposed to sparse columns)
feature_columns = [tf.contrib.layers.real_valued_column('f'+str(i)) for i in range(len(dataset.data[0]))]

# create a dictionary of generic fNNN feature names -> 1d array of values for that feature
# I don't know why tf.contrib.learn wants the data in this format
def input_fn(data, target):
    feature_cols = {'f'+str(i): tf.constant(data[:,i]) for i in range(len(data[0]))}
    label = tf.constant(target)
    return feature_cols, label

def train_input_fn():
    return input_fn(train_data, train_target)

def test_input_fn():
    return input_fn(test_data, test_target)

# linear classifier
classifier = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, model_dir='../models/testsingle-layer-model')

# train classifier - ignore the warnings
classifier.fit(input_fn=train_input_fn, steps=100000)

# evaluate classifier on train data
train_results = classifier.evaluate(input_fn=train_input_fn, steps=1)

# evaluate classifier on test data
test_results = classifier.evaluate(input_fn=test_input_fn, steps=1)

print('train accuracy', train_results['accuracy'])
print('test accuracy', test_results['accuracy'])
