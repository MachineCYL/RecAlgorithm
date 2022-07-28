# -*-coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report


def dssm_model(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.Model(feature_inputs, output)
    return model


def gen_dataset(data_df: pd.DataFrame, columns: dict):
    data_dict = dict()

    def _get_type(type_str):
        if type_str == "int32":
            return np.int32
        elif type_str == "float32":
            return np.float32
        elif type_str == "string" or type_str == "str":
            return np.str
        else:
            return np.int32

    for key in columns.keys():
        data_dict[key] = np.array(data_df[key]).astype(_get_type(columns[key]))

    return data_dict


def parse_argvs():
    parser = argparse.ArgumentParser(description='[DSSM]')
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--model_path", type=str, default='./model_param')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--monitor", type=str, default="val_accuracy", choices=["val_accuracy", "val_auc"])
    parser.add_argument("--batch_size", type=int, default=12)
    args = parser.parse_args()
    print('[input params] {}'.format(args))

    return parser, args


if __name__ == '__main__':
    parser, args = parse_argvs()
    data_path = args.data_path
    model_path = args.model_path
    monitor = args.monitor
    epoch = args.epoch
    batch_size = args.batch_size

    # ====================================================================================
    # read data
    data_path = os.path.abspath(data_path)
    print("[DSSM] read file path: {}".format(data_path))
    train_data = pd.read_csv(os.path.join(data_path, "trainingSamples.csv"), sep=",")
    test_data = pd.read_csv(os.path.join(data_path, "testSamples.csv"), sep=",")
    data_pd = pd.concat([train_data, test_data])

    # ====================================================================================
    # define input for keras model
    columns_dict = {
        'movieId': 'int32',
        'movieGenre1': 'string',
        'movieAvgRating': 'float32',
        'userId': 'int32',
        'userGenre1': 'string',
        'userAvgRating': 'float32'
    }

    inputs = dict()
    for key in columns_dict.keys():
        inputs[key] = tf.keras.layers.Input(name=key, shape=(), dtype=columns_dict[key])
    print("[DSSM] input for keras model: \n {}".format(inputs))

    # ====================================================================================
    # movie embedding feature
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

    movie_genre_1_vocab = data_pd['movieGenre1'].dropna().unique()
    movie_genre_1_col = tf.feature_column.categorical_column_with_vocabulary_list(key='movieGenre1',
                                                                                  vocabulary_list=movie_genre_1_vocab)
    movie_genre_1_emb_col = tf.feature_column.embedding_column(movie_genre_1_col, 10)

    movie_avg_rating = tf.feature_column.numeric_column(key='movieAvgRating')

    # user embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)

    user_genre_1_vocab = data_pd['userGenre1'].dropna().unique()
    user_genre_1_col = tf.feature_column.categorical_column_with_vocabulary_list(key='userGenre1',
                                                                                 vocabulary_list=user_genre_1_vocab)
    user_genre_1_emb_col = tf.feature_column.embedding_column(user_genre_1_col, 100)

    user_avg_rating = tf.feature_column.numeric_column(key='userAvgRating')

    # ====================================================================================
    # train model
    model = dssm_model(feature_inputs=inputs,
                       item_feature_columns=[movie_emb_col, movie_genre_1_emb_col, movie_avg_rating],
                       user_feature_columns=[user_emb_col, user_genre_1_emb_col, user_avg_rating],
                       hidden_units=[30, 10])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])

    filepath = os.path.join(model_path, "checkpoint", "dssm-weights-best.hdf5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor=monitor, verbose=1, save_best_only=True, mode='max')

    train_data_input = gen_dataset(data_df=train_data, columns=columns_dict)
    model.fit(x=train_data_input, y=train_data["label"].values,
              epochs=epoch, callbacks=[checkpoint], verbose=2, batch_size=batch_size, validation_split=0.1)

    # ====================================================================================
    # predict, use best model.
    test_data_input = gen_dataset(data_df=test_data, columns=columns_dict)
    model.load_weights(filepath=filepath)

    pred_ans = model.predict(x=test_data_input, batch_size=batch_size)
    print("\n[BEST] ===============================================================")
    print("[test] LogLoss: {} ".format(round(log_loss(test_data["label"].values, pred_ans), 4)))
    print("[test] Accuracy: {} ".format(round(accuracy_score(test_data["label"].values, pred_ans >= 0.5), 4)))
    print("[test] AUC: {} ".format(round(roc_auc_score(test_data["label"].values, pred_ans), 4)))
    print("[test] classification_report: \n{} ".format(classification_report(test_data["label"].values, pred_ans >= 0.5, digits=4)))

    # ====================================================================================
    # save model
    model_path = os.path.abspath(model_path)
    print("[DSSM] save model path: {}".format(model_path))

    model.summary()
    tf.keras.models.save_model(
        model,
        os.path.join(model_path, "dssm"),
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
