# -*-coding:utf-8 -*-
import os
import argparse
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F

from model.collaborative_filtering import CollaborativeFiltering


def parse_argvs():
    parser = argparse.ArgumentParser(description='[collaborativeFiltering]')
    parser.add_argument("--data_path", type=str, default='./data/ratings.csv')
    parser.add_argument("--model_path", type=str, default='./model_param')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--train_flag", type=bool, default=False)
    args = parser.parse_args()
    print('[input params] {}'.format(args))

    return parser, args


if __name__ == '__main__':
    parser, args = parse_argvs()
    data_path = args.data_path
    model_path = args.model_path
    train_flag = args.train_flag
    epoch = args.epoch

    conf = SparkConf().setAppName('collaborativeFiltering').setMaster("local[*]")
    spark_session = SparkSession.builder.config(conf=conf).getOrCreate()

    # read data
    data_path = os.path.abspath(data_path)
    data_path = "file://" + data_path
    print("[spark] read file path: {}".format(data_path))

    ratingSamples = spark_session.read.format('csv').option('header', 'true').load(data_path) \
        .withColumn("userIdInt", F.col("userId").cast(IntegerType())) \
        .withColumn("movieIdInt", F.col("movieId").cast(IntegerType())) \
        .withColumn("ratingFloat", F.col("rating").cast(FloatType()))
    training, test = ratingSamples.randomSplit((0.8, 0.2), seed=2022)

    # collaborative filtering start
    cf = CollaborativeFiltering(spark_session=spark_session)

    if train_flag is True:
        cf.train(train_set=training,
                 user_col='userIdInt',
                 item_col='movieIdInt',
                 rating_col='ratingFloat',
                 epoch=epoch)

        cf.save(model_dir=model_path)
    else:
        cf.load(model_dir=model_path)

    loss = cf.eval(test_set=test, label_col='ratingFloat', metric='rmse')
    print("[Root-mean-square error] {}".format(loss))

    # Generate top 10 movie recommendations for each user
    user_recs = cf.recommend_for_all_users(num_items=10)
    user_recs.show(10, False)

    # Generate top 10 user recommendations for each movie
    movie_recs = cf.recommend_for_all_items(num_users=10)
    movie_recs.show(10, False)

    # Generate top 10 movie recommendations for a specified set of users
    user_data = ratingSamples.select("userIdInt").distinct().limit(10)
    user_sub_recs = cf.recommend_for_user_subset(dataset=user_data, num_items=10)
    user_sub_recs.show(10, False)

    # Generate top 10 user recommendations for a specified set of movies
    movie_data = ratingSamples.select("movieIdInt").distinct().limit(10)
    movie_sub_recs = cf.recommend_for_item_subset(dataset=movie_data, num_users=10)
    movie_sub_recs.show(10, False)

    spark_session.stop()
