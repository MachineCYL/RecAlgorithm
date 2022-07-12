# -*-coding:utf-8 -*-
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator


class CollaborativeFiltering(object):
    def __init__(self, spark_session):
        self.spark_session = spark_session
        self.model = None

    def train(self, train_set, user_col, item_col, rating_col, epoch=10):
        """
        Build the recommendation model using ALS on the training data
        Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        """
        als = ALS(regParam=0.01, maxIter=epoch, userCol=user_col, itemCol=item_col, ratingCol=rating_col,
                  coldStartStrategy='drop')
        self.model = als.fit(train_set)

    def eval(self, test_set, label_col='ratingFloat', metric='rmse'):
        """ Evaluate the model on the test data """
        predictions = self.model.transform(test_set)

        # self.model.itemFactors.show(10, truncate=False)
        # self.model.userFactors.show(10, truncate=False)
        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=label_col, metricName=metric)
        loss = evaluator.evaluate(predictions)
        return loss

    def save(self, model_dir="./model_param"):
        self.model.write().overwrite().save(model_dir)

    def load(self, model_dir="./model_param"):
        self.model = ALSModel.load(model_dir)

    def recommend_for_all_users(self, num_items=10):
        user_recs = self.model.recommendForAllUsers(numItems=num_items)
        return user_recs

    def recommend_for_all_items(self, num_users=10):
        item_recs = self.model.recommendForAllItems(numUsers=num_users)
        return item_recs

    def recommend_for_user_subset(self, dataset, num_items=10):
        user_recs = self.model.recommendForUserSubset(dataset=dataset, numItems=num_items)
        return user_recs

    def recommend_for_item_subset(self, dataset, num_users=10):
        item_recs = self.model.recommendForItemSubset(dataset=dataset, numUsers=num_users)
        return item_recs

