
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('rec').getOrCreate()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
data = spark.read.csv('movielens_ratings.csv',inferSchema=True,header=True)
data.head()
data.describe().show()
(training, test) = data.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

predictions = model.transform(test)
print("Predictions")
predictions.show()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

single_user = test.filter(test['userId']==11).select(['movieId','userId'])

single_user.show()
reccomendations = model.transform(single_user)
reccomendations.orderBy('prediction',ascending=False).show()

print("Accurcy is " ,end=" ")
x =100-rmse
print(x ,'%')










