import requests

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
url = "https://data.cityofnewyork.us/api/views/pvqr-7yc4/rows.csv?accessType=DOWNLOAD"
# Takes a crapload of time, download the data and load it manually
#response = requests.get(url)
#data = response.content.decode('utf-8')
#print(data)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NYC Data Analysis").config("spark.driver.memory", "4g").getOrCreate()

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import avg, to_date, split, col, lit, desc, count, expr, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import ClusteringEvaluator

df = spark.read.csv("NYC.csv", header=True, inferSchema=True)
#df = df.dropna()
df = df.filter(col("Vehicle Year") >= 1900)
df = df.filter(col("Street Code1") > 0)
df = df.filter(col("Street Code2") > 0)
df = df.filter(col("Street Code3") > 0)

# Filter the DataFrame to include only parking tickets for Black vehicles and illegal parking violations
#illegal_parking_codes = [20, 21, 23, 24, 27] # These are the violation codes for illegal parking 

subset_df = df.select(col("Vehicle Color"), col("Street Code1"), col("Street Code2"), col("Street Code3")).na.drop()
#subset_df.write.format('csv').save('subset.csv')
#subset_df = subset_df(col("Violation Code").isin(illegal_parking_codes))
# Indexing the colors 
color_indexer = StringIndexer(inputCol="Vehicle Color", outputCol="Indexed Color")

subset_df = color_indexer.fit(subset_df).transform(subset_df)
#subset_df = subset_df.drop('Vehicle Color')

# Scale the numerical columns
assembler = VectorAssembler(inputCols=["Street Code1", "Street Code2", "Street Code3", "Indexed Color"], outputCol="features")
subset_df = assembler.transform(subset_df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(subset_df)
subset_df = scaler_model.transform(subset_df)
 
# Apply K-means clustering to the data
# calculating the optimal cluster count using silhouette scores

K_max = 4
K_min = 2
silhouette_scores = [] 

for i in range(K_min, K_max+1):

    kmeans = KMeans(featuresCol="scaled_features", k=i).setSeed(12)
    kmeans_model = kmeans.fit(subset_df)
    predictions = kmeans_model.transform(subset_df)
    evaluator = ClusteringEvaluator()
    silhouette_score = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette_score)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
optimal_k = silhouette_scores.index(min(silhouette_scores))
optimal_k += K_min # The list should start with K_min
print('Based on Silhoutte Score Evaluation, the most optimal cluster count is \n', optimal_k)


# Fitting a KMeans model with optmial K
kmeans = KMeans(featuresCol="scaled_features", k=optimal_k).setSeed(12)
kmeans_model = kmeans.fit(subset_df)
clustered_df = kmeans_model.transform(subset_df)

#spark = SparkSession.builder.appName("NYC Data Analysis").getOrCreate()

#cluster_id = kmeans_model.transform(scaler_model.transform(assembler.transform(spark.createDataFrame([(34510, 10030, 34050, "Black")], ["Street Code1", "Street Code2", "Street Code3", "Vehicle Color"])))).select("cluster_id").collect()[0][0]
test_df = spark.createDataFrame([
    {'Street Code1':34510, 'Street Code2':10030, 'Street Code3':34050, 'Vehicle Color': "BLACK", "Indexed Color": 0.0}])
#print(test_df.schema)
#clustered_df.write.format('csv').save('clustered_df.csv')
#print('T')
#test_df.show() 


feature_vector = assembler.transform(test_df)
scaled_fv = scaler_model.transform(feature_vector)
cluster_id = kmeans_model.transform(scaled_fv).select("prediction").collect()[0][0]
predictions = kmeans_model.transform(scaled_fv) 
#cluster_probabilities = kmeans_model.transform(scaled_fv).select(col('prediction').alias('cluster'), 'probability')

#cluster_probabilities = kmeans_model.transform(predictions).select('prediction', 'probability')
 

#print(cluster_id)
#print(clustered_df.show())
#print(clustered_df.filter((col('Vehicle Color')== "BLACK")).count())
# Count the number of Black vehicles in the same cluster that received a ticket
black_count = clustered_df.filter((col("Vehicle Color") == "BLACK")).count()# & (col("cluster_id") == cluster_id) 
                                #  & (col("Street Code1") == 34510) &
                                #    (col("Street Code2")==10030) & (col("Street Code3") == 34050) )#.count()
total_count = clustered_df.count()
print('The probability of the ticketed black car in the cluster is:\n')
##print(black_count)
print(total_count) 
ticket_prob = black_count / total_count
print(ticket_prob)

spark.stop()