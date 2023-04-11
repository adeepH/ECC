import requests
# Ignore all warnings 
import warnings 
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NBA Shot logs Analysis").config("spark.driver.memory", "4g").getOrCreate()

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import avg, to_date, split, col, lit, desc, count, expr, udf, when, monotonically_increasing_id,collect_list, array_distinct
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField, StringType, IntegerType 

from pyspark.ml.evaluation import ClusteringEvaluator

df = spark.read.csv("shot_logs.csv", header=True, inferSchema=True)
df = df.dropna()

#print(df.columns)
#df.show(10)

# columns
# ['GAME_ID', 'MATCHUP', 'LOCATION', 'W', 'FINAL_MARGIN', 'SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK',
#  'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'SHOT_RESULT', 'CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID',
#  'CLOSE_DEF_DIST', 'FGM', 'PTS', 'player_name', 'player_id']

# Based on 2.1, we retain, player_name, CLOSTEST_DEFENDER, and SHOT_RESULT
subset_df = df[['player_name','SHOT_DIST', 'CLOSE_DEF_DIST', 'SHOT_CLOCK']]

# Scale the numerical columns
assembler = VectorAssembler(inputCols=['SHOT_DIST', 'CLOSE_DEF_DIST', 'SHOT_CLOCK'], outputCol="comfortable_zone")
subset_df = assembler.transform(subset_df)
subset_df.show()

scaler = StandardScaler(inputCol="comfortable_zone", outputCol="scaled_comfortable_zone")
scaler_model = scaler.fit(subset_df)
subset_df = scaler_model.transform(subset_df)

# Using a Kmeans model with 4 zones, k=4
kmeans = KMeans(featuresCol="scaled_comfortable_zone", k=4).setSeed(12)
kmeans_model = kmeans.fit(subset_df)
clustered_df = kmeans_model.transform(subset_df)

clustered_df.show()

# grouping by players and cluster count
groupby_clustered_df = clustered_df.groupBy("player_name", "prediction").agg(count("*").alias("cluster_counts"))
#groupby_clustered_df = groupby_clustered_df.orderBy(desc('cluster_counts'))
groupby_clustered_df.show()

# Finding most frequent comfortable zone for players
# 1) James Harden
name_list = ['james harden', 'chris paul', 'stephen curry', 'lebron james']

for name in name_list:
    subset = groupby_clustered_df[col("player_name")==name]
    subset = subset.orderBy(desc('cluster_counts'))
    subset.show()
    most_common_cluster = subset.select("prediction").first()[0]

    print(f'From the cluster counts, we observe that the most comfortable zone for {name} is Zone: {most_common_cluster}')