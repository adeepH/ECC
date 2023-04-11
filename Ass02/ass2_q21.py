import requests
# Ignore all warnings 
import warnings 
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("NBA Shot logs Analysis").config("spark.driver.memory", "4g").getOrCreate()

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import avg, to_date, split, col, lit, desc, count, expr, udf, when, monotonically_increasing_id,collect_list, array_distinct, row_number
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField, StringType, IntegerType 

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.window import Window

df = spark.read.csv("shot_logs.csv", header=True, inferSchema=True)

#print(df.columns)
#df.show(10)

# columns
# ['GAME_ID', 'MATCHUP', 'LOCATION', 'W', 'FINAL_MARGIN', 'SHOT_NUMBER', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK',
#  'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'SHOT_RESULT', 'CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID',
#  'CLOSE_DEF_DIST', 'FGM', 'PTS', 'player_name', 'player_id']

# Based on 2.1, we retain, player_name, CLOSTEST_DEFENDER, and SHOT_RESULT
df = df[['player_name', 'CLOSEST_DEFENDER', 'SHOT_RESULT']]
df.df = df.dropna()

# Printing the unique types of shots in SHOT_RESULT
uniq = df.select('SHOT_RESULT').distinct()
#print(uniq.show())

# We encode SHOT_RESULT as a binary variable, 1 for made, and 0 for missed
df = df.withColumn("SHOT_RESULT", when(col("SHOT_RESULT") == 'missed', 0).otherwise(1))

#printing unique values after encoding 
#print(df.select('SHOT_RESULT').distinct().show())

# Grouping and ordering with the most shots made
sorted_df = df.groupBy(col('player_name'), col('CLOSEST_DEFENDER')).sum("SHOT_RESULT").orderBy(col("sum(SHOT_RESULT)"))#.show(10)
sorted_df = sorted_df.withColumnRenamed("sum(SHOT_RESULT)", "TOTAL_SHOT_COUNT")

print('player defender pairs with the total number of shots made, in an ascending order:')
sorted_df.show()
# Finding out for each player, who is the unwanted defender

"""
player_defender_list = sorted_df[['player_name', 'CLOSEST_DEFENDER']]

rows = sorted_df.collect()

schema = StructType([
    StructField("player_name", StringType(), True),
    StructField("unwanted_defender", StringType(), True)
    # Add shot score later
])

# Create an empty list or an empty RDD
empty_list = []
empty_rdd = spark.sparkContext.emptyRDD()

# Create a DataFrame with only the schema
unwanted_defender_df = spark.createDataFrame(empty_list, schema=schema)

for row in rows:
    player_name = row['player_name']
    filtered_df = unwanted_defender_df.filter(col("player_name") == player_name)

    if filtered_df.count() == 0:
        player_pair = (row['player_name'], row['CLOSEST_DEFENDER'])
        new_row = spark.createDataFrame([player_pair], schema=schema)
        unwanted_defender_df = unwanted_defender_df.union(new_row)
    else:
        pass


unwanted_defender_df.show()
"""

window_spec = Window.partitionBy(col("player_name"), col("CLOSEST_DEFENDER")).orderBy("TOTAL_SHOT_COUNT")

# Add a row number column to the DataFrame based on the window specification
df = sorted_df.withColumn("row_number", row_number().over(window_spec))

# Filter the DataFrame to get only the rows where row number is greater than 1
duplicate_rows = df.filter(col("row_number") > 1)

# Get the indexes of the duplicate rows
duplicate_indexes = [row["row_number"] - 1 for row in duplicate_rows.collect()]

# Drop the duplicate rows from the DataFrame
df = df.filter(col("row_number") == 1).drop("row_number")

# Show the resulting DataFrame
df.show()

# Save the DataFrame as CSV
df.write.csv("player_unwanted_defender_pairs.csv", header=True, mode="overwrite")