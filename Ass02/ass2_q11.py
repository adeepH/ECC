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

spark = SparkSession.builder.appName("NYC Open Data Analysis").getOrCreate()
df = spark.read.csv("NYC.csv", header=True, inferSchema=True)
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import avg, to_date, split, col, lit, desc, count

# columns

#'Summons Number', 'Plate ID', 'Registration State', 'Plate Type', 'Issue Date', 
# 'Violation Code', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1',
#  'Street Code2', 'Street Code3', 'Vehicle Expiration Date', 'Violation Location', 
# 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Issuer Command', 'Issuer Squad', 
# 'Violation Time', 'Time First Observed', 'Violation County', 'Violation In Front Of Or Opposite',
#  'House Number', 'Street Name', 'Intersecting Street', 'Date First Observed', 'Law Section', 'Sub Division',
#  'Violation Legal Code', 'Days Parking In Effect    ', 'From Hours In Effect', 'To Hours In Effect', '
# Vehicle Color', 'Unregistered Vehicle?', 'Vehicle Year', 'Meter Number', 'Feet From Curb', 
# 'Violation Post Code', 'Violation Description', 'No Standing or Stopping Violation', '
# Hydrant Violation', 'Double Parking Violation'

# • When are tickets most likely to be issued? (15 pts)
# • What are the most common years and types of cars to be ticketed? (15 pts)
# • Where are tickets most commonly issued? (15 pts)
# • Which color of the vehicle is most likely to get a ticket? (15 pts)

df = df[['Issue Date', 'Violation County', 'Vehicle Make', 'Vehicle Color',
         'Vehicle Body Type', 'Vehicle Year', 'Violation Time', 'Street Name','Violation Precinct']]
df = df.dropna()

# Dropping vehicles with vehicle year < 1900 (There are several 0 values, who knows how many more) 
threshold = 1900
# dropping all records that have a street codeX of 0
zip_code_threshold = 10^4
# As zipcode has to be six digits, we divide it by 10^6 and check if 
df = df.filter((col("Vehicle Year") >= threshold) &
                (col("Street Code1")//zip_code_threshold!=0) &
                  (col("Street Code2")//zip_code_threshold!=0) &
                    (col("Street Code3")//zip_code_threshold != 0))
#df = df.drop('Date')

# 1. When are tickets most likely to be issued?
date = df.groupBy('Issue Date').count().sort(desc('count'))
time = df.groupBy('Violation Time').count().sort(desc('count'))
#most_frequent_time = df.groupBy("Issue Date").count().orderBy(desc("count")).first()

#most_frequent_date = df.groupBy("Violation Time").count().orderBy(desc("count")).first()

print(f'The tickets are most likely to be issued on \n {time.first()} time of day')
print(f'The tickets are most likely to be issued on the date \n {date.first()}')
print('-----------------x------------x-------------------x-------------------x--')
# 2. What are the most common years and types of cars to be ticketed
# Vehicle year is
df2 = df[['Vehicle Body Type', 'Vehicle Year']]#.show(10))
#frequent_car_type = df2.groupBy('Vehicle Body Type','Vehicle Year').agg(count('*').alias('count')).orderBy(col('count').desc())#.show(1)
print(f'The most common years and types of cars to be ticketed are \n')
print(df.groupBy('Vehicle Body Type','Vehicle Year').agg(count('*').alias('count')).orderBy(col('count').desc()).show(1))

 
# 3.Where are tickets most commonly issued? 
print('The tickets are most commonly issued at\n')
frequent_location = df.groupBy(col('Street Name'), col('Violation County')).agg(count('*').alias('count')).orderBy(col('count').desc()).show(1)
#print(f'The most common years and types of cars to be ticketed are \n {frequent_car_type}')

# 4. Which color of the vehicle is most likely to get a ticket
color = df.groupBy('Vehicle Color').count().sort(desc('count'))
print(f'The most likely color of the vehicle to get a ticket is \n {color.first()}')
 
spark.stop()
