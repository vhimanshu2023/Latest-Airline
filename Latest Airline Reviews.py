# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr , to_date
spark = SparkSession.builder.appName('Latest Airline Reviews').getOrCreate()

# COMMAND ----------

from pyspark.sql.types import StructField , StructType , IntegerType , StringType , DateType , FloatType

Schema = StructType([
    StructField('_c0',IntegerType(),True),
    StructField('recorded_date',StringType(),False),
    StructField('airline',StringType(),True),
    StructField('user_name',StringType(),True),
    StructField('rating',IntegerType(),True),
    StructField('experience',StringType(),True),
    StructField('review_date',StringType(),True),
    # StructField('review',StringType(),True),
    StructField('_c07',IntegerType(),True),
    StructField('type_of_traveller',StringType(),True),
    StructField('seat_type',StringType(),True),
    StructField('route',StringType(),True),
    StructField('date_flown',StringType(),True),
    StructField('seat_comfort',FloatType(),True),
    StructField('cabin_staff_service',FloatType(),True),
    StructField('ground_service',FloatType(),True),
    StructField('value_for_money',IntegerType(),True),
    # StructField('recommended',StringType(),True),
    StructField('food_beverages',FloatType(),True),    
    StructField('inflight_entertainment',FloatType(),True),    
    StructField('wifi_connectivity',FloatType(),True),    
    StructField('aircraft',StringType(),True),
    StructField('recommended',StringType(),True),
    StructField('review',StringType(),True)
])

airline = spark.read.option("header",True).option('mode','permissive').schema(Schema).csv('/FileStore/All_Airline_Reviews_Data.csv')

airline.display()

# COMMAND ----------

from pyspark.sql.functions import to_date , col

airline = airline.withColumn('recorded_date',to_date(col('recorded_date'),'dd-MM-yyyy'))

# COMMAND ----------

airline = airline.dropna(how='any',subset=['recorded_date'])

# COMMAND ----------

airline.display()

# COMMAND ----------

airline = airline.drop('_c07')

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Data cleaning process
from pyspark.sql.functions import expr , to_date

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

airline = airline.withColumn('review_date',expr("regexp_replace(review_date, '([0-9]+)(st|nd|rd|th)', '$1')"))

# convert the preprocess date string to date format
airline = airline.withColumn('review_date',to_date("review_date","dd MMMM yyyy"))

# show the result
airline.display()

# COMMAND ----------

airline = airline.fillna({
    "route" : "unknown",
    "seat_comfort" : 0,
    "cabin_staff_service" : 0,
    "ground_service" : 0,
    "value_for_money" : 0,
    "food_beverages" : 0,
    "inflight_entertainment" : 0,
    "wifi_connectivity" : 0,
    "aircraft" : "unknown"
})

# COMMAND ----------

from pyspark.sql.functions import split,udf,col

airline = airline.withColumn('verification',split(col('review'),'\|')[0])
airline = airline.withColumn('review',split(col('review'),'\|')[1])

airline.display()

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

# Define UDF to extract start of route
def startroute(route):
    if route:
        start = route.split(' to ')[0]
        return start
    else:
        return "unknown"

start = udf(startroute, StringType())

# Apply UDF to create 'start' column
airline = airline.withColumn("start", start(col('route')))

#---------------------------------------------------------------------------

# COMMAND ----------

# Define UDF to extract end of route
def endroute(route):
    if " to " in route:        
        end = route.split(' to ')[1]
        if end:
            ends = end.split(' via ')[0]
            return ends
        else:
            return "unknown"
    else:
        return "unknown"

end = udf(endroute, StringType())
airline = airline.withColumn("End", end(col('route')))

#---------------------------------------------------------------------------

# Define UDF to extract via of route
def via(route):
    if "via" in route:
        via_str = route.split(' via ')[1]
        return via_str
    else:
        return "unknown"

throw = udf(via, StringType())
airline = airline.withColumn("via", throw(col('route')))

airline.display()

# COMMAND ----------

traveller = ['Couple Leisure','Family Leisure','Solo Leisure','Business']
airline = airline.filter(col("type_of_traveller").isin(traveller))

airline.display()

# COMMAND ----------

# DBTITLE 1,Top 5 route with highest average rating
from pyspark.sql.functions import avg,col,desc,round
# Top 5 route with highest average rating
top_route = airline.groupBy('route').agg(avg(col('rating')).alias('highest_average_rating')).orderBy(desc('highest_average_rating'))
top_route.display(5)

# Top 5 airline with highest average rating
top_airline = airline.groupBy('airline').agg(avg(col('rating')).alias('highest_average_rating')).orderBy(desc('highest_average_rating'))
top_airline.display(5)

# COMMAND ----------

# DBTITLE 1,Average rating with AirCraft
aircraft = airline.groupBy('aircraft').agg(
    round(avg(col('rating'))).alias('overall_rating'),
    round(avg(col('seat_comfort'))).alias('avg_rating_seat_comfort'),
    round(avg(col('cabin_staff_service'))).alias('avg_rating_cabin_staff_service'),
    round(avg(col('food_beverages'))).alias('avg_rating_food_beverages'),
    round(avg(col('inflight_entertainment'))).alias('avg_rating_inflight_entertainment'),
    round(avg(col('ground_service'))).alias('avg_rating_ground_service'),
    round(avg(col('wifi_connectivity'))).alias('avg_rating_wifi_connectivity')
    )
aircraft.display()

# COMMAND ----------

# DBTITLE 1,customer segmentation
selected_column = ['type_of_traveller','seat_comfort','cabin_staff_service','food_beverages','ground_service','value_for_money']

customers = airline.select(*selected_column)

customers.display()


# COMMAND ----------

# DBTITLE 1,Average rating by Type of traveller
traveller = airline.groupBy('type_of_traveller').agg(
    round(avg(col('seat_comfort'))).alias('avg_rating_seat_comfort'),
    round(avg(col('cabin_staff_service'))).alias('avg_rating_cabin_staff_service'),
    round(avg(col('food_beverages'))).alias('avg_rating_food_beverages'),
    round(avg(col('ground_service'))).alias('avg_rating_ground_service'),
    round(avg(col('value_for_money'))).alias('avg_rating_value_for_money')
)
traveller.display()


# COMMAND ----------

# DBTITLE 1,Sentiment Analysis
from pyspark.sql.functions import when,col

airline = airline.withColumn('recommended',when(col('recommended')=='yes','Recommended').otherwise('Not Recommended'))

airline.groupBy('recommended').count().display()

airline.show()

# COMMAND ----------

# pip install textblob

# COMMAND ----------

from textblob import TextBlob
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

def sentiment(review):
    if review is not None:  # Check if review is not None
        analysis = TextBlob(review)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    else:
        return None  # Return None for null reviews

rule = udf(sentiment, StringType())

airline = airline.withColumn('sentiment', rule(col('review')))


airline.groupBy('sentiment').count().display()
airline.display()
