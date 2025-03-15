from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Initialize Spark Session
spark = (SparkSession.builder.appName("Pyspark Tutorial")
         .config("spark.memory.offHeap.enabled", "true")
         .config("spark.memory.offHeap.size", "10g")
         .getOrCreate())

# Read data
df = spark.read.csv("C:/Users/DuccHuyy/Desktop/BigData/OnlineRetail.csv", header=True, escape="\"")

# Remove canceled invoices
df = df.filter(~col("InvoiceNo").startswith("C"))

# Convert date column to timestamp
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn('date', to_timestamp("InvoiceDate", 'yy-MM-dd HH:mm:ss'))

# Define reference date
df = df.withColumn("from_date", to_timestamp(lit("2010-12-01 08:26:00"), 'yy-MM-dd HH:mm'))

# Calculate recency
df = df.withColumn('recency', col("date").cast("long") - col('from_date').cast("long"))
recency_df = df.groupBy('CustomerID').agg(max('recency').alias('recency'))

# Calculate frequency
df_freq = df.groupBy('CustomerID').agg(count('InvoiceNo').alias('frequency'))

# Calculate monetary value
df = df.withColumn('TotalAmount', col("Quantity") * col("UnitPrice"))
df_monetary = df.groupBy('CustomerID').agg(sum('TotalAmount').alias('monetary_value'))

# Merge all data
final_df = recency_df.join(df_freq, on='CustomerID', how='inner')
final_df = final_df.join(df_monetary, on='CustomerID', how='inner')

# Feature Scaling
assemble = VectorAssembler(inputCols=['recency', 'frequency', 'monetary_value'], outputCol='features')
assembled_data = assemble.transform(final_df)

scaler = StandardScaler(inputCol='features', outputCol='standardized')
data_scaled = scaler.fit(assembled_data).transform(assembled_data)

# Determine optimal K using Elbow Method
cost = np.zeros(10)
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', metricName='silhouette')

for i in tqdm(range(2, 10)):
    kmeans = KMeans(featuresCol='standardized', k=i)
    model = kmeans.fit(data_scaled)
    cost[i] = model.summary.trainingCost

# Plot Elbow Curve
df_cost = pd.DataFrame({'cluster': range(2, 10), 'cost': cost[2:]})
plt.plot(df_cost.cluster, df_cost.cost)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Apply KMeans with optimal K (assuming K=4 based on Elbow Curve)
kmeans = KMeans(featuresCol='standardized', k=4)
kmeans_model = kmeans.fit(data_scaled)
preds = kmeans_model.transform(data_scaled)

# Convert to Pandas for Visualization
df_viz = preds.select('recency', 'frequency', 'monetary_value', 'prediction').toPandas()
avg_df = df_viz.groupby(['prediction'], as_index=False).mean()

# Plot Cluster Characteristics
features = ['recency', 'frequency', 'monetary_value']
for feature in features:
    sns.barplot(x='prediction', y=feature, data=avg_df)
    plt.show()
