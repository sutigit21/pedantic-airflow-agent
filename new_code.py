import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
import random

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TwoTowerSampling") \
    .config("spark.sql.shuffle.partitions", "10000") \
    .getOrCreate()

# Load data (adjust path)
df = spark.read.parquet("user_entity_interaction_all.parquet")

# Filter positive interactions (purchases + ad clicks)
positive_df = df.filter(F.col("action_type").isin(["item-purchase", "display-ad-click"]))

# 1. Generate User Positive Entities
user_positives = positive_df.groupBy("user_id") \
    .agg(F.collect_set("entity_id").alias("positive_entities"))

# 2. Generate Entity Positive Users
entity_positives = positive_df.groupBy("entity_id") \
    .agg(F.collect_set("user_id").alias("positive_users"))

# 3. Generate Global Entity/User List (for random negatives)
all_entities = [row["entity_id"] for row in positive_df.select("entity_id").distinct().collect()]
all_users = [row["user_id"] for row in positive_df.select("user_id").distinct().collect()]

# Broadcast these for distributed access
all_entities_bc = spark.sparkContext.broadcast(set(all_entities))
all_users_bc = spark.sparkContext.broadcast(set(all_users))

# 4. Sampling Function
def sample_batch(batch_df):
    batch = batch_df.toPandas()
    batch_size = len(batch)
    
    # In-batch negatives
    all_batch_entities = set(batch['entity_id'])
    all_batch_users = set(batch['user_id'])
    
    results = []
    for _, row in batch.iterrows():
        user_id = row['user_id']
        entity_id = row['entity_id']
        user_pos = set(row['positive_entities'])
        entity_pos = set(row['positive_users'])
        
        # User→Entity Negatives
        in_batch_neg_entities = list(all_batch_entities - user_pos - {entity_id})
        random_neg_entities = random.sample(list(all_entities_bc.value - user_pos), 100)
        
        # Entity→User Negatives
        in_batch_neg_users = list(all_batch_users - entity_pos - {user_id})
        random_neg_users = random.sample(list(all_users_bc.value - entity_pos), 100)
        
        results.append({
            "user_id": user_id,
            "entity_id": entity_id,
            "u2e_in_batch_negs": in_batch_neg_entities[:500],  # Sample 500
            "u2e_random_negs": random_neg_entities,
            "e2u_in_batch_negs": in_batch_neg_users[:500],
            "e2u_random_negs": random_neg_users
        })
    
    return spark.createDataFrame(pd.DataFrame(results))

# 5. Generate Training Batches
batch_size = 1024  # Adjust based on memory
num_batches = 100000  # Total batches needed

# Create base positive pairs with metadata
base_data = positive_df.join(user_positives, "user_id") \
                       .join(entity_positives, "entity_id")

# Stratified sampling by entity frequency
window = Window.partitionBy("entity_id").orderBy(F.rand())
balanced_data = base_data.withColumn("rank", F.rank().over(window)) \
                        .filter(F.col("rank") <= 100)  # Max 100 samples per entity

# Generate batches
training_data = balanced_data.repartition(10000) \
    .groupBy("entity_id") \
    .applyInPandas(sample_batch, schema=balanced_data.schema) \
    .limit(num_batches * batch_size)

# 6. Save Training Data
training_data.write.parquet("training_batches/", mode="overwrite")

# Stop Spark
spark.stop()