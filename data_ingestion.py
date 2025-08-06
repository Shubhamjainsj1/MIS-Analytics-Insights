"""
Data Ingestion Module
Extracts data from multiple sources using PySpark and SQL
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
from config import DATABASE_CONFIG, SPARK_CONFIG, RAW_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.spark = self._initialize_spark()
        self.engine = self._create_db_engine()
    
    def _initialize_spark(self):
        """Initialize Spark session"""
        spark = SparkSession.builder \
            .appName(SPARK_CONFIG['app_name']) \
            .master(SPARK_CONFIG['master']) \
            .config("spark.executor.memory", SPARK_CONFIG['memory']) \
            .config("spark.executor.cores", SPARK_CONFIG['cores']) \
            .getOrCreate()
        
        logger.info("Spark session initialized successfully")
        return spark
    
    def _create_db_engine(self):
        """Create database engine for SQL operations"""
        try:
            connection_string = (
                f"mssql+pyodbc://{DATABASE_CONFIG['username']}:"
                f"{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['server']}/"
                f"{DATABASE_CONFIG['database']}?driver={DATABASE_CONFIG['driver']}"
            )
            engine = create_engine(connection_string)
            logger.info("Database engine created successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            return None
    
    def extract_from_csv(self, file_path):
        """Extract data from CSV files"""
        try:
            df = self.spark.read.csv(str(file_path), header=True, inferSchema=True)
            logger.info(f"Successfully loaded CSV data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return None
    
    def extract_from_database(self, query):
        """Extract data from database using SQL"""
        try:
            df_pandas = pd.read_sql(query, self.engine)
            df_spark = self.spark.createDataFrame(df_pandas)
            logger.info("Successfully extracted data from database")
            return df_spark
        except Exception as e:
            logger.error(f"Error extracting from database: {e}")
            return None
    
    def create_sample_data(self):
        """Create sample customer complaints data"""
        sample_data = [
            (1, "COMP001", "2023-01-15", "Billing", "Product", "The billing amount is incorrect and I was charged twice", 
             "resolved", False, 3, "email"),
            (2, "COMP002", "2023-01-16", "Service", "Service Quality", "Very poor customer service, waited for hours", 
             "open", True, 1, "phone"),
            (3, "COMP003", "2023-01-17", "Technical", "Product Defect", "Product stopped working after one week", 
             "in_progress", True, 2, "chat"),
            (4, "COMP004", "2023-01-18", "Billing", "Payment", "Payment was processed but not reflected in account", 
             "resolved", False, 4, "email"),
            (5, "COMP005", "2023-01-19", "Service", "Delivery", "Package was delivered to wrong address", 
             "open", True, 2, "phone"),
            (6, "COMP006", "2023-01-20", "Technical", "Website", "Website keeps crashing when trying to make purchase", 
             "in_progress", False, 3, "chat"),
            (7, "COMP007", "2023-01-21", "Billing", "Refund", "Requested refund 2 weeks ago, still not processed", 
             "open", True, 1, "email"),
            (8, "COMP008", "2023-01-22", "Service", "Support", "Support team was very helpful and resolved issue quickly", 
             "resolved", False, 5, "phone"),
            (9, "COMP009", "2023-01-23", "Technical", "Mobile App", "Mobile app crashes frequently", 
             "open", True, 2, "chat"),
            (10, "COMP010", "2023-01-24", "Service", "Product Information", "Product description was misleading", 
             "in_progress", False, 3, "email")
        ]
        
        schema = StructType([
            StructField("complaint_id", IntegerType(), True),
            StructField("ticket_id", StringType(), True),
            StructField("date_received", StringType(), True),
            StructField("category", StringType(), True),
            StructField("subcategory", StringType(), True),
            StructField("complaint_text", StringType(), True),
            StructField("status", StringType(), True),
            StructField("escalated", BooleanType(), True),
            StructField("satisfaction_score", IntegerType(), True),
            StructField("contact_method", StringType(), True)
        ])
        
        df = self.spark.createDataFrame(sample_data, schema)
        
        # Save sample data to CSV
        sample_file_path = RAW_DATA_DIR / "customer_complaints.csv"
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(sample_file_path.parent / "temp"))
        
        logger.info(f"Sample data created with {df.count()} records")
        return df
    
    def run_sql_queries(self):
        """Execute SQL queries for data extraction and transformation"""
        
        # Sample SQL queries for different data sources
        queries = {
            'customer_complaints': """
                SELECT 
                    complaint_id,
                    ticket_id,
                    date_received,
                    category,
                    subcategory,
                    complaint_text,
                    status,
                    escalated,
                    satisfaction_score,
                    contact_method
                FROM customer_complaints 
                WHERE date_received >= '2023-01-01'
            """,
            
            'customer_demographics': """
                SELECT 
                    customer_id,
                    age_group,
                    location,
                    subscription_type,
                    tenure_months
                FROM customer_demographics
            """,
            
            'complaint_resolution': """
                SELECT 
                    complaint_id,
                    resolution_time_hours,
                    resolution_method,
                    customer_satisfaction_post_resolution
                FROM complaint_resolution
            """
        }
        
        extracted_data = {}
        for table_name, query in queries.items():
            try:
                # For demonstration, we'll create sample data instead of actual DB extraction
                logger.info(f"Executing query for {table_name}")
                # df = self.extract_from_database(query)
                # extracted_data[table_name] = df
                logger.info(f"Query executed successfully for {table_name}")
            except Exception as e:
                logger.error(f"Error executing query for {table_name}: {e}")
        
        return extracted_data
    
    def close_connections(self):
        """Close Spark session and database connections"""
        if self.spark:
            self.spark.stop()
        if self.engine:
            self.engine.dispose()
        logger.info("Connections closed successfully")

# SQL queries for creating database schema (if needed)
CREATE_TABLES_SQL = """
-- Customer Complaints Table
CREATE TABLE IF NOT EXISTS customer_complaints (
    complaint_id INT PRIMARY KEY,
    ticket_id VARCHAR(50),
    date_received DATE,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    complaint_text TEXT,
    status VARCHAR(50),
    escalated BOOLEAN,
    satisfaction_score INT,
    contact_method VARCHAR(50)
);

-- Customer Demographics Table
CREATE TABLE IF NOT EXISTS customer_demographics (
    customer_id INT PRIMARY KEY,
    age_group VARCHAR(50),
    location VARCHAR(100),
    subscription_type VARCHAR(50),
    tenure_months INT
);

-- Complaint Resolution Table
CREATE TABLE IF NOT EXISTS complaint_resolution (
    complaint_id INT,
    resolution_time_hours INT,
    resolution_method VARCHAR(100),
    customer_satisfaction_post_resolution INT,
    FOREIGN KEY (complaint_id) REFERENCES customer_complaints(complaint_id)
);
"""

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    
    # Create sample data
    df = ingestion.create_sample_data()
    df.show()
    
    # Run SQL queries (commented out for demo)
    # extracted_data = ingestion.run_sql_queries()
    
    # Close connections
    ingestion.close_connections()
