"""
Data Preprocessing Module
Cleans and transforms customer complaint data using PySpark
"""

import logging
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import nltk
from config import PROCESSED_DATA_DIR

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, spark_session=None):
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = SparkSession.builder \
                .appName("DataPreprocessing") \
                .master("local[*]") \
                .getOrCreate()
        
        # Define stop words
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                          'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                          'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                          'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                          'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                          'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                          'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                          'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                          'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                          'under', 'again', 'further', 'then', 'once']
    
    def clean_text_data(self, df, text_column='complaint_text'):
        """Clean and preprocess text data"""
        
        # UDF for text cleaning
        def clean_text(text):
            if text is None:
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        
        clean_text_udf = udf(clean_text, StringType())
        
        # Apply text cleaning
        df_cleaned = df.withColumn(f'clean_{text_column}', clean_text_udf(col(text_column)))
        
        logger.info(f"Text data cleaned for column: {text_column}")
        return df_cleaned
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Fill missing text with empty string
        df = df.fillna({'complaint_text': ''})
        
        # Fill missing categorical values with 'Unknown'
        categorical_cols = ['category', 'subcategory', 'status', 'contact_method']
        for col_name in categorical_cols:
            if col_name in df.columns:
                df = df.fillna({col_name: 'Unknown'})
        
        # Fill missing numerical values with median or mean
        if 'satisfaction_score' in df.columns:
            # Calculate median satisfaction score
            median_score = df.select(median('satisfaction_score')).collect()[0][0]
            df = df.fillna({'satisfaction_score': median_score})
        
        # Fill missing boolean values
        if 'escalated' in df.columns:
            df = df.fillna({'escalated': False})
        
        logger.info("Missing values handled successfully")
        return df
    
    def extract_text_features(self, df, text_column='clean_complaint_text'):
        """Extract features from text using NLP techniques"""
        
        # Tokenization
        tokenizer = Tokenizer(inputCol=text_column, outputCol="words")
        
        # Remove stop words
        stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        stop_words_remover.setStopWords(self.stop_words)
        
        # Count Vectorization
        count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", 
                                         vocabSize=1000, minDF=2.0)
        
        # TF-IDF
        idf = IDF(inputCol="raw_features", outputCol="text_features")
        
        # Create pipeline for text processing
        text_pipeline = Pipeline(stages=[tokenizer, stop_words_remover, count_vectorizer, idf])
        
        # Fit and transform
        text_model = text_pipeline.fit(df)
        df_with_features = text_model.transform(df)
        
        logger.info("Text features extracted successfully")
        return df_with_features, text_model
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        
        categorical_columns = ['category', 'subcategory', 'status', 'contact_method']
        indexers = []
        encoders = []
        
        for col_name in categorical_columns:
            if col_name in df.columns:
                # String indexing
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
                indexers.append(indexer)
                
                # One-hot encoding
                encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_encoded")
                encoders.append(encoder)
        
        # Create pipeline for categorical encoding
        cat_pipeline = Pipeline(stages=indexers + encoders)
        
        # Fit and transform
        cat_model = cat_pipeline.fit(df)
        df_encoded = cat_model.transform(df)
        
        logger.info("Categorical features encoded successfully")
        return df_encoded, cat_model
    
    def create_feature_vector(self, df):
        """Create final feature vector for machine learning"""
        
        # Define feature columns
        feature_cols = []
        
        # Text features
        if 'text_features' in df.columns:
            feature_cols.append('text_features')
        
        # Encoded categorical features
        categorical_features = ['category_encoded', 'subcategory_encoded', 
                              'status_encoded', 'contact_method_encoded']
        for col_name in categorical_features:
            if col_name in df.columns:
                feature_cols.append(col_name)
        
        # Numerical features
        numerical_features = ['satisfaction_score']
        for col_name in numerical_features:
            if col_name in df.columns:
                feature_cols.append(col_name)
        
        # Vector assembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_final = assembler.transform(df)
        
        logger.info("Feature vector created successfully")
        return df_final
    
    def add_derived_features(self, df, text_column='clean_complaint_text'):
        """Add derived features from text and other data"""
        
        # Text length
        df = df.withColumn('text_length', length(col(text_column)))
        
        # Word count
        df = df.withColumn('word_count', size(split(col(text_column), ' ')))
        
        # Exclamation marks count (indicator of urgency)
        exclamation_udf = udf(lambda text: text.count('!') if text else 0, IntegerType())
        df = df.withColumn('exclamation_count', exclamation_udf(col('complaint_text')))
        
        # Capital letters ratio (indicator of anger/frustration)
        def capital_ratio(text):
            if not text or len(text) == 0:
                return 0.0
            capitals = sum(1 for c in text if c.isupper())
            return float(capitals) / len(text)
        
        capital_ratio_udf = udf(capital_ratio, DoubleType())
        df = df.withColumn('capital_ratio', capital_ratio_udf(col('complaint_text')))
        
        # Day of week from date
        df = df.withColumn('date_received', to_date(col('date_received')))
        df = df.withColumn('day_of_week', dayofweek(col('date_received')))
        
        # Month extraction
        df = df.withColumn('month', month(col('date_received')))
        
        logger.info("Derived features added successfully")
        return df
    
    def detect_outliers(self, df, column='satisfaction_score'):
        """Detect outliers using IQR method"""
        
        if column not in df.columns:
            return df
        
        # Calculate quartiles
        quantiles = df.select(
            expr(f'percentile_approx({column}, 0.25)').alias('q1'),
            expr(f'percentile_approx({column}, 0.75)').alias('q3')
        ).collect()[0]
        
        q1, q3 = quantiles['q1'], quantiles['q3']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Mark outliers
        df = df.withColumn(
            f'{column}_outlier',
            when((col(column) < lower_bound) | (col(column) > upper_bound), True).otherwise(False)
        )
        
        outlier_count = df.filter(col(f'{column}_outlier') == True).count()
        logger.info(f"Detected {outlier_count} outliers in {column}")
        
        return df
    
    def preprocess_pipeline(self, df):
        """Complete preprocessing pipeline"""
        
        logger.info("Starting data preprocessing pipeline")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Clean text data
        df = self.clean_text_data(df)
        
        # Step 3: Add derived features
        df = self.add_derived_features(df)
        
        # Step 4: Detect outliers
        df = self.detect_outliers(df, 'satisfaction_score')
        
        # Step 5: Extract text features
        df, text_model = self.extract_text_features(df)
        
        # Step 6: Encode categorical features
        df, cat_model = self.encode_categorical_features(df)
        
        # Step 7: Create final feature vector
        df = self.create_feature_vector(df)
        
        # Save processed data
        output_path = PROCESSED_DATA_DIR / "processed_complaints"
        df.coalesce(1).write.mode("overwrite").parquet(str(output_path))
        
        logger.info("Data preprocessing pipeline completed successfully")
        
        return df, {'text_model': text_model, 'categorical_model': cat_model}
    
    def get_data_quality_report(self, df):
        """Generate data quality report"""
        
        total_rows = df.count()
        total_cols = len(df.columns)
        
        # Missing values per column
        missing_counts = {}
        for col_name in df.columns:
            missing_count = df.filter(col(col_name).isNull()).count()
            missing_counts[col_name] = {
                'count': missing_count,
                'percentage': (missing_count / total_rows) * 100
            }
        
        # Data types
        data_types = {col_name: col_type for col_name, col_type in df.dtypes}
        
        # Basic statistics for numerical columns
        numerical_stats = {}
        numerical_columns = [col_name for col_name, col_type in df.dtypes 
                           if col_type in ['int', 'bigint', 'float', 'double']]
        
        if numerical_columns:
            stats_df = df.select(numerical_columns).describe()
            numerical_stats = {row['summary']: {col: row[col] for col in numerical_columns} 
                             for row in stats_df.collect()}
        
        report = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_counts,
            'data_types': data_types,
            'numerical_statistics': numerical_stats
        }
        
        logger.info(f"Data quality report generated for {total_rows} rows and {total_cols} columns")
        
        return report

if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    
    # Initialize components
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor(ingestion.spark)
    
    # Create sample data
    df = ingestion.create_sample_data()
    
    # Generate data quality report
    quality_report = preprocessor.get_data_quality_report(df)
    print("Data Quality Report:")
    print(f"Total Rows: {quality_report['total_rows']}")
    print(f"Total Columns: {quality_report['total_columns']}")
    
    # Run preprocessing pipeline
    processed_df, models = preprocessor.preprocess_pipeline(df)
    
    # Show processed data sample
    processed_df.select('complaint_id', 'clean_complaint_text', 'text_length', 'word_count').show(5, truncate=False)
    
    # Close connections
    ingestion.close_connections()
