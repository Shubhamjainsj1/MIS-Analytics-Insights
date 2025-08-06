"""
Sentiment Analysis Module
Performs sentiment analysis on customer complaints using NLP techniques
"""

import logging
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from config import MODELS_DIR, OUTPUTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_model = None
        self.vectorizer = None
        
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        if not text or pd.isna(text):
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if not text or pd.isna(text):
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0, 'sentiment': 'neutral'}
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        
        # Classify sentiment based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'compound': compound,
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg'],
            'sentiment': sentiment
        }
    
    def create_sentiment_labels(self, df, text_column='clean_complaint_text'):
        """Create sentiment labels for training data"""
        
        # Apply TextBlob sentiment analysis
        textblob_results = df[text_column].apply(self.analyze_sentiment_textblob)
        df['textblob_polarity'] = textblob_results.apply(lambda x: x['polarity'])
        df['textblob_subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
        df['textblob_sentiment'] = textblob_results.apply(lambda x: x['sentiment'])
        
        # Apply VADER sentiment analysis
        vader_results = df[text_column].apply(self.analyze_sentiment_vader)
        df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
        df['vader_positive'] = vader_results.apply(lambda x: x['positive'])
        df['vader_neutral'] = vader_results.apply(lambda x: x['neutral'])
        df['vader_negative'] = vader_results.apply(lambda x: x['negative'])
        df['vader_sentiment'] = vader_results.apply(lambda x: x['sentiment'])
        
        # Create ensemble sentiment (combination of TextBlob and VADER)
        def ensemble_sentiment(row):
            textblob_sent = row['textblob_sentiment']
            vader_sent = row['vader_sentiment']
            
            # If both agree, use that sentiment
            if textblob_sent == vader_sent:
                return textblob_sent
            
            # If they disagree, use the one with stronger confidence
            if abs(row['textblob_polarity']) > abs(row['vader_compound']):
                return textblob_sent
            else:
                return vader_sent
        
        df['ensemble_sentiment'] = df.apply(ensemble_sentiment, axis=1)
        
        logger.info("Sentiment labels created using TextBlob and VADER")
        return df
    
    def train_sentiment_classifier(self, df, text_column='clean_complaint_text', target_column='ensemble_sentiment'):
        """Train a custom sentiment classifier"""
        
        # Prepare data
        X = df[text_column].fillna('')
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with TF-IDF and classifier
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Try different classifiers
        classifiers = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, classifier in classifiers.items():
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', classifier)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'model': pipeline
            }
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = pipeline
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Save best model
        self.sentiment_model = best_model
        model_path = MODELS_DIR / "sentiment_classifier.pkl"
        joblib.dump(best_model, model_path)
        
        logger.info(f"Best sentiment model saved with accuracy: {best_score:.4f}")
        return results
    
    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        if self.sentiment_model is None:
            logger.error("No sentiment model loaded. Please train or load a model first.")
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.sentiment_model.predict(texts)
        probabilities = self.sentiment_model.predict_proba(texts)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'predicted_sentiment': predictions[i],
                'confidence': max(probabilities[i]),
                'probabilities': dict(zip(self.sentiment_model.classes_, probabilities[i]))
            })
        
        return results
    
    def load_sentiment_model(self, model_path=None):
        """Load pre-trained sentiment model"""
        if model_path is None:
            model_path = MODELS_DIR / "sentiment_classifier.pkl"
        
        try:
            self.sentiment_model = joblib.load(model_path)
            logger.info(f"Sentiment model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
    
    def analyze_sentiment_distribution(self, df, sentiment_column='ensemble_sentiment'):
        """Analyze sentiment distribution and create visualizations"""
        
        # Sentiment distribution
        sentiment_counts = df[sentiment_column].value_counts()
        sentiment_percentages = df[sentiment_column].value_counts(normalize=True) * 100
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Sentiment Distribution')
        
        # Bar chart
        sentiment_counts.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Sentiment Counts')
        axes[1].set_xlabel('Sentiment')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sentiment_counts, sentiment_percentages
    
    def create_sentiment_wordclouds(self, df, text_column='clean_complaint_text', sentiment_column='ensemble_sentiment'):
        """Create word clouds for different sentiments"""
        
        sentiments = df[sentiment_column].unique()
        
        fig, axes = plt.subplots(1, len(sentiments), figsize=(5*len(sentiments), 5))
        if len(sentiments) == 1:
            axes = [axes]
        
        for i, sentiment in enumerate(sentiments):
            # Get text for specific sentiment
            sentiment_text = ' '.join(df[df[sentiment_column] == sentiment][text_column].dropna())
            
            if sentiment_text.strip():
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(sentiment_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.title()} Sentiment Words')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment} text found', 
                           horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(f'{sentiment.title()} Sentiment Words')
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def sentiment_by_category(self, df, category_column='category', sentiment_column='ensemble_sentiment'):
        """Analyze sentiment by complaint category"""
        
        # Create cross-tabulation
        sentiment_by_cat = pd.crosstab(df[category_column], df[sentiment_column], normalize='index') * 100
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(sentiment_by_cat, annot=True, fmt='.1f', cmap='RdYlBu_r')
        plt.title('Sentiment Distribution by Complaint Category (%)')
        plt.xlabel('Sentiment')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'sentiment_by_category.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sentiment_by_cat
    
    def temporal_sentiment_analysis(self, df, date_column='date_received', sentiment_column='ensemble_sentiment'):
        """Analyze sentiment trends over time"""
        
        # Convert date column
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and sentiment
        daily_sentiment = df.groupby([df[date_column].dt.date, sentiment_column]).size().unstack(fill_value=0)
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        for sentiment in daily_sentiment_pct.columns:
            plt.plot(daily_sentiment_pct.index, daily_sentiment_pct[sentiment], 
                    marker='o', label=sentiment.title())
        
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Percentage')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'sentiment_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return daily_sentiment_pct
    
    def get_sentiment_insights(self, df, text_column='clean_complaint_text', sentiment_column='ensemble_sentiment'):
        """Generate comprehensive sentiment insights"""
        
        insights = {}
        
        # Overall sentiment distribution
        sentiment_dist = df[sentiment_column].value_counts()
        insights['overall_distribution'] = sentiment_dist.to_dict()
        
        # Average sentiment scores
        if 'textblob_polarity' in df.columns:
            insights['average_polarity'] = df['textblob_polarity'].mean()
        if 'vader_compound' in df.columns:
            insights['average_vader_compound'] = df['vader_compound'].mean()
        
        # Most negative and positive complaints
        if 'textblob_polarity' in df.columns:
            most_negative_idx = df['textblob_polarity'].idxmin()
            most_positive_idx = df['textblob_polarity'].idxmax()
            
            insights['most_negative_complaint'] = {
                'text': df.loc[most_negative_idx, text_column],
                'polarity': df.loc[most_negative_idx, 'textblob_polarity']
            }
            
            insights['most_positive_complaint'] = {
                'text': df.loc[most_positive_idx, text_column],
                'polarity': df.loc[most_positive_idx, 'textblob_polarity']
            }
        
        # Sentiment by satisfaction score correlation
        if 'satisfaction_score' in df.columns:
            correlation = df.groupby(sentiment_column)['satisfaction_score'].mean()
            insights['sentiment_satisfaction_correlation'] = correlation.to_dict()
        
        # Escalation rate by sentiment
        if 'escalated' in df.columns:
            escalation_by_sentiment = df.groupby(sentiment_column)['escalated'].mean()
            insights['escalation_rate_by_sentiment'] = escalation_by_sentiment.to_dict()
        
        logger.info("Comprehensive sentiment insights generated")
        return insights

def analyze_sentiment_with_spark(spark_df, text_column='clean_complaint_text'):
    """Analyze sentiment using Spark DataFrame (for large datasets)"""
    
    # Define UDFs for sentiment analysis
    def textblob_sentiment(text):
        if not text:
            return ('neutral', 0.0, 0.0)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return (sentiment, float(polarity), float(subjectivity))
    
    # Register UDF
    sentiment_udf = udf(textblob_sentiment, StructType([
        StructField("sentiment", StringType(), True),
        StructField("polarity", DoubleType(), True),
        StructField("subjectivity", DoubleType(), True)
    ]))
    
    # Apply sentiment analysis
    df_with_sentiment = spark_df.withColumn("sentiment_analysis", sentiment_udf(col(text_column)))
    
    # Extract sentiment components
    df_with_sentiment = df_with_sentiment.select(
        "*",
        col("sentiment_analysis.sentiment").alias("sentiment"),
        col("sentiment_analysis.polarity").alias("polarity"),
        col("sentiment_analysis.subjectivity").alias("subjectivity")
    ).drop("sentiment_analysis")
    
    return df_with_sentiment

if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from data_preprocessing import DataPreprocessor
    
    # Initialize components
    ingestion = DataIngestion()
    preprocessor = DataPreprocessor(ingestion.spark)
    analyzer = SentimentAnalyzer()
    
    # Create and preprocess sample data
    df_spark = ingestion.create_sample_data()
    processed_df, models = preprocessor.preprocess_pipeline(df_spark)
    
    # Convert to pandas for sentiment analysis
    df_pandas = processed_df.select('complaint_id', 'clean_complaint_text', 'category', 
                                   'satisfaction_score', 'escalated', 'date_received').toPandas()
    
    # Perform sentiment analysis
    df_with_sentiment = analyzer.create_sentiment_labels(df_pandas)
    
    # Train sentiment classifier
    training_results = analyzer.train_sentiment_classifier(df_with_sentiment)
    
    # Generate insights and visualizations
    sentiment_dist, sentiment_pct = analyzer.analyze_sentiment_distribution(df_with_sentiment)
    analyzer.create_sentiment_wordclouds(df_with_sentiment)
    sentiment_by_cat = analyzer.sentiment_by_category(df_with_sentiment)
    
    # Get comprehensive insights
    insights = analyzer.get_sentiment_insights(df_with_sentiment)
    
    print("Sentiment Analysis Results:")
    print(f"Overall Distribution: {insights['overall_distribution']}")
    print(f"Average Polarity: {insights.get('average_polarity', 'N/A')}")
    print(f"Escalation Rate by Sentiment: {insights.get('escalation_rate_by_sentiment', 'N/A')}")
    
    # Close connections
    ingestion.close_connections()
