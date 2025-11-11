# advanced_statistical_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import shapiro, normaltest, chi2_contingency, pearsonr, spearmanr, kendalltau
import warnings
import requests
import io
import joblib
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class Config:
    """Configuration constants"""
    SAMPLE_SIZE = 200
    DEFAULT_TEST_SIZE = 0.2
    MAX_ROWS_FOR_ANALYSIS = 10000
    PLOT_HEIGHT = 600
    PLOT_WIDTH = 800
    MAX_FORECAST_PERIODS = 365

class DataLoader:
    """Handles data loading from multiple sources"""
    
    @staticmethod
    def load_from_url(url: str) -> pd.DataFrame:
        """Load data from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                return pd.read_csv(io.StringIO(response.text))
            elif url.endswith(('.xlsx', '.xls')):
                return pd.read_excel(io.BytesIO(response.content))
            else:
                return pd.read_csv(io.StringIO(response.text))
                
        except Exception as e:
            logger.error(f"Failed to load data from URL: {e}")
            raise ValueError(f"Error loading data from URL: {e}")
    
    @staticmethod
    def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
        except Exception as e:
            logger.error(f"Failed to load uploaded file: {e}")
            raise ValueError(f"Error loading file: {e}")
    
    @staticmethod
    def generate_sample_data(dataset_type: str = "mixed") -> pd.DataFrame:
        """Generate sample datasets for demonstration"""
        np.random.seed(42)
        
        generators = {
            "time_series": DataLoader._generate_time_series_data,
            "categorical": DataLoader._generate_categorical_data,
            "numerical": DataLoader._generate_numerical_data,
            "mixed": DataLoader._generate_mixed_data,
            "forecasting": DataLoader._generate_forecasting_data
        }
        
        generator = generators.get(dataset_type, DataLoader._generate_mixed_data)
        return generator(Config.SAMPLE_SIZE)
    
    @staticmethod
    def _generate_time_series_data(n_samples: int) -> pd.DataFrame:
        """Generate time series sample data"""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        trend = np.linspace(100, 150, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 5, n_samples)
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': trend + seasonal + noise,
            'temperature': np.random.normal(25, 5, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        })
    
    @staticmethod
    def _generate_forecasting_data(n_samples: int) -> pd.DataFrame:
        """Generate forecasting sample data with multiple patterns"""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Multiple time series with different patterns
        trend = np.linspace(100, 200, n_samples)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        cycle = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 90)
        noise = np.random.normal(0, 8, n_samples)
        
        return pd.DataFrame({
            'date': dates,
            'sales': trend + seasonal + cycle + noise,
            'temperature': np.random.normal(25, 5, n_samples),
            'advertising_spend': np.random.exponential(1000, n_samples),
            'holiday': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'weekday': [d.weekday() for d in dates]
        })
    
    @staticmethod
    def _generate_categorical_data(n_samples: int) -> pd.DataFrame:
        """Generate categorical-rich sample data"""
        return pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples),
            'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples)
        })
    
    @staticmethod
    def _generate_numerical_data(n_samples: int) -> pd.DataFrame:
        """Generate numerical-rich sample data"""
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.uniform(-5, 5, n_samples),
            'target_continuous': np.random.normal(50, 15, n_samples),
            'target_binary': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
    
    @staticmethod
    def _generate_mixed_data(n_samples: int) -> pd.DataFrame:
        """Generate mixed-type sample data"""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(1000, 200, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home'], n_samples),
            'customer_rating': np.random.uniform(1, 5, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })

class DataCleaner:
    """Handles comprehensive data cleaning operations"""
    
    @staticmethod
    def clean_data(df: pd.DataFrame, cleaning_options: dict) -> pd.DataFrame:
        """Apply comprehensive data cleaning based on options"""
        df_clean = df.copy()
        
        # Handle missing values
        if cleaning_options.get('handle_missing'):
            df_clean = DataCleaner._handle_missing_values(df_clean, cleaning_options)
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates'):
            df_clean = df_clean.drop_duplicates()
        
        # Handle outliers
        if cleaning_options.get('handle_outliers'):
            df_clean = DataCleaner._handle_outliers(df_clean, cleaning_options)
        
        # Convert data types
        if cleaning_options.get('convert_dtypes'):
            df_clean = DataCleaner._convert_data_types(df_clean)
        
        # Standardize text data
        if cleaning_options.get('standardize_text'):
            df_clean = DataCleaner._standardize_text_data(df_clean)
        
        # Feature engineering
        if cleaning_options.get('feature_engineering'):
            df_clean = DataCleaner._feature_engineering(df_clean)
        
        return df_clean
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame, options: dict) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        strategy = options.get('missing_strategy', 'mean')
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if strategy == 'remove':
            return df.dropna()
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        elif strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        # For categorical columns, fill with 'Unknown'
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    @staticmethod
    def _handle_outliers(df: pd.DataFrame, options: dict) -> pd.DataFrame:
        """Handle outliers based on strategy"""
        strategy = options.get('outlier_strategy', 'clip')
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif strategy == 'clip':
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            elif strategy == 'transform':
                df[col] = np.log1p(df[col] - df[col].min() + 1)
        
        return df
    
    @staticmethod
    def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types automatically"""
        # Convert to categorical if low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.05:  # Less than 5% unique values
                df[col] = df[col].astype('category')
        
        # Convert potential numeric columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        return df
    
    @staticmethod
    def _standardize_text_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text data"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
    
    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        # Extract date features
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        # Create interaction terms for highly correlated numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(correlations)):
                for j in range(i+1, len(correlations)):
                    if correlations.iloc[i, j] > 0.7:
                        col1, col2 = correlations.index[i], correlations.columns[j]
                        df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        return df

class DataPreprocessor:
    """Handles data preprocessing and validation"""
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, target_variable: str = None) -> pd.DataFrame:
        """Preprocess data for analysis"""
        df_processed = df.copy()
        
        # Auto-detect and convert date columns
        for col in df_processed.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col])
                except:
                    pass
        
        # Convert target variable if categorical
        if target_variable and df_processed[target_variable].dtype == 'object':
            le = LabelEncoder()
            df_processed[target_variable] = le.fit_transform(df_processed[target_variable])
        
        return df_processed
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, target_variable: str) -> tuple:
        """Prepare features for modeling"""
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        return X, y

class StatisticalTests:
    """Comprehensive statistical testing suite"""
    
    @staticmethod
    def normality_tests(df: pd.DataFrame, column: str) -> dict:
        """Perform normality tests"""
        data = df[column].dropna()
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = shapiro(data)
        
        # D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        return {
            'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'dagostino': {'statistic': dagostino_stat, 'p_value': dagostino_p},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p}
        }
    
    @staticmethod
    def correlation_tests(df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Perform correlation tests"""
        data = df[[col1, col2]].dropna()
        x = data[col1]
        y = data[col2]
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(x, y)
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(x, y)
        
        # Kendall's tau
        kendall_tau, kendall_p = kendalltau(x, y)
        
        return {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'kendall': {'correlation': kendall_tau, 'p_value': kendall_p}
        }
    
    @staticmethod
    def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Perform chi-square test of independence"""
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table,
            'expected_frequencies': expected
        }
    
    @staticmethod
    def t_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """Perform t-test between groups"""
        groups = df[categorical_col].unique()
        if len(groups) != 2:
            return {"error": "Categorical variable must have exactly 2 groups"}
        
        group1 = df[df[categorical_col] == groups[0]][numerical_col]
        group2 = df[df[categorical_col] == groups[1]][numerical_col]
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std()
        }
    
    @staticmethod
    def anova_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """Perform one-way ANOVA"""
        groups = [group for name, group in df.groupby(categorical_col)[numerical_col]]
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'group_means': df.groupby(categorical_col)[numerical_col].mean().to_dict(),
            'group_stds': df.groupby(categorical_col)[numerical_col].std().to_dict()
        }

class ForecastingEngine:
    """Time series forecasting engine"""
    
    @staticmethod
    def arima_forecast(series: pd.Series, periods: int, order: tuple = (1, 1, 1)) -> dict:
        """ARIMA forecasting"""
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            return {
                'model': 'ARIMA',
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'model_summary': fitted_model.summary(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def exponential_smoothing_forecast(series: pd.Series, periods: int, seasonal_periods: int = None) -> dict:
        """Exponential smoothing forecasting"""
        try:
            if seasonal_periods:
                model = ExponentialSmoothing(series, seasonal_periods=seasonal_periods, trend='add', seasonal='add')
            else:
                model = ExponentialSmoothing(series, trend='add')
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
            return {
                'model': 'Exponential Smoothing',
                'forecast': forecast,
                'smoothing_level': fitted_model.params['smoothing_level'],
                'smoothing_trend': fitted_model.params.get('smoothing_trend', None),
                'smoothing_seasonal': fitted_model.params.get('smoothing_seasonal', None),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def linear_regression_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> dict:
        """Linear regression based forecasting"""
        try:
            # Create time features
            df_forecast = df.copy()
            df_forecast['time_index'] = range(len(df_forecast))
            
            # Fit linear regression
            X = df_forecast[['time_index']]
            y = df_forecast[value_col]
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future predictions
            future_indices = np.arange(len(df_forecast), len(df_forecast) + periods).reshape(-1, 1)
            future_predictions = model.predict(future_indices)
            
            return {
                'model': 'Linear Regression',
                'forecast': future_predictions,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': model.score(X, y)
            }
        except Exception as e:
            return {'error': str(e)}

class PDFReportGenerator:
    """Generate SPSS-like PDF reports"""
    
    @staticmethod
    def create_spss_report(df: pd.DataFrame, analyses: dict, filename: str = "statistical_report.pdf"):
        """Create comprehensive SPSS-like PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Statistical Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Dataset Information
        story.append(Paragraph("Dataset Information", styles['Heading2']))
        dataset_info = [
            ["Total Cases", str(len(df))],
            ["Total Variables", str(len(df.columns))],
            ["Missing Values", str(df.isnull().sum().sum())],
            ["Date Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        dataset_table = Table(dataset_info, colWidths=[200, 200])
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 20))
        
        # Variable Information
        story.append(Paragraph("Variable Information", styles['Heading2']))
        var_info = [["Variable", "Type", "Missing", "Unique", "Mean/Mode"]]
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            if df[col].dtype in ['int64', 'float64']:
                central = f"{df[col].mean():.2f}"
            else:
                central = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            
            var_info.append([col, dtype, str(missing), str(unique), str(central)])
        
        var_table = Table(var_info, colWidths=[120, 80, 60, 60, 100])
        var_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(var_table)
        story.append(Spacer(1, 20))
        
        # Add analysis results
        for analysis_name, analysis_data in analyses.items():
            story.append(Paragraph(analysis_name, styles['Heading2']))
            
            if isinstance(analysis_data, dict):
                analysis_table_data = [["Statistic", "Value"]]
                for key, value in analysis_data.items():
                    if isinstance(value, (int, float)):
                        analysis_table_data.append([key, f"{value:.4f}"])
                    else:
                        analysis_table_data.append([key, str(value)])
                
                analysis_table = Table(analysis_table_data, colWidths=[200, 200])
                analysis_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(analysis_table)
            
            story.append(Spacer(1, 15))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class DataAnalyzer:
    """Analyzes dataset characteristics and quality"""
    
    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> dict:
        """Comprehensive dataset analysis"""
        return {
            'basic_info': DataAnalyzer._get_basic_info(df),
            'data_quality': DataAnalyzer._assess_data_quality(df),
            'type_analysis': DataAnalyzer._analyze_data_types(df),
            'patterns': DataAnalyzer._detect_patterns(df)
        }
    
    @staticmethod
    def _get_basic_info(df: pd.DataFrame) -> dict:
        """Get basic dataset information"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum()
        }
    
    @staticmethod
    def _assess_data_quality(df: pd.DataFrame) -> dict:
        """Assess data quality metrics"""
        missing_values = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        
        return {
            'missing_values_total': missing_values.sum(),
            'missing_values_percentage': (missing_values.sum() / total_cells) * 100,
            'columns_with_missing': (missing_values > 0).sum(),
            'completeness_score': ((len(df) - df.isnull().sum()) / len(df) * 100).mean(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    
    @staticmethod
    def _analyze_data_types(df: pd.DataFrame) -> dict:
        """Analyze data type distribution"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        return {
            'numerical_count': len(numerical_cols),
            'categorical_count': len(categorical_cols),
            'date_count': len(date_cols),
            'primary_type': DataAnalyzer._determine_primary_type(numerical_cols, categorical_cols, date_cols)
        }
    
    @staticmethod
    def _determine_primary_type(numerical_cols, categorical_cols, date_cols) -> str:
        """Determine primary dataset type"""
        if len(date_cols) > 0:
            return "Time Series"
        elif len(categorical_cols) > len(numerical_cols):
            return "Categorical"
        elif len(numerical_cols) > len(categorical_cols):
            return "Numerical"
        else:
            return "Mixed"
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> dict:
        """Detect data patterns"""
        return {
            'outlier_pattern': DataAnalyzer._assess_outliers(df),
            'correlation_strength': DataAnalyzer._assess_correlations(df),
            'temporal_patterns': DataAnalyzer._check_temporal_patterns(df)
        }
    
    @staticmethod
    def _assess_outliers(df: pd.DataFrame) -> str:
        """Assess outlier presence"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return "No numerical columns"
        
        outlier_percentage = DataAnalyzer._calculate_outlier_percentage(df, numerical_cols)
        
        if outlier_percentage > 10:
            return "High"
        elif outlier_percentage > 5:
            return "Moderate"
        else:
            return "Low"
    
    @staticmethod
    def _calculate_outlier_percentage(df: pd.DataFrame, numerical_cols) -> float:
        """Calculate percentage of outliers"""
        outlier_counts = []
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts.append(outliers)
        
        total_outliers = sum(outlier_counts)
        return (total_outliers / (len(df) * len(numerical_cols))) * 100
    
    @staticmethod
    def _assess_correlations(df: pd.DataFrame) -> str:
        """Assess correlation strength"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return "Insufficient numerical data"
        
        corr_matrix = df[numerical_cols].corr().abs()
        mean_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if mean_correlation > 0.7:
            return "Strong"
        elif mean_correlation > 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    @staticmethod
    def _check_temporal_patterns(df: pd.DataFrame) -> str:
        """Check for temporal patterns"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            return "No temporal data"
        
        for col in date_cols:
            if df[col].is_monotonic_increasing:
                return "Sequential"
        return "Non-sequential"

class AnalysisPhase(ABC):
    """Abstract base class for analysis phases"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    @abstractmethod
    def execute(self):
        """Execute the analysis phase"""
        pass

class DataCleaningPhase(AnalysisPhase):
    """Handles comprehensive data cleaning and preprocessing"""
    
    def execute(self):
        st.header("🧹 Data Cleaning & Preprocessing")
        
        st.subheader("Current Data Overview")
        self._display_data_overview()
        
        st.subheader("Cleaning Options")
        cleaning_options = self._get_cleaning_options()
        
        if st.button("Apply Data Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_df = DataCleaner.clean_data(self.df, cleaning_options)
                st.session_state.cleaned_df = cleaned_df
                st.success("✅ Data cleaning completed!")
                
                # Show cleaning results
                self._display_cleaning_results(cleaned_df)
        
        # Show cleaned data if available
        if 'cleaned_df' in st.session_state:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
    
    def _display_data_overview(self):
        """Display current data overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Original Rows", len(self.df))
        col1.metric("Original Columns", len(self.df.columns))
        
        missing_values = self.df.isnull().sum().sum()
        col2.metric("Missing Values", missing_values)
        col2.metric("Duplicate Rows", self.df.duplicated().sum())
        
        col3.metric("Numerical Columns", len(self.numerical_cols))
        col3.metric("Categorical Columns", len(self.categorical_cols))
        
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        col4.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
    
    def _get_cleaning_options(self) -> dict:
        """Get data cleaning options from user"""
        options = {}
        
        st.write("**Missing Value Handling**")
        col1, col2 = st.columns(2)
        
        with col1:
            options['handle_missing'] = st.checkbox("Handle missing values", value=True)
            if options['handle_missing']:
                options['missing_strategy'] = st.selectbox(
                    "Missing value strategy:",
                    ["mean", "median", "mode", "knn", "remove"]
                )
        
        with col2:
            options['remove_duplicates'] = st.checkbox("Remove duplicate rows", value=True)
            options['handle_outliers'] = st.checkbox("Handle outliers", value=True)
            if options['handle_outliers']:
                options['outlier_strategy'] = st.selectbox(
                    "Outlier handling strategy:",
                    ["clip", "remove", "transform"]
                )
        
        st.write("**Data Transformation**")
        col3, col4 = st.columns(2)
        
        with col3:
            options['convert_dtypes'] = st.checkbox("Optimize data types", value=True)
            options['standardize_text'] = st.checkbox("Standardize text data", value=True)
        
        with col4:
            options['feature_engineering'] = st.checkbox("Create new features", value=True)
        
        return options
    
    def _display_cleaning_results(self, cleaned_df):
        """Display cleaning results"""
        st.subheader("Cleaning Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Rows After Cleaning", len(cleaned_df))
        col2.metric("Missing Values Remaining", cleaned_df.isnull().sum().sum())
        col3.metric("Duplicates Removed", len(self.df) - len(cleaned_df))
        col4.metric("Memory Reduction", 
                   f"{(self.df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()) / 1024**2:.2f} MB")
        
        # Show data type changes
        st.write("**Data Type Changes:**")
        dtype_changes = pd.DataFrame({
            'Column': self.df.columns,
            'Original Type': self.df.dtypes,
            'New Type': cleaned_df.dtypes
        })
        st.dataframe(dtype_changes, use_container_width=True)

class StatisticalAnalysisPhase(AnalysisPhase):
    """Comprehensive statistical analysis with SPSS-like functionality"""
    
    def execute(self):
        st.header("📊 Advanced Statistical Analysis")
        
        # Use cleaned data if available
        df = st.session_state.get('cleaned_df', self.df)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Descriptive Statistics", "Normality Tests", "Correlation Analysis", 
            "Hypothesis Testing", "Regression Analysis"
        ])
        
        with tab1:
            self._descriptive_statistics(df)
        
        with tab2:
            self._normality_analysis(df)
        
        with tab3:
            self._correlation_analysis(df)
        
        with tab4:
            self._hypothesis_testing(df)
        
        with tab5:
            self._regression_analysis(df)
        
        # PDF Report Generation
        st.subheader("📄 Generate SPSS-like Report")
        if st.button("Generate Comprehensive PDF Report"):
            self._generate_pdf_report(df)
    
    def _descriptive_statistics(self, df):
        """Display comprehensive descriptive statistics"""
        st.subheader("Descriptive Statistics")
        
        if self.numerical_cols:
            selected_cols = st.multiselect("Select numerical variables:", self.numerical_cols, default=self.numerical_cols[:3])
            
            if selected_cols:
                # Basic statistics
                desc_stats = df[selected_cols].describe().T
                desc_stats['variance'] = df[selected_cols].var()
                desc_stats['skewness'] = df[selected_cols].skew()
                desc_stats['kurtosis'] = df[selected_cols].kurtosis()
                desc_stats['missing'] = df[selected_cols].isnull().sum()
                
                st.write("**Basic Statistics:**")
                st.dataframe(desc_stats.round(4), use_container_width=True)
                
                # Distribution plots
                st.write("**Distribution Plots:**")
                for col in selected_cols:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram with KDE
                    sns.histplot(df[col].dropna(), kde=True, ax=ax1)
                    ax1.set_title(f'Distribution of {col}')
                    ax1.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
                    ax1.axvline(df[col].median(), color='green', linestyle='--', label='Median')
                    ax1.legend()
                    
                    # Box plot
                    sns.boxplot(y=df[col].dropna(), ax=ax2)
                    ax2.set_title(f'Box Plot of {col}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        if self.categorical_cols:
            st.subheader("Categorical Variable Analysis")
            cat_var = st.selectbox("Select categorical variable:", self.categorical_cols)
            
            if cat_var:
                freq_table = df[cat_var].value_counts()
                st.write("**Frequency Table:**")
                st.dataframe(freq_table, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                freq_table.plot(kind='bar', ax=ax)
                ax.set_title(f'Frequency Distribution of {cat_var}')
                ax.set_ylabel('Frequency')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    def _normality_analysis(self, df):
        """Perform normality tests"""
        st.subheader("Normality Tests")
        
        if not self.numerical_cols:
            st.info("No numerical variables for normality testing")
            return
        
        selected_var = st.selectbox("Select variable for normality testing:", self.numerical_cols)
        
        if selected_var:
            results = StatisticalTests.normality_tests(df, selected_var)
            
            st.write("**Normality Test Results:**")
            for test_name, test_results in results.items():
                st.write(f"**{test_name.title()}:**")
                col1, col2 = st.columns(2)
                col1.metric("Test Statistic", f"{test_results['statistic']:.4f}")
                col2.metric("P-value", f"{test_results['p_value']:.4f}")
                
                # Interpretation
                if test_results['p_value'] > 0.05:
                    st.success(f"✅ {test_name.title()}: Data appears normal (p > 0.05)")
                else:
                    st.error(f"❌ {test_name.title()}: Data does not appear normal (p ≤ 0.05)")
            
            # Q-Q plot
            st.write("**Q-Q Plot:**")
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(df[selected_var].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot for {selected_var}')
            st.pyplot(fig)
    
    def _correlation_analysis(self, df):
        """Perform correlation analysis"""
        st.subheader("Correlation Analysis")
        
        if len(self.numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for correlation analysis")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select first variable:", self.numerical_cols, key='corr_var1')
        with col2:
            var2 = st.selectbox("Select second variable:", self.numerical_cols, key='corr_var2')
        
        if var1 and var2:
            results = StatisticalTests.correlation_tests(df, var1, var2)
            
            st.write("**Correlation Results:**")
            for test_name, test_results in results.items():
                st.write(f"**{test_name.title()} Correlation:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Correlation", f"{test_results['correlation']:.4f}")
                col2.metric("P-value", f"{test_results['p_value']:.4f}")
                
                # Strength interpretation
                corr_strength = abs(test_results['correlation'])
                if corr_strength > 0.7:
                    strength = "Strong"
                elif corr_strength > 0.3:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                col3.metric("Strength", strength)
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=var1, y=var2, ax=ax)
            ax.set_title(f'Scatter Plot: {var1} vs {var2}')
            st.pyplot(fig)
    
    def _hypothesis_testing(self, df):
        """Perform hypothesis testing"""
        st.subheader("Hypothesis Testing")
        
        test_type = st.selectbox(
            "Select test type:",
            ["T-Test", "ANOVA", "Chi-Square Test"]
        )
        
        if test_type == "T-Test":
            self._t_test_interface(df)
        elif test_type == "ANOVA":
            self._anova_interface(df)
        elif test_type == "Chi-Square Test":
            self._chi_square_interface(df)
    
    def _t_test_interface(self, df):
        """T-test interface"""
        col1, col2 = st.columns(2)
        with col1:
            numerical_var = st.selectbox("Select numerical variable:", self.numerical_cols, key='ttest_num')
        with col2:
            categorical_var = st.selectbox("Select categorical variable (2 groups):", 
                                         [col for col in self.categorical_cols if df[col].nunique() == 2],
                                         key='ttest_cat')
        
        if numerical_var and categorical_var:
            results = StatisticalTests.t_test(df, numerical_var, categorical_var)
            
            if 'error' not in results:
                st.write("**T-Test Results:**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("T-Statistic", f"{results['t_statistic']:.4f}")
                col2.metric("P-value", f"{results['p_value']:.4f}")
                col3.metric("Group 1 Mean", f"{results['group1_mean']:.4f}")
                col4.metric("Group 2 Mean", f"{results['group2_mean']:.4f}")
                
                # Interpretation
                if results['p_value'] < 0.05:
                    st.success("✅ Significant difference between groups (p < 0.05)")
                else:
                    st.info("ℹ️ No significant difference between groups (p ≥ 0.05)")
    
    def _anova_interface(self, df):
        """ANOVA interface"""
        col1, col2 = st.columns(2)
        with col1:
            numerical_var = st.selectbox("Select numerical variable:", self.numerical_cols, key='anova_num')
        with col2:
            categorical_var = st.selectbox("Select categorical variable:", self.categorical_cols, key='anova_cat')
        
        if numerical_var and categorical_var:
            results = StatisticalTests.anova_test(df, numerical_var, categorical_var)
            
            st.write("**ANOVA Results:**")
            col1, col2 = st.columns(2)
            col1.metric("F-Statistic", f"{results['f_statistic']:.4f}")
            col2.metric("P-value", f"{results['p_value']:.4f}")
            
            st.write("**Group Means:**")
            st.dataframe(pd.DataFrame.from_dict(results['group_means'], orient='index', columns=['Mean']))
            
            if results['p_value'] < 0.05:
                st.success("✅ Significant differences between groups (p < 0.05)")
            else:
                st.info("ℹ️ No significant differences between groups (p ≥ 0.05)")
    
    def _chi_square_interface(self, df):
        """Chi-square test interface"""
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select first categorical variable:", self.categorical_cols, key='chi1')
        with col2:
            var2 = st.selectbox("Select second categorical variable:", self.categorical_cols, key='chi2')
        
        if var1 and var2:
            results = StatisticalTests.chi_square_test(df, var1, var2)
            
            st.write("**Chi-Square Test Results:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-Square", f"{results['chi2_statistic']:.4f}")
            col2.metric("P-value", f"{results['p_value']:.4f}")
            col3.metric("Degrees of Freedom", results['degrees_of_freedom'])
            
            st.write("**Contingency Table:**")
            st.dataframe(results['contingency_table'], use_container_width=True)
            
            if results['p_value'] < 0.05:
                st.success("✅ Significant association between variables (p < 0.05)")
            else:
                st.info("ℹ️ No significant association between variables (p ≥ 0.05)")
    
    def _regression_analysis(self, df):
        """Perform regression analysis"""
        st.subheader("Regression Analysis")
        
        regression_type = st.radio(
            "Select regression type:",
            ["Linear Regression", "Logistic Regression"],
            horizontal=True
        )
        
        if regression_type == "Linear Regression":
            self._linear_regression_interface(df)
        else:
            self._logistic_regression_interface(df)
    
    def _linear_regression_interface(self, df):
        """Linear regression interface"""
        st.write("**Linear Regression**")
        
        target = st.selectbox("Select target variable:", self.numerical_cols, key='lin_target')
        features = st.multiselect("Select predictor variables:", 
                                [col for col in self.numerical_cols if col != target],
                                key='lin_features')
        
        if target and features:
            try:
                # Prepare data
                X = df[features]
                y = df[target]
                
                # Handle missing values
                X = X.dropna()
                y = y.loc[X.index]
                
                # Add constant
                X = sm.add_constant(X)
                
                # Fit model
                model = sm.OLS(y, X).fit()
                
                # Display results
                st.write("**Regression Results:**")
                st.text(str(model.summary()))
                
                # Diagnostic plots
                st.write("**Diagnostic Plots:**")
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                # Residuals vs Fitted
                ax1.scatter(model.fittedvalues, model.resid)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_xlabel('Fitted values')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Residuals vs Fitted')
                
                # Q-Q plot
                stats.probplot(model.resid, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot')
                
                # Scale-Location
                standardized_resid = model.resid / np.sqrt(model.mse_resid)
                ax3.scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)))
                ax3.set_xlabel('Fitted values')
                ax3.set_ylabel('√|Standardized Residuals|')
                ax3.set_title('Scale-Location')
                
                # Residuals vs Leverage
                from statsmodels.stats.outliers_influence import OLSInfluence
                influence = OLSInfluence(model)
                ax4.scatter(influence.hat_matrix_diag, standardized_resid)
                ax4.set_xlabel('Leverage')
                ax4.set_ylabel('Standardized Residuals')
                ax4.set_title('Residuals vs Leverage')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Regression failed: {str(e)}")
    
    def _logistic_regression_interface(self, df):
        """Logistic regression interface"""
        st.write("**Logistic Regression**")
        
        # For logistic regression, target should be binary
        binary_targets = [col for col in self.numerical_cols if df[col].nunique() == 2]
        if not binary_targets:
            binary_targets = [col for col in self.categorical_cols if df[col].nunique() == 2]
        
        if not binary_targets:
            st.info("No suitable binary target variable found for logistic regression")
            return
        
        target = st.selectbox("Select binary target variable:", binary_targets, key='log_target')
        features = st.multiselect("Select predictor variables:", 
                                [col for col in self.numerical_cols if col != target],
                                key='log_features')
        
        if target and features:
            try:
                # Prepare data
                X = df[features]
                y = df[target]
                
                # Handle missing values and encode if categorical
                X = X.dropna()
                y = y.loc[X.index]
                
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                # Add constant
                X = sm.add_constant(X)
                
                # Fit model
                model = sm.Logit(y, X).fit(disp=False)
                
                # Display results
                st.write("**Logistic Regression Results:**")
                st.text(str(model.summary()))
                
                # Model performance
                y_pred = model.predict(X) > 0.5
                accuracy = accuracy_score(y, y_pred)
                st.metric("Model Accuracy", f"{accuracy:.3f}")
                
            except Exception as e:
                st.error(f"Logistic regression failed: {str(e)}")
    
    def _generate_pdf_report(self, df):
        """Generate comprehensive PDF report"""
        with st.spinner("Generating PDF report..."):
            try:
                # Collect all analyses
                analyses = {}
                
                # Descriptive statistics
                if self.numerical_cols:
                    desc_stats = df[self.numerical_cols].describe().T
                    analyses["Descriptive Statistics"] = {
                        'Mean': desc_stats['mean'].to_dict(),
                        'Std Dev': desc_stats['std'].to_dict(),
                        'Min': desc_stats['min'].to_dict(),
                        'Max': desc_stats['max'].to_dict()
                    }
                
                # Generate PDF
                pdf_buffer = PDFReportGenerator.create_spss_report(df, analyses)
                
                # Create download link
                b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="statistical_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ PDF report generated successfully!")
                
            except Exception as e:
                st.error(f"Failed to generate PDF report: {str(e)}")

class ForecastingPhase(AnalysisPhase):
    """Time series forecasting analysis"""
    
    def execute(self):
        st.header("📈 Time Series Forecasting")
        
        if not self.date_cols:
            st.info("No date columns found for forecasting analysis")
            return
        
        # Use cleaned data if available
        df = st.session_state.get('cleaned_df', self.df)
        
        st.subheader("Data Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox("Select date column:", self.date_cols)
        with col2:
            value_col = st.selectbox("Select value column:", self.numerical_cols)
        with col3:
            forecast_periods = st.slider("Forecast periods:", 1, Config.MAX_FORECAST_PERIODS, 30)
        
        if date_col and value_col:
            # Prepare time series data
            ts_data = df.set_index(date_col)[value_col].sort_index()
            
            st.subheader("Time Series Analysis")
            self._time_series_analysis(ts_data, value_col)
            
            st.subheader("Forecasting Models")
            self._forecasting_interface(ts_data, forecast_periods, value_col)
    
    def _time_series_analysis(self, ts_data, value_col):
        """Analyze time series properties"""
        # Stationarity test
        adf_result = adfuller(ts_data.dropna())
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
        col2.metric("P-value", f"{adf_result[1]:.4f}")
        col3.metric("Observations", len(ts_data))
        col4.metric("Stationary", "Yes" if adf_result[1] < 0.05 else "No")
        
        # Time series decomposition
        st.write("**Time Series Decomposition:**")
        try:
            decomposition = seasonal_decompose(ts_data.dropna(), period=min(30, len(ts_data)//4), extrapolate_trend='freq')
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
            
            decomposition.observed.plot(ax=ax1, title='Original')
            decomposition.trend.plot(ax=ax2, title='Trend')
            decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            decomposition.resid.plot(ax=ax4, title='Residual')
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Decomposition failed: {e}")
    
    def _forecasting_interface(self, ts_data, periods, value_col):
        """Forecasting model interface"""
        model_type = st.selectbox(
            "Select forecasting model:",
            ["ARIMA", "Exponential Smoothing", "Linear Regression"]
        )
        
        if st.button("Generate Forecast"):
            with st.spinner("Training forecasting model..."):
                if model_type == "ARIMA":
                    results = ForecastingEngine.arima_forecast(ts_data, periods)
                elif model_type == "Exponential Smoothing":
                    results = ForecastingEngine.exponential_smoothing_forecast(ts_data, periods)
                else:
                    # For linear regression, we need to convert back to DataFrame
                    temp_df = pd.DataFrame({
                        'date': ts_data.index,
                        'value': ts_data.values
                    })
                    results = ForecastingEngine.linear_regression_forecast(temp_df, 'date', 'value', periods)
                
                if 'error' in results:
                    st.error(f"Forecasting failed: {results['error']}")
                else:
                    self._display_forecast_results(ts_data, results, periods, value_col)
    
    def _display_forecast_results(self, historical_data, forecast_results, periods, value_col):
        """Display forecasting results"""
        st.subheader("Forecast Results")
        
        # Display model metrics
        if 'aic' in forecast_results:
            col1, col2 = st.columns(2)
            col1.metric("AIC", f"{forecast_results['aic']:.2f}")
            if 'bic' in forecast_results:
                col2.metric("BIC", f"{forecast_results['bic']:.2f}")
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, label='Historical', color='blue')
        
        # Generate future dates
        last_date = historical_data.index[-1]
        if isinstance(last_date, datetime):
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
        else:
            future_dates = range(len(historical_data), len(historical_data) + periods)
        
        # Plot forecast
        ax.plot(future_dates, forecast_results['forecast'], label='Forecast', color='red', linestyle='--')
        
        # Plot confidence intervals if available
        if 'confidence_intervals' in forecast_results:
            ci = forecast_results['confidence_intervals']
            ax.fill_between(future_dates, ci.iloc[:, 0], ci.iloc[:, 1], color='red', alpha=0.1, label='95% CI')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(f'{forecast_results["model"]} Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Forecast values table
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_results['forecast']
        })
        st.write("**Forecast Values:**")
        st.dataframe(forecast_df, use_container_width=True)

class DatasetCharacteristicsPhase(AnalysisPhase):
    """Handles dataset characteristics analysis"""
    
    def execute(self):
        st.header("🔍 Dataset Characteristics")
        
        analysis = DataAnalyzer.analyze_dataset(self.df)
        
        # Display dataset type
        st.info(f"**Dataset Type:** {analysis['type_analysis']['primary_type']}")
        
        self._display_basic_info(analysis['basic_info'])
        self._display_data_quality(analysis['data_quality'])
        self._display_type_analysis(analysis['type_analysis'])
        self._display_patterns(analysis['patterns'])
    
    def _display_basic_info(self, basic_info: dict):
        """Display basic dataset information"""
        st.subheader("📋 Basic Information")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{basic_info['rows']:,}")
        col2.metric("Total Columns", basic_info['columns'])
        col3.metric("Memory Usage", f"{basic_info['memory_usage_mb']:.2f} MB")
        col4.metric("Duplicate Rows", basic_info['duplicate_rows'])
    
    def _display_data_quality(self, data_quality: dict):
        """Display data quality metrics"""
        st.subheader("📈 Data Quality")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completeness", f"{data_quality['completeness_score']:.1f}%")
        col2.metric("Missing Values", f"{data_quality['missing_values_percentage']:.2f}%")
        col3.metric("Duplicate Rows", f"{data_quality['duplicate_percentage']:.2f}%")
        col4.metric("Columns with Missing", data_quality['columns_with_missing'])
    
    def _display_type_analysis(self, type_analysis: dict):
        """Display data type analysis"""
        st.subheader("🔧 Data Types")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Numerical Columns", type_analysis['numerical_count'])
        col2.metric("Categorical Columns", type_analysis['categorical_count'])
        col3.metric("Date Columns", type_analysis['date_count'])
        col4.metric("Primary Type", type_analysis['primary_type'])
    
    def _display_patterns(self, patterns: dict):
        """Display detected patterns"""
        st.subheader("🔄 Patterns")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Outlier Presence", patterns['outlier_pattern'])
        col2.metric("Correlation Strength", patterns['correlation_strength'])
        col3.metric("Temporal Patterns", patterns['temporal_patterns'])

class DataValidationPhase(AnalysisPhase):
    """Handles data validation and quality assessment"""
    
    def execute(self):
        st.header("📋 Data Validation")
        
        st.subheader("Data Profiling")
        self._display_data_profiling()
        
        st.subheader("Missing Data Analysis")
        self._analyze_missing_data()
        
        st.subheader("Outlier Detection")
        self._detect_outliers()
    
    def _display_data_profiling(self):
        """Display data profiling information"""
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", len(self.df))
        col1.metric("Total Features", len(self.df.columns))
        
        col2.metric("Numerical Features", len(self.numerical_cols))
        col2.metric("Categorical Features", len(self.categorical_cols))
        
        duplicate_count = self.df.duplicated().sum()
        col3.metric("Duplicate Records", duplicate_count)
        
        total_missing = self.df.isnull().sum().sum()
        col4.metric("Total Missing Values", total_missing)
        
        # Data preview
        with st.expander("Data Preview"):
            st.dataframe(self.df.head(10), use_container_width=True)
    
    def _analyze_missing_data(self):
        """Analyze missing data patterns"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Percentage', ascending=False)
        
        # Display missing data summary
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        if missing_data.sum() == 0:
            st.success("✅ No missing values found!")
    
    def _detect_outliers(self):
        """Detect and visualize outliers"""
        if not self.numerical_cols:
            st.info("No numerical columns for outlier detection")
            return
        
        selected_var = st.selectbox("Select variable for outlier analysis:", self.numerical_cols)
        
        if selected_var:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=self.df[selected_var], ax=ax)
                ax.set_title(f'Box Plot - {selected_var}')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(self.df[selected_var], kde=True, ax=ax)
                ax.set_title(f'Distribution - {selected_var}')
                st.pyplot(fig)

class ExploratoryAnalysisPhase(AnalysisPhase):
    """Handles exploratory data analysis"""
    
    def execute(self):
        st.header("🔍 Exploratory Analysis")
        
        if self.numerical_cols:
            st.subheader("Distribution Analysis")
            self._distribution_analysis()
            
            st.subheader("Correlation Analysis")
            self._correlation_analysis()
        
        if len(self.numerical_cols) >= 2:
            st.subheader("Multivariate Analysis")
            self._multivariate_analysis()
    
    def _distribution_analysis(self):
        """Distribution analysis"""
        selected_var = st.selectbox("Select variable for distribution analysis:", self.numerical_cols)
        
        if selected_var:
            data = self.df[selected_var].dropna()
            
            # Calculate statistics
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            sns.histplot(data, kde=True, ax=ax1)
            ax1.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
            ax1.set_title(f'Distribution of {selected_var}')
            ax1.legend()
            
            # Box plot
            sns.boxplot(y=data, ax=ax2)
            ax2.set_title(f'Box Plot of {selected_var}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{mean:.2f}")
            col2.metric("Standard Deviation", f"{std:.2f}")
            col3.metric("Median", f"{np.median(data):.2f}")
            col4.metric("Sample Size", len(data))
    
    def _correlation_analysis(self):
        """Correlation analysis"""
        if len(self.numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for correlation analysis")
            return
        
        corr_matrix = self.df[self.numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    
    def _multivariate_analysis(self):
        """Multivariate analysis using PCA"""
        X = self.df[self.numerical_cols].dropna()
        
        if len(X) < 2:
            st.info("Insufficient data for PCA")
            return
        
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA()
        principal_components = pca.fit_transform(X_scaled)
        
        exp_variance = pca.explained_variance_ratio_
        cum_variance = np.cumsum(exp_variance)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        components = range(1, len(exp_variance) + 1)
        ax.bar(components, exp_variance, alpha=0.6, color='skyblue', label='Individual')
        ax.step(components, cum_variance, where='mid', label='Cumulative', color='red')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Scree Plot')
        ax.legend()
        st.pyplot(fig)
        
        n_components_95 = np.argmax(cum_variance >= 0.95) + 1
        st.info(f"**Components needed for 95% variance: {n_components_95}**")

class ModelEvaluationPhase(AnalysisPhase):
    """Handles model evaluation and validation"""
    
    def execute(self):
        st.header("🤖 Model Evaluation")
        
        problem_type = st.radio(
            "Select Problem Type:", 
            ["Classification", "Regression"],
            horizontal=True
        )
        
        # Target variable selection
        if problem_type == "Classification":
            target_options = [col for col in self.df.columns 
                            if self.df[col].nunique() <= 10 or self.df[col].dtype == 'object']
        else:
            target_options = self.numerical_cols
        
        if not target_options:
            st.error("No suitable target variables found")
            return
            
        target_var = st.selectbox("Select Target Variable:", target_options)
        
        if st.button("Train and Evaluate Model"):
            self._train_and_evaluate(problem_type, target_var)
    
    def _train_and_evaluate(self, problem_type: str, target_var: str):
        """Train and evaluate model"""
        with st.spinner("Training model..."):
            try:
                # Preprocess data
                df_processed = DataPreprocessor.preprocess_data(self.df, target_var)
                X, y = DataPreprocessor.prepare_features(df_processed, target_var)
                
                if X.empty:
                    st.error("No valid features available")
                    return
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=Config.DEFAULT_TEST_SIZE, random_state=42,
                    stratify=y if problem_type == "Classification" else None
                )
                
                # Train model
                if problem_type == "Classification":
                    results = self._evaluate_classification_model(X_train, X_test, y_train, y_test)
                else:
                    results = self._evaluate_regression_model(X_train, X_test, y_train, y_test)
                
                self._display_model_results(results, problem_type)
                
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")
    
    def _evaluate_classification_model(self, X_train, X_test, y_train, y_test) -> dict:
        """Evaluate classification model"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        return {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'feature_names': X_train.columns.tolist()
        }
    
    def _evaluate_regression_model(self, X_train, X_test, y_train, y_test) -> dict:
        """Evaluate regression model"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        return {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'feature_names': X_train.columns.tolist()
        }
    
    def _display_model_results(self, results, problem_type):
        """Display model evaluation results"""
        st.subheader("📊 Model Performance")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if problem_type == "Classification":
            col1.metric("Accuracy", f"{results['accuracy']:.3f}")
            col2.metric("Precision", f"{results['precision']:.3f}")
            col3.metric("Recall", f"{results['recall']:.3f}")
            col4.metric("F1-Score", f"{results['f1']:.3f}")
            
            # Confusion Matrix
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
        else:
            col1.metric("R² Score", f"{results['r2']:.3f}")
            col2.metric("RMSE", f"{results['rmse']:.3f}")
            col3.metric("MAE", f"{results['mae']:.3f}")
            
            # Actual vs Predicted plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_test'], results['y_pred'], alpha=0.6)
            ax.plot([results['y_test'].min(), results['y_test'].max()], 
                   [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted')
            st.pyplot(fig)
        
        # Feature importance
        self._plot_feature_importance(results)
    
    def _plot_feature_importance(self, results):
        """Plot feature importance"""
        st.subheader("🔍 Feature Importance")
        
        try:
            if hasattr(results['model'], 'feature_importances_'):
                importances = results['model'].feature_importances_
                feature_names = results['feature_names']
                
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(fi_df['feature'], fi_df['importance'])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.info("Feature importance not available for this model.")
                
        except Exception as e:
            st.warning(f"Could not compute feature importance: {str(e)}")

class StatisticalAnalysisApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Advanced Statistical Analysis Suite",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main application runner"""
        st.title("📊 Advanced Statistical Analysis Suite")
        st.markdown("""
        Comprehensive statistical analysis platform with SPSS-like functionality, 
        forecasting capabilities, and automated reporting for undergraduate research.
        """)
        
        # Data loading
        df = self.load_data()
        
        if df is not None:
            self.run_analysis(df)
        else:
            self.show_instructions()
    
    def load_data(self):
        """Handle data loading"""
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "URL", "Sample Data"],
            horizontal=True
        )
        
        try:
            if data_source == "Upload File":
                return self._load_from_upload()
            elif data_source == "URL":
                return self._load_from_url()
            elif data_source == "Sample Data":
                return self._load_sample_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def _load_from_upload(self):
        """Load data from file upload"""
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            df = DataLoader.load_from_uploaded_file(uploaded_file)
            st.success(f"✅ Loaded {uploaded_file.name}")
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        return None
    
    def _load_from_url(self):
        """Load data from URL"""
        url = st.text_input(
            "Enter dataset URL:",
            placeholder="https://example.com/data.csv"
        )
        
        if url:
            with st.spinner("Downloading data..."):
                df = DataLoader.load_from_url(url)
            st.success("✅ Data loaded from URL")
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        return None
    
    def _load_sample_data(self):
        """Load sample data"""
        dataset_type = st.selectbox(
            "Select sample dataset type:",
            ["Mixed", "Time Series", "Categorical", "Numerical", "Forecasting"]
        )
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df = DataLoader.generate_sample_data(dataset_type.lower().replace(" ", "_"))
            st.success("✅ Sample data generated")
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        return None
    
    def run_analysis(self, df):
        """Run selected analysis phases"""
        # Sample large datasets for performance
        if len(df) > Config.MAX_ROWS_FOR_ANALYSIS:
            st.warning(f"Large dataset detected. Sampling to {Config.MAX_ROWS_FOR_ANALYSIS} rows.")
            df = df.sample(n=Config.MAX_ROWS_FOR_ANALYSIS, random_state=42)
        
        # Phase selection
        st.sidebar.header("Analysis Configuration")
        phases = st.sidebar.multiselect(
            "Select analysis phases:",
            [
                "Data Cleaning", "Dataset Characteristics", "Data Validation", 
                "Statistical Analysis", "Forecasting", "Model Evaluation"
            ],
            default=["Data Cleaning", "Dataset Characteristics", "Statistical Analysis"]
        )
        
        # Initialize and run phases
        analysis_phases = {
            "Data Cleaning": DataCleaningPhase(df),
            "Dataset Characteristics": DatasetCharacteristicsPhase(df),
            "Data Validation": DataValidationPhase(df),
            "Statistical Analysis": StatisticalAnalysisPhase(df),
            "Forecasting": ForecastingPhase(df),
            "Model Evaluation": ModelEvaluationPhase(df)
        }
        
        for phase_name in phases:
            phase = analysis_phases.get(phase_name)
            if phase:
                phase.execute()
                st.markdown("---")
    
    def show_instructions(self):
        """Show application instructions"""
        st.info("👆 Please load your data to begin analysis")
        
        with st.expander("📋 Comprehensive Guide to Statistical Analysis"):
            st.markdown("""
            ## 📊 Key Statistical Techniques for Undergraduate Research

            ### 1. Exploratory Data Analysis (EDA)
            EDA uses visualization and summary statistics to understand data patterns and relationships.

            * **Descriptive Statistics:** Calculate measures like **mean, median, mode** (central tendency), **standard deviation, variance, range** (variability), and **skewness/kurtosis** (distribution shape)
            * **Data Visualization:** 
                - **Histograms/Box Plots:** Show distribution and identify outliers
                - **Scatter Plots:** Visualize relationships between continuous variables
                - **Bar Charts/Cross-Tabulation:** Explore categorical variable relationships

            ### 2. Measuring Relationships (Association)
            Quantify connections between variables:

            * **Correlation Analysis:** Measures strength and direction of linear relationships
                - **Pearson's Correlation ($r$)**: For linear relationships (-1 to +1)
                - **Spearman's Rank Correlation**: For non-linear or ordinal data
            * **Chi-Square Test ($χ²$)**: Tests association between categorical variables

            ### 3. Predictive and Explanatory Modeling
            Model and predict relationships:

            * **Regression Analysis:** 
                - **Linear Regression:** Models relationships using straight lines
                - **Logistic Regression:** For categorical outcomes (yes/no predictions)

            ### 4. Advanced Pattern Recognition
            Identify complex structures:

            * **Cluster Analysis:** Groups similar data points into clusters
            * **Time Series Analysis:** Identifies trends, seasonality, and temporal patterns
            * **Principal Component Analysis (PCA):** Reduces dimensionality and uncovers hidden patterns

            ### Workflow:
            1. **Visualize and Summarize** (EDA)
            2. **Quantify Associations** (Correlation/Chi-Square)  
            3. **Model and Predict** (Regression/Clustering)
            """)

def main():
    """Main entry point"""
    try:
        app = StatisticalAnalysisApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the console for detailed error messages.")

if __name__ == "__main__":
    main()
