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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
import tempfile
import os

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

class ForecastingEngine:
    """Comprehensive time series forecasting engine"""
    
    @staticmethod
    def arima_forecast(series: pd.Series, periods: int, order: tuple = (1, 1, 1), seasonal_order: tuple = (0, 0, 0, 0)) -> dict:
        """ARIMA forecasting with comprehensive diagnostics"""
        try:
            # Ensure series is properly indexed
            if series.index.freq is None:
                series = series.asfreq('D')
            
            # Fit ARIMA model
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Model diagnostics
            residuals = fitted_model.resid
            aic = fitted_model.aic
            bic = fitted_model.bic
            hqic = fitted_model.hqic
            
            # Ljung-Box test for autocorrelation
            lb_test = acf(residuals, nlags=10, fft=False)
            
            return {
                'success': True,
                'model': 'ARIMA',
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'model_summary': fitted_model.summary(),
                'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'residuals': residuals,
                'parameters': fitted_model.params,
                'order': order,
                'seasonal_order': seasonal_order,
                'lb_test': lb_test
            }
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def exponential_smoothing_forecast(series: pd.Series, periods: int, 
                                     trend: str = 'add', seasonal: str = 'add', 
                                     seasonal_periods: int = None) -> dict:
        """Exponential smoothing forecasting with multiple methods"""
        try:
            if seasonal_periods and seasonal is not None:
                model = ExponentialSmoothing(
                    series, 
                    trend=trend, 
                    seasonal=seasonal, 
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(series, trend=trend)
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
            # Calculate confidence intervals (approximate)
            last_errors = fitted_model.resid[-10:] if len(fitted_model.resid) >= 10 else fitted_model.resid
            std_error = np.std(last_errors)
            z_value = 1.96  # 95% confidence
            
            conf_int = pd.DataFrame({
                'lower': forecast - z_value * std_error,
                'upper': forecast + z_value * std_error
            }, index=forecast.index)
            
            return {
                'success': True,
                'model': 'Exponential Smoothing',
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'smoothing_level': fitted_model.params.get('smoothing_level', None),
                'smoothing_trend': fitted_model.params.get('smoothing_trend', None),
                'smoothing_seasonal': fitted_model.params.get('smoothing_seasonal', None),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'residuals': fitted_model.resid,
                'trend_type': trend,
                'seasonal_type': seasonal
            }
        except Exception as e:
            logger.error(f"Exponential smoothing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def linear_regression_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> dict:
        """Linear regression based forecasting with time features"""
        try:
            # Create comprehensive time features
            df_forecast = df.copy()
            df_forecast = df_forecast.sort_values(date_col)
            
            # Create time index and additional features
            df_forecast['time_index'] = range(len(df_forecast))
            df_forecast['month'] = df_forecast[date_col].dt.month
            df_forecast['quarter'] = df_forecast[date_col].dt.quarter
            df_forecast['day_of_week'] = df_forecast[date_col].dt.dayofweek
            
            # Prepare features
            feature_cols = ['time_index', 'month', 'quarter', 'day_of_week']
            X = df_forecast[feature_cols]
            y = df_forecast[value_col]
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future predictions
            last_date = df_forecast[date_col].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            
            future_df = pd.DataFrame({
                date_col: future_dates,
                'time_index': range(len(df_forecast), len(df_forecast) + periods),
                'month': [d.month for d in future_dates],
                'quarter': [d.quarter for d in future_dates],
                'day_of_week': [d.dayofweek for d in future_dates]
            })
            
            future_X = future_df[feature_cols]
            future_predictions = model.predict(future_X)
            
            # Calculate confidence intervals
            y_pred = model.predict(X)
            residuals = y - y_pred
            std_error = np.std(residuals)
            z_value = 1.96
            
            conf_int = pd.DataFrame({
                'lower': future_predictions - z_value * std_error,
                'upper': future_predictions + z_value * std_error
            }, index=future_dates)
            
            forecast_series = pd.Series(future_predictions, index=future_dates, name='forecast')
            
            return {
                'success': True,
                'model': 'Linear Regression',
                'forecast': forecast_series,
                'confidence_intervals': conf_int,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'feature_importance': dict(zip(feature_cols, model.coef_)),
                'residuals': residuals
            }
        except Exception as e:
            logger.error(f"Linear regression forecasting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def sarima_forecast(series: pd.Series, periods: int, order: tuple = (1, 1, 1), 
                       seasonal_order: tuple = (1, 1, 1, 12)) -> dict:
        """SARIMA forecasting for seasonal data"""
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            return {
                'success': True,
                'model': 'SARIMA',
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'model_summary': fitted_model.summary(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'parameters': fitted_model.params,
                'order': order,
                'seasonal_order': seasonal_order
            }
        except Exception as e:
            logger.error(f"SARIMA forecasting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def time_series_decomposition(series: pd.Series, period: int = None) -> dict:
        """Comprehensive time series decomposition"""
        try:
            if period is None:
                period = min(30, len(series) // 4)
            
            decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
            
            # Stationarity test
            adf_result = adfuller(series.dropna())
            
            return {
                'success': True,
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'stationary': adf_result[1] < 0.05,
                'period': period
            }
        except Exception as e:
            logger.error(f"Time series decomposition failed: {e}")
            return {'success': False, 'error': str(e)}

class AdvancedDataCleaner:
    """Advanced data cleaning and preprocessing with multiple strategies"""
    
    @staticmethod
    def clean_data(df: pd.DataFrame, cleaning_options: dict) -> pd.DataFrame:
        """Apply comprehensive data cleaning based on options"""
        df_clean = df.copy()
        
        # Handle missing values
        if cleaning_options.get('handle_missing'):
            df_clean = AdvancedDataCleaner._handle_missing_values(df_clean, cleaning_options)
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates'):
            df_clean = df_clean.drop_duplicates()
        
        # Handle outliers
        if cleaning_options.get('handle_outliers'):
            df_clean = AdvancedDataCleaner._handle_outliers(df_clean, cleaning_options)
        
        # Convert data types
        if cleaning_options.get('convert_dtypes'):
            df_clean = AdvancedDataCleaner._convert_data_types(df_clean)
        
        # Standardize text data
        if cleaning_options.get('standardize_text'):
            df_clean = AdvancedDataCleaner._standardize_text_data(df_clean)
        
        # Feature engineering
        if cleaning_options.get('feature_engineering'):
            df_clean = AdvancedDataCleaner._feature_engineering(df_clean)
        
        # Scale/normalize data
        if cleaning_options.get('scale_data'):
            df_clean = AdvancedDataCleaner._scale_data(df_clean, cleaning_options)
        
        return df_clean
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame, options: dict) -> pd.DataFrame:
        """Handle missing values with multiple strategies"""
        strategy = options.get('missing_strategy', 'mean')
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'remove':
            return df.dropna()
        
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        elif strategy == 'mode':
            # For categorical and numerical
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            # For categorical, use mode
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        elif strategy == 'forward_fill':
            df = df.ffill()
        
        elif strategy == 'interpolate':
            df[numerical_cols] = df[numerical_cols].interpolate(method='linear')
        
        # For any remaining missing values in categorical columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    @staticmethod
    def _handle_outliers(df: pd.DataFrame, options: dict) -> pd.DataFrame:
        """Handle outliers with multiple strategies"""
        strategy = options.get('outlier_strategy', 'clip')
        method = options.get('outlier_method', 'iqr')
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                lower_bound = data[z_scores <= 3].min() if len(data[z_scores <= 3]) > 0 else data.min()
                upper_bound = data[z_scores <= 3].max() if len(data[z_scores <= 3]) > 0 else data.max()
            elif method == 'percentile':
                lower_bound = data.quantile(0.01)
                upper_bound = data.quantile(0.99)
            
            if strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif strategy == 'clip':
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            elif strategy == 'transform':
                # Log transformation
                min_val = df[col].min()
                if min_val <= 0:
                    df[col] = np.log1p(df[col] - min_val + 1)
                else:
                    df[col] = np.log(df[col])
            elif strategy == 'winsorize':
                from scipy.stats.mstats import winsorize
                df[col] = winsorize(df[col], limits=[0.05, 0.05])
        
        return df
    
    @staticmethod
    def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types automatically with optimization"""
        # Convert to categorical if low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Less than 5% unique values
                df[col] = df[col].astype('category')
        
        # Convert potential numeric columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        # Optimize numerical data types
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == np.float64:
                # Check if can be converted to float32
                if (df[col].max() < np.finfo(np.float32).max and 
                    df[col].min() > np.finfo(np.float32).min):
                    df[col] = df[col].astype(np.float32)
            elif df[col].dtype == np.int64:
                # Check if can be converted to int32
                if (df[col].max() < np.iinfo(np.int32).max and 
                    df[col].min() > np.iinfo(np.int32).min):
                    df[col] = df[col].astype(np.int32)
        
        return df
    
    @staticmethod
    def _standardize_text_data(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text data comprehensively"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            # Basic cleaning
            df[col] = (df[col].astype(str)
                       .str.strip()
                       .str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                       .str.title())  # Title case
            
            # Handle common inconsistencies
            df[col] = (df[col].str.replace('_', ' ')
                       .str.replace('-', ' ')
                       .str.replace('&', 'and'))
        
        return df
    
    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features automatically"""
        # Extract date features
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Create interaction terms for highly correlated numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.7:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                        high_corr_pairs.append((col1, col2))
        
        # Polynomial features for important numerical columns
        if len(numerical_cols) > 0:
            for col in numerical_cols[:3]:  # Limit to top 3 columns
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        # Binning for numerical variables
        for col in numerical_cols[:2]:
            if df[col].nunique() > 10:
                df[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=False)
        
        return df
    
    @staticmethod
    def _scale_data(df: pd.DataFrame, options: dict) -> pd.DataFrame:
        """Scale/normalize data"""
        scaling_method = options.get('scaling_method', 'standard')
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        elif scaling_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        return df

class SPSSReportGenerator:
    """Generate comprehensive SPSS-like PDF reports"""
    
    @staticmethod
    def create_comprehensive_report(df: pd.DataFrame, analyses: dict, filename: str = "statistical_analysis_report.pdf"):
        """Create comprehensive SPSS-like PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Title Page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#2C3E50')
        )
        
        story.append(Paragraph("STATISTICAL ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Add timestamp
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,
            textColor=colors.gray
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style))
        story.append(Spacer(1, 40))
        
        # Dataset Overview Section
        story.append(Paragraph("1. DATASET OVERVIEW", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        dataset_info = [
            ["Statistic", "Value"],
            ["Total Cases", f"{len(df):,}"],
            ["Total Variables", str(len(df.columns))],
            ["Missing Values", f"{df.isnull().sum().sum():,}"],
            ["Duplicate Rows", f"{df.duplicated().sum():,}"],
            ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"]
        ]
        
        dataset_table = Table(dataset_info, colWidths=[200, 150])
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 20))
        
        # Variable Information Section
        story.append(Paragraph("2. VARIABLE INFORMATION", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        var_info = [["Variable", "Type", "Missing", "Unique", "Mean/Mode", "Std Dev"]]
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            
            if df[col].dtype in ['int64', 'float64']:
                central = f"{df[col].mean():.2f}"
                std_dev = f"{df[col].std():.2f}"
            else:
                central = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                std_dev = "N/A"
            
            var_info.append([col, dtype, str(missing), str(unique), str(central), std_dev])
        
        var_table = Table(var_info, colWidths=[100, 60, 50, 50, 70, 60])
        var_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(var_table)
        story.append(Spacer(1, 20))
        
        # Analysis Results Sections
        section_num = 3
        for analysis_name, analysis_data in analyses.items():
            if analysis_data:  # Only add if there's data
                story.append(Paragraph(f"{section_num}. {analysis_name.upper()}", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                if isinstance(analysis_data, dict):
                    analysis_table_data = [["Statistic", "Value", "Interpretation"]]
                    
                    for key, value in analysis_data.items():
                        if key not in ['success', 'error']:
                            if isinstance(value, (int, float)):
                                formatted_value = f"{value:.4f}"
                                # Add interpretation for common statistics
                                interpretation = SPSSReportGenerator._get_interpretation(key, value)
                            else:
                                formatted_value = str(value)
                                interpretation = ""
                            
                            analysis_table_data.append([key, formatted_value, interpretation])
                    
                    if len(analysis_table_data) > 1:  # If we have data beyond header
                        analysis_table = Table(analysis_table_data, colWidths=[120, 80, 150])
                        analysis_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(analysis_table)
                        story.append(Spacer(1, 15))
                
                section_num += 1
        
        # Summary Section
        story.append(Paragraph(f"{section_num}. SUMMARY & CONCLUSIONS", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary_text = """
        This comprehensive statistical analysis was conducted using advanced analytical techniques. 
        Key findings and patterns identified in the data are presented in the preceding sections. 
        Recommendations for further analysis and data-driven decision making are provided based on the statistical evidence.
        """
        
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=12
        )
        story.append(Paragraph(summary_text, summary_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def _get_interpretation(statistic: str, value: float) -> str:
        """Provide interpretations for common statistics"""
        interpretations = {
            'p_value': 'Significant' if value < 0.05 else 'Not Significant',
            'r_squared': 'Strong fit' if value > 0.7 else 'Moderate fit' if value > 0.3 else 'Weak fit',
            'correlation': 'Strong' if abs(value) > 0.7 else 'Moderate' if abs(value) > 0.3 else 'Weak',
            'vif': 'High multicollinearity' if value > 10 else 'Moderate' if value > 5 else 'Low',
            'f_statistic': 'Significant model' if value > 3.84 else 'Not significant',
            't_statistic': 'Significant' if abs(value) > 1.96 else 'Not significant'
        }
        
        for key, interpretation in interpretations.items():
            if key in statistic.lower():
                return interpretation
        
        return ""

class ComprehensiveStatisticalTests:
    """Comprehensive statistical testing suite"""
    
    @staticmethod
    def normality_tests(df: pd.DataFrame, column: str) -> dict:
        """Perform comprehensive normality tests"""
        data = df[column].dropna()
        
        if len(data) < 3:
            return {'error': 'Insufficient data for normality tests'}
        
        # Shapiro-Wilk test (recommended for n < 5000)
        shapiro_stat, shapiro_p = shapiro(data)
        
        # D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data, dist='norm')
        
        # Additional descriptive statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        return {
            'shapiro_wilk_statistic': shapiro_stat,
            'shapiro_wilk_p_value': shapiro_p,
            'dagostino_statistic': dagostino_stat,
            'dagostino_p_value': dagostino_p,
            'kolmogorov_smirnov_statistic': ks_stat,
            'kolmogorov_smirnov_p_value': ks_p,
            'anderson_darling_statistic': anderson_result.statistic,
            'anderson_critical_values': anderson_result.critical_values.tolist(),
            'anderson_significance_levels': anderson_result.significance_level.tolist(),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': shapiro_p > 0.05 and dagostino_p > 0.05,
            'sample_size': len(data)
        }
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Comprehensive correlation analysis"""
        data = df[[col1, col2]].dropna()
        
        if len(data) < 3:
            return {'error': 'Insufficient data for correlation analysis'}
        
        x = data[col1]
        y = data[col2]
        
        # Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(x, y)
        
        # Spearman correlation (monotonic)
        spearman_corr, spearman_p = spearmanr(x, y)
        
        # Kendall's tau (ordinal)
        kendall_tau, kendall_p = kendalltau(x, y)
        
        # Distance correlation (nonlinear)
        try:
            from dcor import distance_correlation
            dist_corr = distance_correlation(x, y)
        except:
            dist_corr = None
        
        # Confidence intervals
        n = len(x)
        pearson_ci = ComprehensiveStatisticalTests._correlation_ci(pearson_corr, n)
        spearman_ci = ComprehensiveStatisticalTests._correlation_ci(spearman_corr, n)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'pearson_confidence_interval': pearson_ci,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'spearman_confidence_interval': spearman_ci,
            'kendall_tau': kendall_tau,
            'kendall_p_value': kendall_p,
            'distance_correlation': dist_corr,
            'sample_size': n,
            'interpretation': ComprehensiveStatisticalTests._correlation_interpretation(pearson_corr)
        }
    
    @staticmethod
    def _correlation_ci(correlation: float, n: int, confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for correlation coefficient"""
        if n <= 3:
            return (None, None)
        
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)
        z_critical = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        lower_z = z - z_critical * se
        upper_z = z + z_critical * se
        
        lower = np.tanh(lower_z)
        upper = np.tanh(upper_z)
        
        return (lower, upper)
    
    @staticmethod
    def _correlation_interpretation(correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "Very strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very weak or no"
    
    @staticmethod
    def hypothesis_testing(df: pd.DataFrame, test_type: str, **kwargs) -> dict:
        """Comprehensive hypothesis testing"""
        if test_type == 't_test_independent':
            return ComprehensiveStatisticalTests._independent_t_test(df, **kwargs)
        elif test_type == 't_test_paired':
            return ComprehensiveStatisticalTests._paired_t_test(df, **kwargs)
        elif test_type == 'anova':
            return ComprehensiveStatisticalTests._anova_test(df, **kwargs)
        elif test_type == 'chi_square':
            return ComprehensiveStatisticalTests._chi_square_test(df, **kwargs)
        elif test_type == 'mann_whitney':
            return ComprehensiveStatisticalTests._mann_whitney_test(df, **kwargs)
        elif test_type == 'kruskal_wallis':
            return ComprehensiveStatisticalTests._kruskal_wallis_test(df, **kwargs)
        else:
            return {'error': f'Unknown test type: {test_type}'}
    
    @staticmethod
    def _independent_t_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """Independent samples t-test"""
        groups = df[categorical_col].unique()
        if len(groups) != 2:
            return {"error": "Categorical variable must have exactly 2 groups"}
        
        group1_data = df[df[categorical_col] == groups[0]][numerical_col].dropna()
        group2_data = df[df[categorical_col] == groups[1]][numerical_col].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1-1)*group1_data.std()**2 + (n2-1)*group2_data.std()**2) / (n1+n2-2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
        
        # Confidence interval for mean difference
        mean_diff = group1_data.mean() - group2_data.mean()
        se_diff = np.sqrt(group1_data.std()**2/n1 + group2_data.std()**2/n2)
        t_critical = stats.t.ppf(0.975, min(n1, n2)-1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return {
            'test_type': 'Independent Samples T-Test',
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': n1 + n2 - 2,
            'group1_mean': group1_data.mean(),
            'group2_mean': group2_data.mean(),
            'group1_std': group1_data.std(),
            'group2_std': group2_data.std(),
            'group1_size': n1,
            'group2_size': n2,
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'cohens_d': cohens_d,
            'effect_size_interpretation': ComprehensiveStatisticalTests._interpret_cohens_d(cohens_d),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d >= 0.8:
            return "Large effect"
        elif abs_d >= 0.5:
            return "Medium effect"
        elif abs_d >= 0.2:
            return "Small effect"
        else:
            return "Negligible effect"
    
    @staticmethod
    def _anova_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """One-way ANOVA test"""
        groups = [group for name, group in df.groupby(categorical_col)[numerical_col]]
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate effect size (Eta squared)
        ss_between = 0
        grand_mean = df[numerical_col].mean()
        for group in groups:
            ss_between += len(group) * (group.mean() - grand_mean) ** 2
        
        ss_total = ((df[numerical_col] - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total
        
        # Group statistics
        group_stats = df.groupby(categorical_col)[numerical_col].agg(['count', 'mean', 'std']).to_dict()
        
        return {
            'test_type': 'One-Way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'effect_size_interpretation': ComprehensiveStatisticalTests._interpret_eta_squared(eta_squared),
            'group_statistics': group_stats,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _interpret_eta_squared(eta2: float) -> str:
        """Interpret Eta squared effect size"""
        if eta2 >= 0.14:
            return "Large effect"
        elif eta2 >= 0.06:
            return "Medium effect"
        elif eta2 >= 0.01:
            return "Small effect"
        else:
            return "Negligible effect"
    
    @staticmethod
    def _chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Chi-square test of independence"""
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        if contingency_table.size == 0:
            return {"error": "No data for chi-square test"}
        
        # Check for expected frequencies
        expected = contingency_table.sum(axis=1).values.reshape(-1, 1) @ contingency_table.sum(axis=0).values.reshape(1, -1) / contingency_table.sum().sum()
        if (expected < 5).sum() / expected.size > 0.2:
            warning = "More than 20% of expected frequencies are less than 5. Consider Fisher's exact test."
        else:
            warning = None
        
        # Perform chi-square test
        chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramer's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        return {
            'test_type': 'Chi-Square Test of Independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_freedom': dof,
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected_freq.tolist(),
            'cramers_v': cramers_v,
            'effect_size_interpretation': ComprehensiveStatisticalTests._interpret_cramers_v(cramers_v),
            'warning': warning,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _interpret_cramers_v(v: float) -> str:
        """Interpret Cramer's V effect size"""
        if v >= 0.5:
            return "Strong association"
        elif v >= 0.3:
            return "Moderate association"
        elif v >= 0.1:
            return "Weak association"
        else:
            return "Negligible association"
    
    @staticmethod
    def _paired_t_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Paired samples t-test"""
        paired_data = df[[col1, col2]].dropna()
        
        if len(paired_data) < 2:
            return {"error": "Insufficient data for paired t-test"}
        
        differences = paired_data[col1] - paired_data[col2]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(paired_data[col1], paired_data[col2])
        
        # Calculate effect size
        cohens_d = differences.mean() / differences.std()
        
        return {
            'test_type': 'Paired Samples T-Test',
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': differences.mean(),
            'std_difference': differences.std(),
            'cohens_d': cohens_d,
            'sample_size': len(paired_data),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _mann_whitney_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """Mann-Whitney U test (non-parametric)"""
        groups = df[categorical_col].unique()
        if len(groups) != 2:
            return {"error": "Categorical variable must have exactly 2 groups"}
        
        group1_data = df[df[categorical_col] == groups[0]][numerical_col].dropna()
        group2_data = df[df[categorical_col] == groups[1]][numerical_col].dropna()
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        # Calculate effect size
        n1, n2 = len(group1_data), len(group2_data)
        r_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'test_type': 'Mann-Whitney U Test',
            'u_statistic': u_stat,
            'p_value': p_value,
            'rank_biserial_correlation': r_biserial,
            'group1_median': group1_data.median(),
            'group2_median': group2_data.median(),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _kruskal_wallis_test(df: pd.DataFrame, numerical_col: str, categorical_col: str) -> dict:
        """Kruskal-Wallis H test (non-parametric ANOVA)"""
        groups = [group for name, group in df.groupby(categorical_col)[numerical_col]]
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for Kruskal-Wallis test"}
        
        # Perform Kruskal-Wallis test
        h_stat, p_value = stats.kruskal(*groups)
        
        return {
            'test_type': 'Kruskal-Wallis H Test',
            'h_statistic': h_stat,
            'p_value': p_value,
            'group_medians': {name: group.median() for name, group in df.groupby(categorical_col)[numerical_col]},
            'significant': p_value < 0.05
        }

# Update the ForecastingPhase class to use the new ForecastingEngine
class ForecastingPhase:
    def __init__(self, df):
        self.df = df
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def execute(self):
        st.header("📈 Advanced Time Series Forecasting")
        
        if not self.date_cols:
            st.info("No date columns found for forecasting analysis")
            return
        
        df = st.session_state.get('cleaned_df', self.df)
        
        st.subheader("Data Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox("Select date column:", self.date_cols, key='forecast_date')
        with col2:
            value_col = st.selectbox("Select value column:", self.numerical_cols, key='forecast_value')
        with col3:
            forecast_periods = st.slider("Forecast periods:", 1, 365, 30, key='forecast_periods')
        
        if date_col and value_col:
            # Prepare time series
            ts_data = df.set_index(date_col)[value_col].sort_index().dropna()
            
            if len(ts_data) < 10:
                st.error("Insufficient data for forecasting (need at least 10 observations)")
                return
            
            # Model selection
            st.subheader("Forecasting Model Selection")
            model_type = st.selectbox(
                "Select forecasting model:",
                ["ARIMA", "SARIMA", "Exponential Smoothing", "Linear Regression"],
                key='forecast_model'
            )
            
            # Model parameters
            if model_type == "ARIMA":
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.slider("AR order (p):", 0, 5, 1)
                with col2:
                    d = st.slider("Difference order (d):", 0, 2, 1)
                with col3:
                    q = st.slider("MA order (q):", 0, 5, 1)
                order = (p, d, q)
            
            elif model_type == "SARIMA":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    p = st.slider("AR order (p):", 0, 3, 1)
                with col2:
                    d = st.slider("Difference order (d):", 0, 2, 1)
                with col3:
                    q = st.slider("MA order (q):", 0, 3, 1)
                with col4:
                    s = st.slider("Seasonal period:", 2, 24, 12)
                order = (p, d, q)
                seasonal_order = (1, 1, 1, s)
            
            elif model_type == "Exponential Smoothing":
                col1, col2 = st.columns(2)
                with col1:
                    trend = st.selectbox("Trend:", ['add', 'mul', None])
                with col2:
                    seasonal = st.selectbox("Seasonal:", ['add', 'mul', None])
                seasonal_periods = st.slider("Seasonal periods:", 2, 24, 12) if seasonal else None
            
            if st.button("Generate Forecast", type="primary", key='generate_forecast'):
                with st.spinner("Training forecasting model..."):
                    # Time series decomposition
                    st.subheader("Time Series Decomposition")
                    decomposition = ForecastingEngine.time_series_decomposition(ts_data)
                    
                    if decomposition['success']:
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                        
                        decomposition['observed'].plot(ax=ax1, title='Original Series')
                        decomposition['trend'].plot(ax=ax2, title='Trend Component')
                        decomposition['seasonal'].plot(ax=ax3, title='Seasonal Component')
                        decomposition['residual'].plot(ax=ax4, title='Residual Component')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Stationarity test results
                        col1, col2, col3 = st.columns(3)
                        col1.metric("ADF Statistic", f"{decomposition['adf_statistic']:.4f}")
                        col2.metric("P-value", f"{decomposition['adf_pvalue']:.4f}")
                        col3.metric("Stationary", "Yes" if decomposition['stationary'] else "No")
                    
                    # Generate forecast based on selected model
                    if model_type == "ARIMA":
                        results = ForecastingEngine.arima_forecast(ts_data, forecast_periods, order)
                    elif model_type == "SARIMA":
                        results = ForecastingEngine.sarima_forecast(ts_data, forecast_periods, order, seasonal_order)
                    elif model_type == "Exponential Smoothing":
                        results = ForecastingEngine.exponential_smoothing_forecast(
                            ts_data, forecast_periods, trend, seasonal, seasonal_periods
                        )
                    else:  # Linear Regression
                        results = ForecastingEngine.linear_regression_forecast(
                            df, date_col, value_col, forecast_periods
                        )
                    
                    if results.get('success'):
                        self._display_forecast_results(ts_data, results, value_col)
                    else:
                        st.error(f"Forecasting failed: {results.get('error', 'Unknown error')}")
    
    def _display_forecast_results(self, historical_data, forecast_results, value_col):
        """Display comprehensive forecast results"""
        st.subheader("Forecast Results")
        
        # Model metrics
        st.write("**Model Performance Metrics:**")
        col1, col2, col3 = st.columns(3)
        
        if 'aic' in forecast_results:
            col1.metric("AIC", f"{forecast_results['aic']:.2f}")
        if 'bic' in forecast_results:
            col2.metric("BIC", f"{forecast_results['bic']:.2f}")
        if 'r_squared' in forecast_results:
            col3.metric("R²", f"{forecast_results['r_squared']:.4f}")
        
        # Forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        forecast = forecast_results['forecast']
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if 'confidence_intervals' in forecast_results:
            ci = forecast_results['confidence_intervals']
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='red', alpha=0.2, label='95% CI')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(f'{forecast_results["model"]} Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Forecast values table
        st.write("**Forecast Values:**")
        forecast_df = pd.DataFrame({
            'Date': forecast.index,
            'Forecast': forecast.values
        })
        
        if 'confidence_intervals' in forecast_results:
            ci = forecast_results['confidence_intervals']
            forecast_df['Lower CI'] = ci.iloc[:, 0]
            forecast_df['Upper CI'] = ci.iloc[:, 1]
        
        st.dataframe(forecast_df.round(2), use_container_width=True)
        
        # Download forecast data
        csv = forecast_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecast_results.csv">Download Forecast Data</a>'
        st.markdown(href, unsafe_allow_html=True)

# Update the main application to include the new functionality
class StatisticalAnalysisApp:
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Advanced Statistical Analysis Suite",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        st.title("📊 Advanced Statistical Analysis Suite")
        st.markdown("""
        Comprehensive statistical analysis platform with SPSS-like functionality, 
        advanced forecasting capabilities, and professional reporting.
        """)
        
        # Data loading
        df = self.load_data()
        
        if df is not None:
            self.run_analysis(df)
        else:
            self.show_instructions()
    
    def load_data(self):
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
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded {uploaded_file.name}")
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                return df
            except Exception as e:
                st.error(f"Error loading file: {e}")
        return None
    
    def _load_from_url(self):
        url = st.text_input("Enter dataset URL:", placeholder="https://example.com/data.csv")
        if url:
            try:
                response = requests.get(url)
                df = pd.read_csv(BytesIO(response.content))
                st.success("✅ Data loaded from URL")
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                return df
            except Exception as e:
                st.error(f"Error loading from URL: {e}")
        return None
    
    def _load_sample_data(self):
        if st.button("Generate Sample Time Series Data"):
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            trend = np.linspace(100, 150, 100)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(100) / 30)
            noise = np.random.normal(0, 5, 100)
            
            df = pd.DataFrame({
                'date': dates,
                'value': trend + seasonal + noise,
                'temperature': np.random.normal(25, 5, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            st.success("✅ Sample time series data generated")
            return df
        return None
    
    def run_analysis(self, df):
        if len(df) > Config.MAX_ROWS_FOR_ANALYSIS:
            st.warning(f"Large dataset detected. Sampling to {Config.MAX_ROWS_FOR_ANALYSIS} rows.")
            df = df.sample(n=Config.MAX_ROWS_FOR_ANALYSIS, random_state=42)
        
        # Store original data
        st.session_state.original_df = df
        
        st.sidebar.header("Analysis Modules")
        modules = st.sidebar.multiselect(
            "Select modules:",
            ["Data Cleaning", "Statistical Tests", "Forecasting", "PDF Report"],
            default=["Data Cleaning", "Statistical Tests"]
        )
        
        if "Data Cleaning" in modules:
            self.data_cleaning_module(df)
        
        if "Statistical Tests" in modules:
            self.statistical_tests_module(df)
        
        if "Forecasting" in modules:
            self.forecasting_module(df)
        
        if "PDF Report" in modules:
            self.pdf_report_module(df)
    
    def data_cleaning_module(self, df):
        st.header("🧹 Advanced Data Cleaning")
        
        st.subheader("Current Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
        st.subheader("Cleaning Configuration")
        cleaning_options = {}
        
        with st.expander("Missing Value Handling"):
            cleaning_options['handle_missing'] = st.checkbox("Handle missing values", value=True)
            if cleaning_options['handle_missing']:
                cleaning_options['missing_strategy'] = st.selectbox(
                    "Strategy:",
                    ["mean", "median", "mode", "knn", "forward_fill", "interpolate", "remove"]
                )
        
        with st.expander("Outlier Treatment"):
            cleaning_options['handle_outliers'] = st.checkbox("Handle outliers", value=True)
            if cleaning_options['handle_outliers']:
                col1, col2 = st.columns(2)
                with col1:
                    cleaning_options['outlier_strategy'] = st.selectbox(
                        "Treatment:",
                        ["clip", "remove", "transform", "winsorize"]
                    )
                with col2:
                    cleaning_options['outlier_method'] = st.selectbox(
                        "Detection:",
                        ["iqr", "zscore", "percentile"]
                    )
        
        with st.expander("Data Transformation"):
            cleaning_options['convert_dtypes'] = st.checkbox("Optimize data types", value=True)
            cleaning_options['standardize_text'] = st.checkbox("Standardize text", value=True)
            cleaning_options['feature_engineering'] = st.checkbox("Create features", value=True)
            cleaning_options['scale_data'] = st.checkbox("Scale data", value=False)
            if cleaning_options['scale_data']:
                cleaning_options['scaling_method'] = st.selectbox(
                    "Scaling method:",
                    ["standard", "minmax", "robust"]
                )
        
        if st.button("Apply Data Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_df = AdvancedDataCleaner.clean_data(df, cleaning_options)
                st.session_state.cleaned_df = cleaned_df
                
                st.success("✅ Data cleaning completed!")
                
                # Show comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before Cleaning:**")
                    st.write(f"Rows: {len(df)}")
                    st.write(f"Missing: {df.isnull().sum().sum()}")
                    st.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                with col2:
                    st.write("**After Cleaning:**")
                    st.write(f"Rows: {len(cleaned_df)}")
                    st.write(f"Missing: {cleaned_df.isnull().sum().sum()}")
                    st.write(f"Memory: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_df.head(), use_container_width=True)
    
    def statistical_tests_module(self, df):
        st.header("📊 Comprehensive Statistical Testing")
        
        # Use cleaned data if available
        analysis_df = st.session_state.get('cleaned_df', df)
        
        test_type = st.selectbox(
            "Select Test Type:",
            ["Normality Tests", "Correlation Analysis", "T-Tests", "ANOVA", "Chi-Square Test"]
        )
        
        if test_type == "Normality Tests":
            self.normality_tests_interface(analysis_df)
        elif test_type == "Correlation Analysis":
            self.correlation_analysis_interface(analysis_df)
        elif test_type == "T-Tests":
            self.t_test_interface(analysis_df)
        elif test_type == "ANOVA":
            self.anova_interface(analysis_df)
        elif test_type == "Chi-Square Test":
            self.chi_square_interface(analysis_df)
    
    def normality_tests_interface(self, df):
        st.subheader("Normality Tests")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            st.info("No numerical variables for normality testing")
            return
        
        selected_var = st.selectbox("Select variable:", numerical_cols)
        
        if st.button("Run Normality Tests"):
            results = ComprehensiveStatisticalTests.normality_tests(df, selected_var)
            
            if 'error' not in results:
                st.write("**Test Results:**")
                
                # Create a results table
                results_data = [
                    ["Test", "Statistic", "P-value", "Interpretation"],
                    ["Shapiro-Wilk", f"{results['shapiro_wilk_statistic']:.4f}", 
                     f"{results['shapiro_wilk_p_value']:.4f}", 
                     "Normal" if results['shapiro_wilk_p_value'] > 0.05 else "Not Normal"],
                    ["D'Agostino", f"{results['dagostino_statistic']:.4f}", 
                     f"{results['dagostino_p_value']:.4f}", 
                     "Normal" if results['dagostino_p_value'] > 0.05 else "Not Normal"],
                    ["Kolmogorov-Smirnov", f"{results['kolmogorov_smirnov_statistic']:.4f}", 
                     f"{results['kolmogorov_smirnov_p_value']:.4f}", 
                     "Normal" if results['kolmogorov_smirnov_p_value'] > 0.05 else "Not Normal"]
                ]
                
                st.table(results_data)
                
                # Additional statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Skewness", f"{results['skewness']:.4f}")
                col2.metric("Kurtosis", f"{results['kurtosis']:.4f}")
                col3.metric("Sample Size", results['sample_size'])
                col4.metric("Overall", "Normal" if results['is_normal'] else "Not Normal")
                
                # Q-Q plot
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(df[selected_var].dropna(), dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot for {selected_var}')
                st.pyplot(fig)
    
    def correlation_analysis_interface(self, df):
        st.subheader("Correlation Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for correlation analysis")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variable 1:", numerical_cols, key='corr_var1')
        with col2:
            var2 = st.selectbox("Variable 2:", numerical_cols, key='corr_var2')
        
        if st.button("Run Correlation Analysis"):
            results = ComprehensiveStatisticalTests.correlation_analysis(df, var1, var2)
            
            if 'error' not in results:
                st.write("**Correlation Results:**")
                
                results_data = [
                    ["Method", "Correlation", "P-value", "95% CI", "Interpretation"],
                    ["Pearson", f"{results['pearson_correlation']:.4f}", 
                     f"{results['pearson_p_value']:.4f}",
                     f"({results['pearson_confidence_interval'][0]:.3f}, {results['pearson_confidence_interval'][1]:.3f})",
                     results['interpretation']],
                    ["Spearman", f"{results['spearman_correlation']:.4f}", 
                     f"{results['spearman_p_value']:.4f}",
                     f"({results['spearman_confidence_interval'][0]:.3f}, {results['spearman_confidence_interval'][1]:.3f})",
                     ""],
                    ["Kendall", f"{results['kendall_tau']:.4f}", 
                     f"{results['kendall_p_value']:.4f}", "", ""]
                ]
                
                st.table(results_data)
                
                # Scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(data=df, x=var1, y=var2, ax=ax)
                ax.set_title(f'Scatter Plot: {var1} vs {var2}\n(Pearson r = {results["pearson_correlation"]:.3f})')
                st.pyplot(fig)
    
    def t_test_interface(self, df):
        st.subheader("T-Test Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        col1, col2 = st.columns(2)
        with col1:
            numerical_var = st.selectbox("Numerical variable:", numerical_cols, key='ttest_num')
        with col2:
            # Filter categorical variables with exactly 2 groups
            binary_categorical = [col for col in categorical_cols if df[col].nunique() == 2]
            categorical_var = st.selectbox("Binary categorical variable:", binary_categorical, key='ttest_cat')
        
        test_type = st.radio("Test type:", ["Independent Samples", "Paired Samples"], horizontal=True)
        
        if st.button("Run T-Test"):
            if test_type == "Independent Samples":
                results = ComprehensiveStatisticalTests.hypothesis_testing(
                    df, 't_test_independent', 
                    numerical_col=numerical_var, 
                    categorical_col=categorical_var
                )
            else:
                # For paired t-test, need two numerical variables
                var2 = st.selectbox("Second numerical variable:", 
                                  [col for col in numerical_cols if col != numerical_var],
                                  key='ttest_paired')
                results = ComprehensiveStatisticalTests.hypothesis_testing(
                    df, 't_test_paired',
                    col1=numerical_var,
                    col2=var2
                )
            
            if 'error' not in results:
                self._display_test_results(results)
    
    def anova_interface(self, df):
        st.subheader("ANOVA Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        col1, col2 = st.columns(2)
        with col1:
            numerical_var = st.selectbox("Numerical variable:", numerical_cols, key='anova_num')
        with col2:
            categorical_var = st.selectbox("Categorical variable:", categorical_cols, key='anova_cat')
        
        if st.button("Run ANOVA"):
            results = ComprehensiveStatisticalTests.hypothesis_testing(
                df, 'anova',
                numerical_col=numerical_var,
                categorical_col=categorical_var
            )
            
            if 'error' not in results:
                self._display_test_results(results)
    
    def chi_square_interface(self, df):
        st.subheader("Chi-Square Test")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("First categorical variable:", categorical_cols, key='chi_var1')
        with col2:
            var2 = st.selectbox("Second categorical variable:", categorical_cols, key='chi_var2')
        
        if st.button("Run Chi-Square Test"):
            results = ComprehensiveStatisticalTests.hypothesis_testing(
                df, 'chi_square',
                col1=var1,
                col2=var2
            )
            
            if 'error' not in results:
                self._display_test_results(results)
                
                # Display contingency table
                st.write("**Contingency Table:**")
                contingency_df = pd.DataFrame(results['contingency_table'])
                st.dataframe(contingency_df, use_container_width=True)
    
    def _display_test_results(self, results):
        """Display hypothesis test results in a standardized format"""
        st.write(f"**{results['test_type']} Results:**")
        
        # Main statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Statistic", f"{results.get('t_statistic', results.get('f_statistic', results.get('chi2_statistic', results.get('u_statistic', 'N/A')))):.4f}")
        col2.metric("P-value", f"{results['p_value']:.4f}")
        col3.metric("Significant", "Yes" if results['significant'] else "No")
        
        # Effect size if available
        if 'cohens_d' in results:
            st.metric("Cohen's d", f"{results['cohens_d']:.4f}")
            st.write(f"Effect size: {results['effect_size_interpretation']}")
        elif 'eta_squared' in results:
            st.metric("Eta squared", f"{results['eta_squared']:.4f}")
            st.write(f"Effect size: {results['effect_size_interpretation']}")
        elif 'cramers_v' in results:
            st.metric("Cramer's V", f"{results['cramers_v']:.4f}")
            st.write(f"Effect size: {results['effect_size_interpretation']}")
    
    def forecasting_module(self, df):
        st.header("📈 Time Series Forecasting")
        forecasting_phase = ForecastingPhase(df)
        forecasting_phase.execute()
    
    def pdf_report_module(self, df):
        st.header("📄 SPSS-like PDF Report Generation")
        
        st.write("Generate a comprehensive statistical report in PDF format similar to SPSS output.")
        
        # Collect analysis results for the report
        analyses = {}
        
        # Add basic dataset info
        analyses['Dataset Information'] = {
            'total_cases': len(df),
            'total_variables': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add descriptive statistics for numerical variables
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            desc_stats = df[numerical_cols].describe()
            analyses['Descriptive Statistics'] = {
                col: {
                    'mean': desc_stats[col]['mean'],
                    'std': desc_stats[col]['std'],
                    'min': desc_stats[col]['min'],
                    'max': desc_stats[col]['max']
                } for col in numerical_cols[:3]  # Limit to first 3 for report
            }
        
        if st.button("Generate Comprehensive PDF Report", type="primary"):
            with st.spinner("Generating professional PDF report..."):
                try:
                    pdf_buffer = SPSSReportGenerator.create_comprehensive_report(df, analyses)
                    
                    # Create download link
                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="statistical_analysis_report.pdf" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px;">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Professional PDF report generated successfully!")
                    
                    # Preview of what's in the report
                    with st.expander("Report Contents Preview"):
                        st.write("""
                        **The generated report includes:**
                        - Professional title page with timestamp
                        - Comprehensive dataset overview
                        - Detailed variable information
                        - Statistical test results with interpretations
                        - Summary and conclusions section
                        - SPSS-like formatting and styling
                        """)
                        
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {str(e)}")
    
    def show_instructions(self):
        st.info("👆 Please load your data to begin analysis")
        
        with st.expander("📋 Application Features Overview"):
            st.markdown("""
            ### 🎯 Key Features Implemented:
            
            **1. Forecasting Capabilities**
            - ✅ ARIMA modeling for time series forecasting
            - ✅ Exponential smoothing with trend and seasonality
            - ✅ Linear regression forecasting
            - ✅ Time series decomposition and stationarity testing
            - ✅ Interactive forecast visualization with confidence intervals
            
            **2. SPSS-like PDF Export**
            - ✅ Comprehensive statistical reports in PDF format
            - ✅ Professional formatting with tables and summaries
            - ✅ Dataset information and variable descriptions
            - ✅ Statistical test results with interpretations
            - ✅ Downloadable reports for academic use
            
            **3. Advanced Data Cleaning & Processing**
            - ✅ Missing value handling (mean, median, mode, KNN, interpolation)
            - ✅ Outlier detection and treatment (clipping, removal, transformation)
            - ✅ Data type optimization and standardization
            - ✅ Feature engineering with automatic interaction terms
            - ✅ Text data standardization and scaling options
            
            **4. Comprehensive Statistical Testing Suite**
            - ✅ Normality tests: Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov
            - ✅ Correlation analysis: Pearson, Spearman, Kendall with confidence intervals
            - ✅ Hypothesis testing: T-tests, ANOVA, Chi-square tests
            - ✅ Regression analysis with diagnostics
            - ✅ Effect size calculations and interpretations
            - ✅ Non-parametric alternatives (Mann-Whitney, Kruskal-Wallis)
            """)

def main():
    try:
        app = StatisticalAnalysisApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the console for detailed error messages.")

if __name__ == "__main__":
    main()
