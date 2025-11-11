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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import io
import joblib
from datetime import datetime
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    logger.warning("Pingouin not available. Install with: pip install pingouin")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")


class DataLoader:
    """Handles data loading from multiple sources with validation"""
    
    @staticmethod
    def load_from_url(url: str) -> pd.DataFrame:
        """Load data from URL with validation"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                return pd.read_csv(io.StringIO(response.text))
            elif url.endswith(('.xlsx', '.xls')):
                return pd.read_excel(io.BytesIO(response.content))
            else:
                # Auto-detect format
                try:
                    return pd.read_csv(io.StringIO(response.text))
                except:
                    return pd.read_excel(io.BytesIO(response.content))
                    
        except Exception as e:
            logger.error(f"Failed to load data from URL: {e}")
            raise Exception(f"Error loading data from URL: {e}")
    
    @staticmethod
    def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file with validation"""
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            logger.error(f"Failed to load uploaded file: {e}")
            raise Exception(f"Error loading uploaded file: {e}")
    
    @staticmethod
    def generate_sample_data(dataset_type: str = "mixed") -> pd.DataFrame:
        """Generate sample datasets for demonstration"""
        np.random.seed(42)
        n_samples = 200
        
        generators = {
            "time_series": DataLoader._generate_time_series_data,
            "categorical": DataLoader._generate_categorical_data,
            "numerical": DataLoader._generate_numerical_data,
            "mixed": DataLoader._generate_mixed_data
        }
        
        generator = generators.get(dataset_type, DataLoader._generate_mixed_data)
        return generator(n_samples)
    
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
            'volume': np.random.exponential(1000, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        })
    
    @staticmethod
    def _generate_categorical_data(n_samples: int) -> pd.DataFrame:
        """Generate categorical-rich sample data"""
        return pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_samples),
            'satisfaction_level': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples)
        })
    
    @staticmethod
    def _generate_numerical_data(n_samples: int) -> pd.DataFrame:
        """Generate numerical-rich sample data"""
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.uniform(-5, 5, n_samples),
            'feature_4': np.random.gamma(2, 2, n_samples),
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
            'temperature': np.random.normal(25, 5, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples),
            'customer_rating': np.random.uniform(1, 5, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'customer_value': np.random.exponential(5000, n_samples)
        })


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
    def prepare_features(df: pd.DataFrame, target_variable: str, problem_type: str):
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


class DatasetAnalyzer:
    """Analyzes dataset characteristics and types"""
    
    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> dict:
        """Comprehensive dataset analysis"""
        return {
            'basic_info': DatasetAnalyzer._get_basic_info(df),
            'data_quality': DatasetAnalyzer._assess_data_quality(df),
            'type_analysis': DatasetAnalyzer._analyze_data_types(df),
            'patterns': DatasetAnalyzer._detect_patterns(df)
        }
    
    @staticmethod
    def _get_basic_info(df: pd.DataFrame) -> dict:
        """Get basic dataset information"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'total_cells': len(df) * len(df.columns)
        }
    
    @staticmethod
    def _assess_data_quality(df: pd.DataFrame) -> dict:
        """Assess data quality metrics"""
        missing_values = df.isnull().sum()
        return {
            'missing_values_total': missing_values.sum(),
            'missing_values_percentage': (missing_values.sum() / (len(df) * len(df.columns))) * 100,
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
            'primary_type': DatasetAnalyzer._determine_primary_type(df, numerical_cols, categorical_cols, date_cols)
        }
    
    @staticmethod
    def _determine_primary_type(df, numerical_cols, categorical_cols, date_cols):
        """Determine primary dataset type"""
        if len(date_cols) > 0 and len(numerical_cols) > 0:
            return "Time Series"
        elif len(categorical_cols) > len(df.columns) * 0.6:
            return "Categorical"
        elif len(numerical_cols) > len(df.columns) * 0.6:
            return "Numerical"
        else:
            return "Mixed"
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> dict:
        """Detect data patterns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        patterns = {
            'outlier_pattern': DatasetAnalyzer._assess_outliers(df),
            'correlation_strength': DatasetAnalyzer._assess_correlations(df),
            'temporal_patterns': DatasetAnalyzer._check_temporal_patterns(df)
        }
        
        if len(numerical_cols) > 0:
            patterns.update({
                'skewed_columns': (df[numerical_cols].skew().abs() > 2).sum(),
                'high_variance_columns': (df[numerical_cols].var() > df[numerical_cols].var().quantile(0.75)).sum()
            })
        
        return patterns
    
    @staticmethod
    def _assess_outliers(df: pd.DataFrame) -> str:
        """Assess outlier presence"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return "No numerical columns"
        
        outlier_counts = []
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts.append(outliers)
        
        total_outliers = sum(outlier_counts)
        outlier_percentage = (total_outliers / (len(df) * len(numerical_cols))) * 100
        
        if outlier_percentage > 10:
            return "High"
        elif outlier_percentage > 5:
            return "Moderate"
        else:
            return "Low"
    
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


class DatasetCharacteristicsPhase(AnalysisPhase):
    """Handles dataset characteristics analysis"""
    
    def execute(self):
        st.markdown('<div class="phase-header"><h2>🔍 Dataset Characteristics Analysis</h2></div>', 
                   unsafe_allow_html=True)
        
        analysis = DatasetAnalyzer.analyze_dataset(self.df)
        
        # Display dataset type
        st.markdown(f"""
        <div class="dataset-type-box">
            <h3>📊 Dataset Type: {analysis['type_analysis']['primary_type']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        self._display_basic_info(analysis['basic_info'])
        self._display_data_quality(analysis['data_quality'])
        self._display_type_analysis(analysis['type_analysis'])
        self._display_patterns(analysis['patterns'])
        self._display_recommendations(analysis)
    
    def _display_basic_info(self, basic_info: dict):
        """Display basic dataset information"""
        st.subheader("📋 Basic Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{basic_info['rows']:,}")
        with col2:
            st.metric("Total Columns", basic_info['columns'])
        with col3:
            st.metric("Memory Usage", f"{basic_info['memory_usage_mb']:.2f} MB")
        with col4:
            st.metric("Duplicate Rows", basic_info['duplicate_rows'])
    
    def _display_data_quality(self, data_quality: dict):
        """Display data quality metrics"""
        st.subheader("📈 Data Quality Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completeness", f"{data_quality['completeness_score']:.1f}%")
        with col2:
            st.metric("Missing Values", f"{data_quality['missing_values_percentage']:.2f}%")
        with col3:
            st.metric("Duplicate Rows", f"{data_quality['duplicate_percentage']:.2f}%")
        with col4:
            st.metric("Columns with Missing", data_quality['columns_with_missing'])
        
        # Quality score
        quality_score = self._calculate_quality_score(data_quality)
        st.progress(quality_score / 100)
        st.write(f"**Overall Data Quality Score: {quality_score:.1f}/100**")
    
    def _display_type_analysis(self, type_analysis: dict):
        """Display data type analysis"""
        st.subheader("🔧 Data Type Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Numerical Columns", type_analysis['numerical_count'])
        with col2:
            st.metric("Categorical Columns", type_analysis['categorical_count'])
        with col3:
            st.metric("Date Columns", type_analysis['date_count'])
        with col4:
            st.metric("Primary Type", type_analysis['primary_type'])
        
        # Data type distribution chart
        self._plot_data_type_distribution(type_analysis)
    
    def _display_patterns(self, patterns: dict):
        """Display detected patterns"""
        st.subheader("🔄 Detected Patterns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Outlier Presence", patterns['outlier_pattern'])
        with col2:
            st.metric("Correlation Strength", patterns['correlation_strength'])
        with col3:
            st.metric("Temporal Patterns", patterns['temporal_patterns'])
    
    def _display_recommendations(self, analysis: dict):
        """Display analysis recommendations"""
        st.subheader("🎯 Recommended Analysis")
        
        recommendations = self._generate_recommendations(analysis)
        for rec in recommendations:
            st.markdown(f"✅ {rec}")
    
    def _plot_data_type_distribution(self, type_analysis: dict):
        """Plot data type distribution"""
        types = ['Numerical', 'Categorical', 'Date/Time']
        counts = [
            type_analysis['numerical_count'],
            type_analysis['categorical_count'], 
            type_analysis['date_count']
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        ax1.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Type Distribution')
        
        # Bar chart
        ax2.bar(types, counts, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Number of Columns')
        ax2.set_title('Data Type Count')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _calculate_quality_score(self, data_quality: dict) -> float:
        """Calculate overall data quality score"""
        completeness = data_quality['completeness_score']
        duplicate_score = 100 - data_quality['duplicate_percentage']
        missing_score = 100 - min(data_quality['missing_values_percentage'] * 2, 100)
        
        return (completeness * 0.4 + duplicate_score * 0.3 + missing_score * 0.3)
    
    def _generate_recommendations(self, analysis: dict) -> list:
        """Generate analysis recommendations"""
        recommendations = []
        primary_type = analysis['type_analysis']['primary_type']
        
        type_recommendations = {
            "Time Series": [
                "Time series decomposition and trend analysis",
                "Stationarity testing and forecasting models",
                "Seasonal pattern detection and analysis"
            ],
            "Categorical": [
                "Frequency analysis and cross-tabulations",
                "Chi-square tests for association",
                "Categorical encoding for modeling"
            ],
            "Numerical": [
                "Correlation analysis and heatmaps",
                "Distribution analysis and normality testing",
                "Regression modeling and feature importance"
            ],
            "Mixed": [
                "Comprehensive exploratory data analysis",
                "Multiple data type-specific analyses",
                "Integrated modeling approaches"
            ]
        }
        
        recommendations.extend(type_recommendations.get(primary_type, []))
        
        # Quality-based recommendations
        quality_score = self._calculate_quality_score(analysis['data_quality'])
        if quality_score < 80:
            recommendations.append("Data cleaning and preprocessing")
        
        return recommendations


class DataValidationPhase(AnalysisPhase):
    """Handles data validation and quality assessment"""
    
    def execute(self):
        st.markdown('<div class="phase-header"><h2>📋 Data Validation & Quality Assessment</h2></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("1.1 Data Profiling")
        self._display_data_profiling()
        
        st.subheader("1.2 Missing Data Analysis")
        self._analyze_missing_data()
        
        st.subheader("1.3 Outlier Detection")
        self._detect_outliers()
        
        st.subheader("1.4 Data Quality Summary")
        self._display_quality_summary()
    
    def _display_data_profiling(self):
        """Display data profiling information"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(self.df))
            st.metric("Total Features", len(self.df.columns))
        
        with col2:
            st.metric("Numerical Features", len(self.numerical_cols))
            st.metric("Categorical Features", len(self.categorical_cols))
        
        with col3:
            duplicate_count = self.df.duplicated().sum()
            st.metric("Duplicate Records", duplicate_count)
            memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
        
        with col4:
            total_missing = self.df.isnull().sum().sum()
            st.metric("Total Missing Values", total_missing)
            completeness = ((len(self.df) - self.df.isnull().sum()) / len(self.df) * 100).mean()
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Data preview
        with st.expander("🔍 Data Preview"):
            tab1, tab2 = st.tabs(["First 10 Rows", "Data Types"])
            with tab1:
                st.dataframe(self.df.head(10), use_container_width=True)
            with tab2:
                dtype_info = pd.DataFrame({
                    'Column': self.df.columns,
                    'Data Type': self.df.dtypes,
                    'Non-Null Count': self.df.count()
                })
                st.dataframe(dtype_info, use_container_width=True)
    
    def _analyze_missing_data(self):
        """Analyze missing data patterns"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Percentage', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Missing Data Summary:**")
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        with col2:
            if missing_data.sum() > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                missing_plot_data = missing_df[missing_df['Missing Count'] > 0].head(10)
                sns.barplot(data=missing_plot_data, y=missing_plot_data.index, x='Missing Percentage', ax=ax)
                ax.set_title('Top 10 Variables with Missing Data')
                st.pyplot(fig)
            else:
                st.success("✅ No missing values found!")
        
        # Missing data heatmap
        if missing_data.sum() > 0:
            st.write("**Missing Data Pattern Heatmap:**")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
            st.pyplot(fig)
    
    def _detect_outliers(self):
        """Detect and visualize outliers"""
        if not self.numerical_cols:
            st.info("No numerical columns for outlier detection")
            return
        
        st.write("**Outlier Detection Summary:**")
        
        outlier_results = {}
        for col in self.numerical_cols:
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            z_outliers = (z_scores > 3).sum()
            
            # IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
            
            outlier_results[col] = {
                'Z-score Outliers': z_outliers,
                'IQR Outliers': iqr_outliers,
                'Outlier %': max(z_outliers, iqr_outliers) / len(self.df) * 100
            }
        
        outlier_df = pd.DataFrame(outlier_results).T
        outlier_df['Outlier Status'] = outlier_df['Outlier %'].apply(
            lambda x: 'High' if x > 5 else 'Medium' if x > 1 else 'Low'
        )
        
        st.dataframe(outlier_df.round(2), use_container_width=True)
        
        # Detailed visualization
        selected_var = st.selectbox("Select variable for detailed visualization:", 
                                   self.numerical_cols, key="outlier_viz")
        
        if selected_var:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=self.df[selected_var], ax=ax)
                ax.set_title(f'Box Plot - {selected_var}')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                Q1 = self.df[selected_var].quantile(0.25)
                Q3 = self.df[selected_var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                sns.histplot(self.df[selected_var], kde=True, ax=ax)
                ax.axvline(lower_bound, color='red', linestyle='--', alpha=0.7, label='Lower Bound')
                ax.axvline(upper_bound, color='red', linestyle='--', alpha=0.7, label='Upper Bound')
                ax.set_title(f'Distribution with Outlier Boundaries - {selected_var}')
                ax.legend()
                st.pyplot(fig)
    
    def _display_quality_summary(self):
        """Display comprehensive quality summary"""
        analysis = DatasetAnalyzer.analyze_dataset(self.df)
        quality_metrics = analysis['data_quality']
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_data = [
            ("Completeness", quality_metrics['completeness_score'], "#3498db"),
            ("Uniqueness", 100 - quality_metrics['duplicate_percentage'], "#2ecc71"),
            ("Validity", 100 - min(quality_metrics['missing_values_percentage'] * 2, 100), "#f39c12"),
            ("Consistency", self._calculate_consistency_score(), "#e74c3c")
        ]
        
        for (title, value, color), col in zip(metrics_data, [col1, col2, col3, col4]):
            with col:
                st.metric(title, f"{value:.1f}%")
        
        overall_quality = np.mean([metric[1] for metric in metrics_data])
        st.write(f"**Overall Data Quality Score: {overall_quality:.1f}%**")
        
        if overall_quality >= 90:
            st.success("✅ Excellent data quality - ready for analysis!")
        elif overall_quality >= 75:
            st.warning("⚠️ Good data quality - some improvements possible")
        else:
            st.error("❌ Poor data quality - significant cleaning needed")
    
    def _calculate_consistency_score(self) -> float:
        """Calculate data consistency score"""
        score = 100
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                if unique_count > len(self.df) * 0.5:
                    score -= 5
        return max(score, 0)


class ExploratoryAnalysisPhase(AnalysisPhase):
    """Handles exploratory data analysis"""
    
    def execute(self):
        st.markdown('<div class="phase-header"><h2>🔍 Exploratory Data Analysis</h2></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("2.1 Pattern & Trend Analysis")
        self._pattern_trend_analysis()
        
        if self._has_date_column():
            st.subheader("2.2 Time Series Analysis")
            self._time_series_analysis()
        
        st.subheader("2.3 Correlation Analysis")
        self._correlation_analysis()
        
        st.subheader("2.4 Distribution Analysis")
        self._distribution_analysis()
        
        st.subheader("2.5 Multivariate Analysis")
        self._multivariate_analysis()
        
        if PINGOUIN_AVAILABLE:
            st.subheader("2.6 Advanced Statistical Tests")
            self._advanced_statistical_tests()
    
    def _pattern_trend_analysis(self):
        """Comprehensive pattern and trend analysis"""
        if not self.numerical_cols:
            st.info("No numerical columns for trend analysis")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            trend_var = st.selectbox("Select variable for trend analysis:", self.numerical_cols)
        with col2:
            confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95)
        
        st.subheader(f"Trend Analysis for {trend_var}")
        
        if len(self.df) > 10:
            self._rolling_analysis(trend_var, confidence_level)
        
        self._linear_trend_analysis(trend_var, confidence_level)
        
        if len(self.df) >= 50:
            self._seasonal_analysis(trend_var)
        
        if self._has_date_column() and PROPHET_AVAILABLE:
            self._prophet_forecasting(trend_var)
    
    def _rolling_analysis(self, variable: str, confidence_level: float):
        """Rolling analysis with confidence intervals"""
        if not self._has_date_column():
            self.df['index'] = range(len(self.df))
            x_var = 'index'
        else:
            date_col = self.date_cols[0]
            x_var = date_col
        
        window_size = min(30, len(self.df) // 10)
        rolling_mean = self.df[variable].rolling(window=window_size, center=True).mean()
        rolling_std = self.df[variable].rolling(window=window_size, center=True).std()
        
        z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_upper = rolling_mean + z_value * rolling_std / np.sqrt(window_size)
        ci_lower = rolling_mean - z_value * rolling_std / np.sqrt(window_size)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(self.df[x_var], self.df[variable], alpha=0.3, label='Actual Data', color='lightblue')
        ax1.plot(self.df[x_var], rolling_mean, label=f'Rolling Mean (window={window_size})', color='red', linewidth=2)
        ax1.fill_between(self.df[x_var], ci_lower, ci_upper, alpha=0.2, 
                        label=f'{confidence_level:.0%} Confidence Interval', color='red')
        ax1.set_title(f'Rolling Mean with {confidence_level:.0%} Confidence Interval')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.df[x_var], rolling_std, color='green', linewidth=2)
        ax2.set_title('Rolling Standard Deviation')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _linear_trend_analysis(self, variable: str, confidence_level: float):
        """Linear trend analysis with confidence intervals"""
        x = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df[variable].values
        
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        
        residuals = y - y_pred
        residual_std = np.std(residuals)
        n = len(x)
        x_mean = np.mean(x)
        
        se = residual_std * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
        t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, n - 2)
        ci = t_value * se
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(x, y, alpha=0.6, label='Actual Data', color='lightblue')
        ax.plot(x, y_pred, color='red', linewidth=2, label='Linear Trend')
        ax.fill_between(x.flatten(), y_pred - ci.flatten(), y_pred + ci.flatten(), 
                       alpha=0.3, label=f'{confidence_level:.0%} Confidence Band', color='red')
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel(variable)
        ax.set_title(f'Linear Trend Analysis with {confidence_level:.0%} Confidence Band')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Statistical summary
        slope = model.coef_[0]
        r_squared = model.score(x, y)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trend Slope", f"{slope:.4f}")
        with col2:
            st.metric("R-squared", f"{r_squared:.4f}")
        with col3:
            trend_strength = "Strong" if abs(slope) > np.std(y) * 0.1 else "Weak"
            st.metric("Trend Strength", trend_strength)
        with col4:
            direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            st.metric("Direction", direction)
    
    def _seasonal_analysis(self, variable: str):
        """Seasonal pattern detection"""
        try:
            if self._has_date_column():
                date_col = self.date_cols[0]
                self.df = self.df.sort_values(date_col)
                ts_data = self.df.set_index(date_col)[variable]
            else:
                ts_data = pd.Series(self.df[variable].values, 
                                  index=pd.date_range(start='2020-01-01', periods=len(self.df), freq='D'))
            
            seasonal_period = min(12, len(ts_data) // 4)
            decomposition = seasonal_decompose(ts_data.dropna(), period=seasonal_period, 
                                             extrapolate_trend='freq')
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
            
            decomposition.observed.plot(ax=ax1, title='Original')
            decomposition.trend.plot(ax=ax2, title='Trend')
            decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            decomposition.resid.plot(ax=ax4, title='Residual')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Seasonal decomposition failed: {e}")
    
    def _prophet_forecasting(self, variable: str):
        """Prophet time series forecasting"""
        if not PROPHET_AVAILABLE:
            return
            
        if st.button("Run Prophet Forecast"):
            try:
                date_col = self.date_cols[0]
                prophet_df = self.df[[date_col, variable]].dropna().copy()
                prophet_df.columns = ['ds', 'y']
                
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                fig1 = model.plot(forecast)
                plt.title(f'Prophet Forecast for {variable}')
                st.pyplot(fig1)
                
            except Exception as e:
                st.error(f"Prophet forecasting failed: {e}")
    
    def _correlation_analysis(self):
        """Correlation analysis with heatmaps"""
        if len(self.numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for correlation analysis")
            return
        
        # Static heatmap
        st.write("**Correlation Heatmap**")
        corr_matrix = self.df[self.numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Interactive heatmap with confidence intervals
        st.write("**Interactive Correlation Matrix**")
        
        corr_matrix_plotly = np.zeros((len(self.numerical_cols), len(self.numerical_cols)))
        p_value_matrix = np.zeros_like(corr_matrix_plotly)
        
        for i, col1 in enumerate(self.numerical_cols):
            for j, col2 in enumerate(self.numerical_cols):
                if i == j:
                    corr_matrix_plotly[i, j] = 1.0
                    continue
                
                valid_data = self.df[[col1, col2]].dropna()
                if len(valid_data) < 3:
                    continue
                
                corr, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                corr_matrix_plotly[i, j] = corr
                p_value_matrix[i, j] = p_value
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix_plotly,
            x=self.numerical_cols,
            y=self.numerical_cols,
            colorscale='RdBu_r',
            zmid=0
        ))
        
        fig.update_layout(title='Interactive Correlation Matrix', width=800, height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    def _distribution_analysis(self):
        """Distribution analysis with confidence intervals"""
        if not self.numerical_cols:
            return
        
        selected_var = st.selectbox("Select variable for distribution analysis:", self.numerical_cols)
        confidence_level = st.slider("Confidence Level for Mean", 0.80, 0.99, 0.95, key="dist_ci")
        
        data = self.df[selected_var].dropna()
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        if n > 1:
            se = std / np.sqrt(n)
            t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se
        else:
            ci_lower = ci_upper = mean
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(data, kde=True, ax=ax1)
        ax1.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label=f'CI Lower: {ci_lower:.2f}')
        ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=2, label=f'CI Upper: {ci_upper:.2f}')
        ax1.fill_betweenx([0, ax1.get_ylim()[1]], ci_lower, ci_upper, alpha=0.2, color='orange')
        ax1.set_title(f'Distribution with {confidence_level:.0%} CI')
        ax1.legend()
        
        sns.boxplot(y=data, ax=ax2)
        ax2.set_title('Box Plot')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{mean:.2f}")
        with col2:
            st.metric("Standard Deviation", f"{std:.2f}")
        with col3:
            st.metric(f"CI Lower ({confidence_level:.0%})", f"{ci_lower:.2f}")
        with col4:
            st.metric(f"CI Upper ({confidence_level:.0%})", f"{ci_upper:.2f}")
    
    def _multivariate_analysis(self):
        """Multivariate analysis using PCA"""
        if len(self.numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for multivariate analysis")
            return
        
        st.subheader("Principal Component Analysis (PCA)")
        
        X = self.df[self.numerical_cols].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        
        pca = PCA()
        principal_components = pca.fit_transform(X_scaled)
        
        exp_variance = pca.explained_variance_ratio_
        cum_variance = np.cumsum(exp_variance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            components = range(1, len(exp_variance) + 1)
            ax.bar(components, exp_variance, alpha=0.6, color='skyblue', label='Individual')
            ax.step(components, cum_variance, where='mid', label='Cumulative', color='red')
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Scree Plot')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(len(exp_variance))],
                index=self.numerical_cols
            )
            st.write("**PCA Loadings (First 5 Components):**")
            st.dataframe(loadings.iloc[:, :5].round(3))
        
        n_components_95 = np.argmax(cum_variance >= 0.95) + 1
        st.info(f"**Components needed for 95% variance: {n_components_95}**")
    
    def _advanced_statistical_tests(self):
        """Advanced statistical tests using Pingouin"""
        if not PINGOUIN_AVAILABLE:
            return
            
        st.subheader("Advanced Statistical Tests")
        
        test_type = st.selectbox(
            "Select Statistical Test:",
            ["ANOVA", "Bayesian T-test", "Mann-Whitney U", "Partial Correlation"]
        )
        
        if test_type == "ANOVA":
            self._pingouin_anova()
        elif test_type == "Bayesian T-test":
            self._pingouin_bayesian_ttest()
        elif test_type == "Mann-Whitney U":
            self._pingouin_mannwhitney()
        elif test_type == "Partial Correlation":
            self._pingouin_partial_corr()
    
    def _pingouin_anova(self):
        """One-way ANOVA using Pingouin"""
        dv = st.selectbox("Dependent Variable:", self.numerical_cols, key="anova_dv")
        between = st.selectbox("Between-subjects factor:", self.categorical_cols, key="anova_between")
        
        if dv and between:
            try:
                anova_result = pg.anova(data=self.df, dv=dv, between=between, detailed=True)
                st.write("**ANOVA Results:**")
                st.dataframe(anova_result.round(4))
            except Exception as e:
                st.error(f"ANOVA failed: {e}")
    
    def _pingouin_bayesian_ttest(self):
        """Bayesian T-test using Pingouin"""
        variable = st.selectbox("Variable:", self.numerical_cols, key="bayes_var")
        group_var = st.selectbox("Grouping variable:", self.categorical_cols, key="bayes_group")
        
        if variable and group_var:
            groups = self.df[group_var].unique()
            if len(groups) == 2:
                group1 = self.df[self.df[group_var] == groups[0]][variable]
                group2 = self.df[self.df[group_var] == groups[1]][variable]
                
                try:
                    bayes_ttest = pg.ttest(group1, group2, paired=False)
                    st.write("**Bayesian T-test Results:**")
                    st.dataframe(bayes_ttest.round(4))
                except Exception as e:
                    st.error(f"Bayesian t-test failed: {e}")
    
    def _pingouin_mannwhitney(self):
        """Mann-Whitney U test using Pingouin"""
        variable = st.selectbox("Variable:", self.numerical_cols, key="mw_var")
        group_var = st.selectbox("Grouping variable:", self.categorical_cols, key="mw_group")
        
        if variable and group_var:
            groups = self.df[group_var].unique()
            if len(groups) == 2:
                group1 = self.df[self.df[group_var] == groups[0]][variable]
                group2 = self.df[self.df[group_var] == groups[1]][variable]
                
                try:
                    mw_test = pg.mwu(group1, group2)
                    st.write("**Mann-Whitney U Test Results:**")
                    st.dataframe(mw_test.round(4))
                except Exception as e:
                    st.error(f"Mann-Whitney test failed: {e}")
    
    def _pingouin_partial_corr(self):
        """Partial correlation using Pingouin"""
        var1 = st.selectbox("Variable 1:", self.numerical_cols, key="pcorr_var1")
        var2 = st.selectbox("Variable 2:", self.numerical_cols, key="pcorr_var2")
        covar = st.multiselect("Covariates:", 
                              [v for v in self.numerical_cols if v not in [var1, var2]],
                              key="pcorr_covar")
        
        if var1 and var2:
            try:
                if covar:
                    pcorr = pg.partial_corr(data=self.df, x=var1, y=var2, covar=covar)
                else:
                    pcorr = pg.corr(self.df[var1], self.df[var2])
                
                st.write("**Partial Correlation Results:**")
                st.dataframe(pcorr.round(4))
            except Exception as e:
                st.error(f"Partial correlation failed: {e}")
    
    def _time_series_analysis(self):
        """Time series stationarity analysis"""
        if not self._has_date_column():
            return
            
        date_col = self.date_cols[0]
        numerical_var = st.selectbox("Select variable for time series:", self.numerical_cols)
        
        self.df = self.df.sort_values(date_col)
        data = self.df[numerical_var].dropna()
        
        adf_result = adfuller(data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
        with col2:
            st.metric("P-value", f"{adf_result[1]:.4f}")
        with col3:
            is_stationary = "Yes" if adf_result[1] < 0.05 else "No"
            st.metric("Stationary", is_stationary)
    
    def _has_date_column(self) -> bool:
        """Check if dataframe has date columns"""
        return len(self.date_cols) > 0


class ModelEvaluationPhase(AnalysisPhase):
    """Handles model evaluation and validation"""
    
    def execute(self):
        st.markdown('<div class="phase-header"><h2>🤖 Model Evaluation & Validation</h2></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("Model Configuration")
        
        problem_type = st.radio(
            "Select Problem Type:", 
            ["Classification", "Regression"],
            horizontal=True
        )
        
        # Dynamic target variable selection
        if problem_type == "Classification":
            target_options = [col for col in self.df.columns if self.df[col].nunique() <= 10 or self.df[col].dtype == 'object']
        else:
            target_options = self.numerical_cols
        
        if not target_options:
            st.error("No suitable target variables found")
            return
            
        target_var = st.selectbox("Select Target Variable:", target_options)
        
        if not target_var:
            return
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_data(self.df, target_var)
        
        # Model selection
        if problem_type == "Classification":
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Logistic Regression", "Support Vector Machine"]
            )
        else:
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Linear Regression", "Support Vector Regression"]
            )
        
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        if st.button("Train and Evaluate Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    X, y = preprocessor.prepare_features(df_processed, target_var, problem_type)
                    
                    if X.empty:
                        st.error("No valid features available")
                        return
                    
                    test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42,
                        stratify=y if problem_type == "Classification" else None
                    )
                    
                    if problem_type == "Classification":
                        results = self._evaluate_classification_model(
                            model_choice, X_train, X_test, y_train, y_test, cv_folds
                        )
                    else:
                        results = self._evaluate_regression_model(
                            model_choice, X_train, X_test, y_train, y_test, cv_folds
                        )
                    
                    self._display_model_results(results, problem_type)
                    
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")
    
    def _evaluate_classification_model(self, model_choice, X_train, X_test, y_train, y_test, cv_folds):
        """Evaluate classification models"""
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Support Vector Machine": SVC(random_state=42, probability=True)
        }
        
        model = models.get(model_choice, RandomForestClassifier(random_state=42))
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        return {
            'model': model,
            'model_name': model_choice,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': X_train.columns.tolist()
        }
    
    def _evaluate_regression_model(self, model_choice, X_train, X_test, y_train, y_test, cv_folds):
        """Evaluate regression models"""
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Support Vector Regression": SVR()
        }
        
        model = models.get(model_choice, RandomForestRegressor(random_state=42))
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        
        return {
            'model': model,
            'model_name': model_choice,
            'y_test': y_test,
            'y_pred': y_pred,
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': X_train.columns.tolist()
        }
    
    def _display_model_results(self, results, problem_type):
        """Display model evaluation results"""
        st.subheader("📊 Model Performance Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if problem_type == "Classification":
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{results['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{results['f1']:.3f}")
        else:
            with col1:
                st.metric("R² Score", f"{results['r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{results['rmse']:.3f}")
            with col3:
                st.metric("MAE", f"{results['mae']:.3f}")
            with col4:
                st.metric("MAPE", f"{results['mape']:.1f}%")
        
        # Cross-validation results
        st.write("**Cross-Validation Results:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Mean CV Score: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
        with col2:
            for i, score in enumerate(results['cv_scores']):
                st.write(f"Fold {i+1}: {score:.3f}")
        
        # Visualizations
        if problem_type == "Classification":
            self._plot_classification_results(results)
        else:
            self._plot_regression_results(results)
        
        # Feature importance
        self._plot_feature_importance(results)
    
    def _plot_classification_results(self, results):
        """Plot classification results"""
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            if results['y_pred_proba'] is not None and len(np.unique(results['y_test'])) == 2:
                fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'][:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
    
    def _plot_regression_results(self, results):
        """Plot regression results"""
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_test'], results['y_pred'], alpha=0.6)
            ax.plot([results['y_test'].min(), results['y_test'].max()], 
                   [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted')
            st.pyplot(fig)
        
        with col2:
            residuals = results['y_test'] - results['y_pred']
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_pred'], residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            st.pyplot(fig)
    
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
                ax.set_title('Feature Importance Plot')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write("**Top 10 Features:**")
                st.dataframe(fi_df.sort_values('importance', ascending=False).head(10))
                
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.warning(f"Could not compute feature importance: {str(e)}")


class StatisticalAnalysisApp:
    """Main application class orchestrating all analysis phases"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_css()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Advanced Statistical Analysis Suite",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_css(self):
        """Setup custom CSS styling"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .phase-header {
                background: linear-gradient(90deg, #1f77b4, #ff7f0e);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            .dataset-type-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            .upload-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }
            .data-source-tab {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        st.markdown('<div class="main-header">📊 Advanced Statistical Analysis Suite</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Comprehensive statistical analysis platform with dataset type detection, 
        exploratory analysis, and machine learning model evaluation.
        """)
        
        # Data loading section
        df = self.load_data()
        
        if df is not None:
            self.run_analysis(df)
        else:
            self.show_instructions()
    
    def load_data(self):
        """Handle data loading from various sources"""
        st.markdown('<div class="upload-section"><h3>📁 Data Source Selection</h3></div>', 
                   unsafe_allow_html=True)
        
        data_source = st.radio(
            "Choose your data source:",
            ["Upload File", "URL", "Sample Data"],
            horizontal=True
        )
        
        df = None
        
        if data_source == "Upload File":
            df = self._load_from_upload()
        elif data_source == "URL":
            df = self._load_from_url()
        elif data_source == "Sample Data":
            df = self._load_sample_data()
        
        return df
    
    def _load_from_upload(self):
        """Load data from file upload"""
        st.markdown('<div class="data-source-tab"><h4>📤 File Upload</h4></div>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (XLSX, XLS)"
        )
        
        if uploaded_file is not None:
            try:
                df = DataLoader.load_from_uploaded_file(uploaded_file)
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                return df
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
        
        return None
    
    def _load_from_url(self):
        """Load data from URL"""
        st.markdown('<div class="data-source-tab"><h4>🌐 URL Data Source</h4></div>', 
                   unsafe_allow_html=True)
        
        url = st.text_input(
            "Enter dataset URL:",
            placeholder="https://example.com/data.csv or https://example.com/data.xlsx",
            help="Supports CSV and Excel files from public URLs"
        )
        
        if url:
            try:
                with st.spinner("Downloading data from URL..."):
                    df = DataLoader.load_from_url(url)
                st.success("✅ Successfully loaded data from URL")
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                return df
            except Exception as e:
                st.error(f"❌ Error loading from URL: {e}")
        
        return None
    
    def _load_sample_data(self):
        """Load sample data"""
        st.markdown('<div class="data-source-tab"><h4>📊 Sample Data</h4></div>', 
                   unsafe_allow_html=True)
        
        dataset_type = st.selectbox(
            "Select sample dataset type:",
            ["Mixed", "Time Series", "Categorical", "Numerical"],
            help="Choose the type of sample data to generate"
        )
        
        if st.button("Generate Sample Dataset"):
            with st.spinner("Generating sample data..."):
                df = DataLoader.generate_sample_data(dataset_type.lower().replace(" ", "_"))
            st.success("✅ Sample dataset generated successfully")
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        
        return None
    
    def run_analysis(self, df):
        """Run selected analysis phases"""
        # Phase selection
        st.sidebar.header("🔬 Analysis Configuration")
        phases = st.sidebar.multiselect(
            "Select analysis phases to run:",
            ["Dataset Characteristics", "Data Validation", "Exploratory Analysis", "Model Evaluation"],
            default=["Dataset Characteristics", "Data Validation", "Exploratory Analysis"]
        )
        
        # Initialize analysis phases
        analysis_phases = {
            "Dataset Characteristics": DatasetCharacteristicsPhase(df),
            "Data Validation": DataValidationPhase(df),
            "Exploratory Analysis": ExploratoryAnalysisPhase(df),
            "Model Evaluation": ModelEvaluationPhase(df)
        }
        
        # Execute selected phases
        for phase_name in phases:
            phase = analysis_phases.get(phase_name)
            if phase:
                phase.execute()
    
    def show_instructions(self):
        """Show application instructions"""
        st.info("👆 Please select a data source and load your data to begin analysis")
        
        with st.expander("📋 Application Instructions"):
            st.markdown("""
            **How to use this application:**
            
            1. **Load Your Data**: Choose from file upload, URL, or sample data
            2. **Select Analysis Phases**: Choose which analyses to run from the sidebar
            3. **Explore Results**: Interactive visualizations and detailed insights
            
            **Supported Analyses:**
            - **Dataset Characteristics**: Automatic type detection and data profiling
            - **Data Validation**: Quality assessment and outlier detection
            - **Exploratory Analysis**: Statistical tests, correlations, and patterns
            - **Model Evaluation**: Machine learning model training and validation
            
            **Key Features:**
            - Automatic dataset type detection
            - Comprehensive statistical testing
            - Interactive visualizations
            - Machine learning model evaluation
            - Professional reporting
            """)


def main():
    """Main entry point"""
    app = StatisticalAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()
