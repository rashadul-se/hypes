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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc,
                           mean_absolute_error, mean_squared_error, r2_score,
                           mean_absolute_percentage_error)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import io
from urllib.parse import urlparse
import openpyxl
import joblib
from datetime import datetime
import re
warnings.filterwarnings('ignore')

# Import additional statistical packages
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Advanced Statistical Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .characteristic-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .trend-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .model-box {
        background-color: #f8f0ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #8e44ad;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .ci-low { color: #e74c3c; font-weight: bold; }
    .ci-high { color: #27ae60; font-weight: bold; }
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
    .stat-test-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 0.5rem 0;
    }
    .significant {
        color: #e74c3c;
        font-weight: bold;
    }
    .not-significant {
        color: #27ae60;
        font-weight: bold;
    }
    .feature-importance {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .dataset-characteristic {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

class DatasetTypeDetector:
    """Detect and characterize dataset types and characteristics"""
    
    @staticmethod
    def detect_dataset_type(df):
        """Detect the primary type of dataset"""
        characteristics = {
            'has_dates': False,
            'has_categorical': False,
            'has_numerical': False,
            'has_text': False,
            'has_geospatial': False,
            'is_time_series': False,
            'is_tabular': True,  # Default assumption
            'is_relational': False,
            'is_transactional': False,
            'has_missing_values': False
        }
        
        # Basic type detection
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        characteristics.update({
            'has_dates': len(date_cols) > 0,
            'has_categorical': len(categorical_cols) > 0,
            'has_numerical': len(numerical_cols) > 0,
            'has_missing_values': df.isnull().sum().sum() > 0
        })
        
        # Detect text columns
        text_columns = []
        for col in categorical_cols:
            if df[col].astype(str).str.len().mean() > 50:  # Average length > 50 characters
                text_columns.append(col)
        characteristics['has_text'] = len(text_columns) > 0
        
        # Detect time series
        if len(date_cols) > 0 and len(numerical_cols) > 0:
            characteristics['is_time_series'] = True
        
        # Detect geospatial data (simple heuristic)
        geo_keywords = ['lat', 'lon', 'longitude', 'latitude', 'address', 'city', 'country', 'zip']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in geo_keywords):
                characteristics['has_geospatial'] = True
                break
        
        # Determine primary dataset type
        dataset_type = "Tabular"
        if characteristics['is_time_series']:
            dataset_type = "Time Series"
        elif characteristics['has_geospatial']:
            dataset_type = "Geospatial"
        elif len(text_columns) > len(df.columns) * 0.5:  # More than 50% text columns
            dataset_type = "Text Data"
        elif len(date_cols) > 0 and len(numerical_cols) > 2:
            dataset_type = "Time Series"
        
        return dataset_type, characteristics
    
    @staticmethod
    def analyze_dataset_characteristics(df):
        """Analyze detailed dataset characteristics"""
        analysis = {
            'basic_info': {},
            'data_quality': {},
            'statistical_properties': {},
            'patterns_detected': {}
        }
        
        # Basic information
        analysis['basic_info'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'total_cells': len(df) * len(df.columns)
        }
        
        # Data quality metrics
        missing_values = df.isnull().sum()
        analysis['data_quality'] = {
            'missing_values_total': missing_values.sum(),
            'missing_values_percentage': (missing_values.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': (missing_values > 0).sum(),
            'completeness_score': ((len(df) - df.isnull().sum()) / len(df) * 100).mean(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Statistical properties
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            numerical_stats = df[numerical_cols].describe()
            analysis['statistical_properties'] = {
                'numerical_columns': len(numerical_cols),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
                'skewness_high': (df[numerical_cols].skew().abs() > 2).sum(),
                'high_variance_columns': (df[numerical_cols].var() > df[numerical_cols].var().quantile(0.75)).sum()
            }
        
        # Pattern detection
        analysis['patterns_detected'] = {
            'potential_outliers': DatasetTypeDetector._detect_outlier_patterns(df),
            'correlation_strength': DatasetTypeDetector._assess_correlation_strength(df),
            'temporal_patterns': DatasetTypeDetector._detect_temporal_patterns(df),
            'categorical_dominance': DatasetTypeDetector._assess_categorical_dominance(df)
        }
        
        return analysis
    
    @staticmethod
    def _detect_outlier_patterns(df):
        """Detect outlier patterns in numerical columns"""
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
            return "High outlier presence"
        elif outlier_percentage > 5:
            return "Moderate outlier presence"
        else:
            return "Low outlier presence"
    
    @staticmethod
    def _assess_correlation_strength(df):
        """Assess overall correlation strength"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return "Insufficient numerical data"
        
        corr_matrix = df[numerical_cols].corr().abs()
        mean_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if mean_correlation > 0.7:
            return "Strong correlations"
        elif mean_correlation > 0.3:
            return "Moderate correlations"
        else:
            return "Weak correlations"
    
    @staticmethod
    def _detect_temporal_patterns(df):
        """Detect temporal patterns"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            return "No temporal data"
        
        # Check if dates are sequential and have regular intervals
        for col in date_cols:
            if df[col].is_monotonic_increasing:
                return "Sequential temporal data"
        return "Non-sequential temporal data"
    
    @staticmethod
    def _assess_categorical_dominance(df):
        """Assess categorical data dominance"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return "No categorical data"
        
        categorical_ratio = len(categorical_cols) / len(df.columns)
        if categorical_ratio > 0.7:
            return "Categorical dominant"
        elif categorical_ratio > 0.3:
            return "Mixed categorical/numerical"
        else:
            return "Numerical dominant"

class DataLoader:
    """Class to handle multiple data loading methods"""
    
    @staticmethod
    def load_from_url(url):
        """Load data from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if url.endswith('.csv'):
                return pd.read_csv(io.StringIO(response.text))
            elif url.endswith(('.xlsx', '.xls')):
                return pd.read_excel(io.BytesIO(response.content))
            else:
                try:
                    return pd.read_csv(io.StringIO(response.text))
                except:
                    return pd.read_excel(io.BytesIO(response.content))
                    
        except Exception as e:
            raise Exception(f"Error loading data from URL: {e}")
    
    @staticmethod
    def load_from_uploaded_file(uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
        except Exception as e:
            raise Exception(f"Error loading uploaded file: {e}")
    
    @staticmethod
    def generate_sample_data(dataset_type="mixed"):
        """Generate sample data for different dataset types"""
        np.random.seed(42)
        n_samples = 200
        
        if dataset_type == "time_series":
            return DataLoader._generate_time_series_data(n_samples)
        elif dataset_type == "categorical":
            return DataLoader._generate_categorical_data(n_samples)
        elif dataset_type == "numerical":
            return DataLoader._generate_numerical_data(n_samples)
        else:  # mixed
            return DataLoader._generate_mixed_data(n_samples)
    
    @staticmethod
    def _generate_time_series_data(n_samples):
        """Generate time series sample data"""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        trend = np.linspace(100, 150, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 5, n_samples)
        
        data = {
            'timestamp': dates,
            'value': trend + seasonal + noise,
            'temperature': np.random.normal(25, 5, n_samples),
            'volume': np.random.exponential(1000, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def _generate_categorical_data(n_samples):
        """Generate categorical-rich sample data"""
        data = {
            'customer_id': range(1, n_samples + 1),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_samples),
            'satisfaction_level': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], n_samples),
            'purchase_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'], n_samples),
            'loyalty_status': np.random.choice(['New', 'Regular', 'VIP'], n_samples)
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def _generate_numerical_data(n_samples):
        """Generate numerical-rich sample data"""
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.uniform(-5, 5, n_samples),
            'feature_4': np.random.gamma(2, 2, n_samples),
            'feature_5': np.random.beta(2, 5, n_samples),
            'target_continuous': np.random.normal(50, 15, n_samples),
            'target_binary': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def _generate_mixed_data(n_samples):
        """Generate mixed-type sample data"""
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        data = {
            'date': dates,
            'sales': np.random.normal(1000, 200, n_samples),
            'temperature': np.random.normal(25, 5, n_samples),
            'marketing_spend': np.random.exponential(1000, n_samples),
            'website_visits': np.random.poisson(500, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples),
            'customer_rating': np.random.uniform(1, 5, n_samples),
            'advertising_budget': np.random.normal(5000, 1500, n_samples),
            'group': np.random.choice(['Control', 'Treatment'], n_samples),
            'satisfaction_score': np.random.normal(4.2, 0.8, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'customer_lifetime_value': np.random.exponential(5000, n_samples)
        }
        return pd.DataFrame(data)

class DataPreprocessor:
    """Handle data preprocessing and type conversions"""
    
    @staticmethod
    def preprocess_data(df, target_variable=None):
        """Preprocess data for analysis"""
        df_processed = df.copy()
        
        # Auto-detect and convert date columns
        for col in df_processed.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col])
                except:
                    pass
        
        # Convert string categoricals to numerical if needed for target
        if target_variable and df_processed[target_variable].dtype == 'object':
            le = LabelEncoder()
            df_processed[target_variable] = le.fit_transform(df_processed[target_variable])
            st.info(f"Converted '{target_variable}' from categorical to numerical for analysis")
        
        return df_processed
    
    @staticmethod
    def prepare_features(df, target_variable, problem_type):
        """Prepare features for modeling"""
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        return X, y

class AdvancedStatisticalAnalysis:
    def __init__(self, df):
        self.df = df
        self.original_df = df.copy()
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def dataset_characteristics_phase(self):
        """Comprehensive dataset type and characteristics analysis"""
        st.markdown('<div class="phase-header"><h2>🔍 Dataset Type & Characteristics Analysis</h2></div>', 
                   unsafe_allow_html=True)
        
        # Detect dataset type
        dataset_type, characteristics = DatasetTypeDetector.detect_dataset_type(self.df)
        detailed_analysis = DatasetTypeDetector.analyze_dataset_characteristics(self.df)
        
        # Display dataset type
        st.markdown(f"""
        <div class="dataset-type-box">
            <h3>📊 Dataset Type: {dataset_type}</h3>
            <p>Comprehensive analysis of your dataset structure and properties</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Characteristics overview
        st.subheader("📋 Dataset Characteristics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{detailed_analysis['basic_info']['rows']:,}")
        with col2:
            st.metric("Total Columns", detailed_analysis['basic_info']['columns'])
        with col3:
            st.metric("Memory Usage", f"{detailed_analysis['basic_info']['memory_usage_mb']:.2f} MB")
        with col4:
            st.metric("Duplicate Rows", detailed_analysis['basic_info']['duplicate_rows'])
        
        # Detailed characteristics in expandable sections
        self._display_detailed_characteristics(characteristics, detailed_analysis, dataset_type)
        
        # Data type distribution
        self._display_data_type_distribution()
        
        # Recommendations based on dataset type
        self._display_dataset_recommendations(dataset_type, characteristics, detailed_analysis)
    
    def _display_detailed_characteristics(self, characteristics, detailed_analysis, dataset_type):
        """Display detailed dataset characteristics"""
        
        # Data Composition
        with st.expander("🔍 Data Composition Analysis", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Numerical Columns", len(self.numerical_cols))
            with col2:
                st.metric("Categorical Columns", len(self.categorical_cols))
            with col3:
                st.metric("Date/Time Columns", len(self.date_cols))
            with col4:
                st.metric("Total Cells", f"{detailed_analysis['basic_info']['total_cells']:,}")
            
            # Data type distribution chart
            fig, ax = plt.subplots(figsize=(10, 6))
            types = ['Numerical', 'Categorical', 'Date/Time']
            counts = [len(self.numerical_cols), len(self.categorical_cols), len(self.date_cols)]
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            bars = ax.bar(types, counts, color=colors, alpha=0.8)
            ax.set_ylabel('Number of Columns')
            ax.set_title('Data Type Distribution')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Data Quality Assessment
        with st.expander("📈 Data Quality Metrics"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                completeness = detailed_analysis['data_quality']['completeness_score']
                st.metric("Completeness Score", f"{completeness:.1f}%")
            with col2:
                missing_pct = detailed_analysis['data_quality']['missing_values_percentage']
                st.metric("Missing Values", f"{missing_pct:.2f}%")
            with col3:
                dup_pct = detailed_analysis['data_quality']['duplicate_percentage']
                st.metric("Duplicate Rows", f"{dup_pct:.2f}%")
            with col4:
                cols_with_missing = detailed_analysis['data_quality']['columns_with_missing']
                st.metric("Columns with Missing", cols_with_missing)
            
            # Quality score gauge
            quality_score = self._calculate_overall_quality_score(detailed_analysis)
            st.progress(quality_score / 100)
            st.write(f"**Overall Data Quality Score: {quality_score:.1f}/100**")
            
            if quality_score >= 90:
                st.success("✅ Excellent data quality - ready for advanced analysis!")
            elif quality_score >= 75:
                st.warning("⚠️ Good data quality - minor improvements possible")
            elif quality_score >= 60:
                st.warning("⚠️ Fair data quality - consider data cleaning")
            else:
                st.error("❌ Poor data quality - significant cleaning needed")
        
        # Statistical Properties
        with st.expander("📊 Statistical Properties"):
            if len(self.numerical_cols) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_skew = detailed_analysis['statistical_properties'].get('skewness_high', 0)
                    st.metric("Highly Skewed Columns", high_skew)
                with col2:
                    high_var = detailed_analysis['statistical_properties'].get('high_variance_columns', 0)
                    st.metric("High Variance Columns", high_var)
                with col3:
                    st.metric("Correlation Strength", 
                             detailed_analysis['patterns_detected']['correlation_strength'])
                with col4:
                    st.metric("Outlier Pattern", 
                             detailed_analysis['patterns_detected']['potential_outliers'])
                
                # Display basic statistics for numerical columns
                st.write("**Numerical Columns Summary Statistics:**")
                st.dataframe(self.df[self.numerical_cols].describe().round(3), use_container_width=True)
            else:
                st.info("No numerical columns available for statistical analysis")
        
        # Pattern Detection
        with st.expander("🔄 Detected Patterns"):
            patterns = detailed_analysis['patterns_detected']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Patterns:**")
                st.markdown(f"""
                - **Temporal Patterns**: {patterns['temporal_patterns']}
                - **Categorical Dominance**: {patterns['categorical_dominance']}
                - **Correlation Strength**: {patterns['correlation_strength']}
                - **Outlier Presence**: {patterns['potential_outliers']}
                """)
            
            with col2:
                st.write("**Dataset Characteristics:**")
                characteristics_list = [
                    "✅ Time Series Data" if characteristics['is_time_series'] else "❌ Time Series Data",
                    "✅ Geospatial Data" if characteristics['has_geospatial'] else "❌ Geospatial Data",
                    "✅ Text Data" if characteristics['has_text'] else "❌ Text Data",
                    "✅ Missing Values Present" if characteristics['has_missing_values'] else "✅ No Missing Values"
                ]
                for char in characteristics_list:
                    st.write(char)
    
    def _display_data_type_distribution(self):
        """Display data type distribution visualization"""
        st.subheader("📈 Data Type Distribution")
        
        # Create a more detailed type breakdown
        type_breakdown = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            if 'int' in dtype or 'float' in dtype:
                type_breakdown['Numerical'] = type_breakdown.get('Numerical', 0) + 1
            elif 'object' in dtype:
                # Check if it's actually text or categorical
                if self.df[col].astype(str).str.len().mean() > 50:
                    type_breakdown['Text'] = type_breakdown.get('Text', 0) + 1
                else:
                    type_breakdown['Categorical'] = type_breakdown.get('Categorical', 0) + 1
            elif 'datetime' in dtype:
                type_breakdown['Date/Time'] = type_breakdown.get('Date/Time', 0) + 1
            else:
                type_breakdown['Other'] = type_breakdown.get('Other', 0) + 1
        
        # Create pie chart
        if type_breakdown:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            labels = list(type_breakdown.keys())
            sizes = list(type_breakdown.values())
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            
            ax1.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
            ax1.set_title('Data Type Distribution')
            
            # Bar chart
            ax2.bar(range(len(labels)), sizes, color=colors[:len(labels)], alpha=0.8)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45)
            ax2.set_ylabel('Number of Columns')
            ax2.set_title('Data Type Count')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    def _display_dataset_recommendations(self, dataset_type, characteristics, detailed_analysis):
        """Display recommendations based on dataset type and characteristics"""
        st.subheader("🎯 Recommended Analysis Approaches")
        
        recommendations = []
        
        # Based on dataset type
        if dataset_type == "Time Series":
            recommendations.extend([
                "✅ **Time Series Decomposition**: Analyze trend, seasonality, and residuals",
                "✅ **Stationarity Testing**: Check if data is stationary using ADF test",
                "✅ **Forecasting Models**: Use ARIMA, Prophet, or LSTM for predictions",
                "✅ **Seasonal Analysis**: Identify and model seasonal patterns",
                "✅ **Anomaly Detection**: Detect unusual patterns or outliers in time series"
            ])
        elif dataset_type == "Geospatial":
            recommendations.extend([
                "✅ **Spatial Analysis**: Analyze geographic patterns and clusters",
                "✅ **Heat Maps**: Visualize density and distribution across locations",
                "✅ **Spatial Autocorrelation**: Check for spatial dependencies",
                "✅ **Geographic Segmentation**: Group locations based on characteristics"
            ])
        elif dataset_type == "Text Data":
            recommendations.extend([
                "✅ **Text Preprocessing**: Clean and tokenize text data",
                "✅ **Sentiment Analysis**: Analyze emotional tone in text",
                "✅ **Topic Modeling**: Discover hidden themes using LDA or NMF",
                "✅ **Word Frequency Analysis**: Identify most common terms and phrases"
            ])
        else:  # Tabular/Mixed
            recommendations.extend([
                "✅ **Correlation Analysis**: Identify relationships between variables",
                "✅ **Cluster Analysis**: Group similar observations together",
                "✅ **Classification/Regression**: Build predictive models",
                "✅ **Feature Importance**: Identify most influential variables"
            ])
        
        # Based on data quality
        quality_score = self._calculate_overall_quality_score(detailed_analysis)
        if quality_score < 80:
            recommendations.extend([
                "⚠️ **Data Cleaning**: Address missing values and duplicates",
                "⚠️ **Outlier Treatment**: Handle extreme values appropriately",
                "⚠️ **Data Validation**: Verify data consistency and accuracy"
            ])
        
        # Based on data composition
        if len(self.numerical_cols) > 10:
            recommendations.append("✅ **Dimensionality Reduction**: Use PCA to reduce feature space")
        
        if len(self.categorical_cols) > 5:
            recommendations.append("✅ **Categorical Encoding**: Properly encode categorical variables for modeling")
        
        # Display recommendations
        col1, col2 = st.columns(2)
        mid_point = len(recommendations) // 2
        
        with col1:
            for rec in recommendations[:mid_point]:
                st.markdown(rec)
        
        with col2:
            for rec in recommendations[mid_point:]:
                st.markdown(rec)
        
        # Suggested next steps
        st.markdown("""
        <div class="insight-box">
        <h4>🚀 Suggested Next Steps:</h4>
        <ol>
            <li><strong>Data Validation</strong>: Verify data quality and address any issues</li>
            <li><strong>Exploratory Analysis</strong>: Deep dive into patterns and relationships</li>
            <li><strong>Feature Engineering</strong>: Create new features based on domain knowledge</li>
            <li><strong>Model Building</strong>: Develop predictive models based on your objectives</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    def _calculate_overall_quality_score(self, detailed_analysis):
        """Calculate overall data quality score"""
        quality_metrics = detailed_analysis['data_quality']
        
        # Calculate scores for different aspects
        completeness_score = quality_metrics['completeness_score']
        duplicate_score = 100 - quality_metrics['duplicate_percentage']
        missing_score = 100 - min(quality_metrics['missing_values_percentage'] * 2, 100)  # More weight to missing values
        
        # Overall score (weighted average)
        overall_score = (completeness_score * 0.4 + duplicate_score * 0.3 + missing_score * 0.3)
        
        return max(0, min(100, overall_score))

    # Phase 1: Data Validation and Quality Assessment
    def data_validation_phase(self):
        st.markdown('<div class="phase-header"><h2>📋 Phase 1: Data Validation & Quality Assessment</h2></div>', 
                   unsafe_allow_html=True)
        
        # Dataset characteristics first
        self.dataset_characteristics_phase()
        
        # Continue with existing data validation...
        st.subheader("1.1 Data Profiling")
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
        
        # Rest of the existing data validation code...
        # ... (include all the existing _missing_data_analysis, _outlier_detection, etc.)

    # Rest of the existing methods (exploratory_analysis_phase, model_evaluation_phase, etc.)
    # ... (include all the existing methods from the previous implementation)

def main():
    st.markdown('<div class="main-header">📊 Advanced Statistical Analysis Suite</div>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    This application provides comprehensive statistical analysis with advanced dataset type detection, 
    characteristics analysis, pattern detection, and confidence interval estimation for robust data insights.
    """)
    
    # Data source selection
    st.markdown('<div class="upload-section"><h3>📁 Data Source Selection</h3></div>', 
                unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose your data source:",
        ["Upload File", "URL", "Sample Data"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Upload File":
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
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    elif data_source == "URL":
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
                
            except Exception as e:
                st.error(f"❌ Error loading from URL: {e}")
    
    elif data_source == "Sample Data":
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
    
    # Proceed with analysis if data is loaded
    if df is not None:
        # Initialize analysis class
        analysis = AdvancedStatisticalAnalysis(df)
        
        # Phase selection
        st.sidebar.header("🔬 Analysis Configuration")
        phases = st.sidebar.multiselect(
            "Select analysis phases to run:",
            ["Dataset Characteristics", "Data Validation", "Exploratory Analysis", "Model Evaluation"],
            default=["Dataset Characteristics", "Data Validation", "Exploratory Analysis"]
        )
        
        # Data export options
        st.sidebar.header("📤 Export Options")
        if st.sidebar.button("Export Current Dataset"):
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="current_dataset.csv",
                mime="text/csv"
            )
        
        # Run selected phases
        if "Dataset Characteristics" in phases:
            analysis.dataset_characteristics_phase()
        
        if "Data Validation" in phases:
            analysis.data_validation_phase()
        
        if "Exploratory Analysis" in phases:
            analysis.exploratory_analysis_phase()
        
        if "Model Evaluation" in phases:
            analysis.model_evaluation_phase()
    
    else:
        st.info("👆 Please select a data source and load your data to begin analysis")
        
        # Show data format requirements
        with st.expander("📋 Data Format Requirements"):
            st.markdown("""
            **Supported File Formats:**
            - CSV (.csv)
            - Excel (.XLSX, .xls)
            
            **Dataset Types Supported:**
            - **Time Series Data**: Data with timestamps and sequential observations
            - **Tabular Data**: Standard structured data with rows and columns
            - **Categorical Data**: Data with multiple categorical variables
            - **Numerical Data**: Data dominated by numerical features
            - **Mixed Data**: Combination of numerical, categorical, and date/time data
            - **Geospatial Data**: Data with location information (latitude/longitude)
            - **Text Data**: Data containing substantial text content
            
            **Data Requirements:**
            - First row should contain column headers
            - Missing values should be empty or marked as NA
            - Date columns will be automatically detected
            - Large datasets are supported (up to 200MB)
            """)

if __name__ == "__main__":
    main()
