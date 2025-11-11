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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
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

class Config:
    """Configuration constants"""
    SAMPLE_SIZE = 200
    DEFAULT_TEST_SIZE = 0.2
    MAX_ROWS_FOR_ANALYSIS = 10000
    PLOT_HEIGHT = 600
    PLOT_WIDTH = 800

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
            "mixed": DataLoader._generate_mixed_data
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
            page_title="Statistical Analysis Suite",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main application runner"""
        st.title("📊 Statistical Analysis Suite")
        st.write("Comprehensive data analysis platform with automated insights and machine learning.")
        
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
            ["Mixed", "Time Series", "Categorical", "Numerical"]
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
        st.sidebar.header("Analysis Phases")
        phases = st.sidebar.multiselect(
            "Select phases to run:",
            ["Dataset Characteristics", "Data Validation", "Exploratory Analysis", "Model Evaluation"],
            default=["Dataset Characteristics", "Data Validation", "Exploratory Analysis"]
        )
        
        # Initialize and run phases
        analysis_phases = {
            "Dataset Characteristics": DatasetCharacteristicsPhase(df),
            "Data Validation": DataValidationPhase(df),
            "Exploratory Analysis": ExploratoryAnalysisPhase(df),
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
        
        with st.expander("📋 Instructions"):
            st.markdown("""
            **How to use:**
            1. **Load Data**: Upload file, enter URL, or use sample data
            2. **Select Analysis**: Choose phases from sidebar
            3. **Explore Results**: Interactive visualizations and insights
            
            **Supported Analyses:**
            - Dataset profiling and type detection
            - Data quality assessment
            - Statistical analysis and visualization
            - Machine learning model evaluation
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
