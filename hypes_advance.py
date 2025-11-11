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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import requests
import io
from urllib.parse import urlparse
import openpyxl
import joblib
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
</style>
""", unsafe_allow_html=True)

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
    def generate_sample_data():
        """Generate sample data for demonstration"""
        np.random.seed(42)
        n_samples = 200
        
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        trend = np.linspace(100, 150, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 5, n_samples)
        
        sample_data = {
            'date': dates,
            'sales': trend + seasonal + noise,
            'temperature': np.random.normal(25, 5, n_samples),
            'marketing_spend': np.random.exponential(1000, n_samples),
            'website_visits': np.random.poisson(500, n_samples) + trend.astype(int),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_samples),
            'customer_rating': np.random.uniform(1, 5, n_samples),
            'advertising_budget': np.random.normal(5000, 1500, n_samples),
            'group': np.random.choice(['Control', 'Treatment'], n_samples),
            'satisfaction_score': np.random.normal(4.2, 0.8, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'customer_lifetime_value': np.random.exponential(5000, n_samples)
        }
        
        return pd.DataFrame(sample_data)

class DataPreprocessor:
    """Handle data preprocessing and type conversions"""
    
    @staticmethod
    def preprocess_data(df, target_variable=None):
        """Preprocess data for analysis"""
        df_processed = df.copy()
        
        # Auto-detect and convert date columns
        for col in df_processed.columns:
            if 'date' in col.lower() or 'time' in col.lower():
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
        
    # Phase 1: Data Validation and Quality Assessment
    def data_validation_phase(self):
        st.markdown('<div class="phase-header"><h2>📋 Phase 1: Data Validation & Quality Assessment</h2></div>', 
                   unsafe_allow_html=True)
        
        # Data Profiling
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
        
        # Data Preview
        with st.expander("🔍 Data Preview", expanded=True):
            tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Data Types"])
            with tab1:
                st.dataframe(self.df.head(10), use_container_width=True)
            with tab2:
                st.dataframe(self.df.tail(10), use_container_width=True)
            with tab3:
                dtype_info = pd.DataFrame({
                    'Column': self.df.columns,
                    'Data Type': self.df.dtypes,
                    'Non-Null Count': self.df.count(),
                    'Null Count': self.df.isnull().sum()
                })
                st.dataframe(dtype_info, use_container_width=True)
        
        # Missing Data Analysis
        st.subheader("1.2 Missing Data Analysis")
        self._missing_data_analysis()
        
        # Outlier Detection
        st.subheader("1.3 Outlier Detection")
        self._outlier_detection()
        
        # Data Quality Summary
        st.subheader("1.4 Data Quality Summary")
        self._data_quality_summary()
    
    def _missing_data_analysis(self):
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
                ax.set_xlabel('Missing Percentage (%)')
                st.pyplot(fig)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        # Missing data heatmap
        if missing_data.sum() > 0:
            st.write("**Missing Data Pattern Heatmap:**")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
            ax.set_title('Missing Data Heatmap (Yellow = Missing)')
            st.pyplot(fig)
    
    def _outlier_detection(self):
        if not self.numerical_cols:
            st.info("No numerical columns for outlier detection")
            return
        
        st.write("**Outlier Detection Summary:**")
        
        outlier_results = {}
        for col in self.numerical_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            z_outliers = (z_scores > 3).sum()
            
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
        
        # Visualize outliers for selected variable
        selected_var = st.selectbox("Select variable for detailed outlier visualization:", 
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
    
    def _data_quality_summary(self):
        """Generate comprehensive data quality summary"""
        quality_metrics = {
            'Completeness': ((len(self.df) - self.df.isnull().sum()) / len(self.df) * 100).mean(),
            'Uniqueness': (1 - self.df.duplicated().sum() / len(self.df)) * 100,
            'Consistency': self._calculate_consistency_score(),
            'Validity': self._calculate_validity_score()
        }
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_data = [
            ("Completeness", quality_metrics['Completeness'], "#3498db"),
            ("Uniqueness", quality_metrics['Uniqueness'], "#2ecc71"),
            ("Consistency", quality_metrics['Consistency'], "#f39c12"),
            ("Validity", quality_metrics['Validity'], "#e74c3c")
        ]
        
        for (title, value, color), col in zip(metrics_data, [col1, col2, col3, col4]):
            with col:
                st.metric(title, f"{value:.1f}%")
        
        overall_quality = np.mean(list(quality_metrics.values()))
        st.write(f"**Overall Data Quality Score: {overall_quality:.1f}%**")
        
        if overall_quality >= 90:
            st.success("✅ Excellent data quality - ready for analysis!")
        elif overall_quality >= 75:
            st.warning("⚠️ Good data quality - some improvements possible")
        else:
            st.error("❌ Poor data quality - significant cleaning needed")
    
    def _calculate_consistency_score(self):
        score = 100
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                if unique_count > len(self.df) * 0.5:
                    score -= 5
        return max(score, 0)
    
    def _calculate_validity_score(self):
        score = 100
        for col in self.numerical_cols:
            if (self.df[col] < 0).any() and col.lower() in ['age', 'price', 'salary']:
                score -= 10
        return max(score, 0)

    # Phase 2: Exploratory Data Analysis
    def exploratory_analysis_phase(self):
        st.markdown('<div class="phase-header"><h2>🔍 Phase 2: Exploratory Data Analysis</h2></div>', 
                   unsafe_allow_html=True)
        
        # Pattern and Trend Analysis
        st.subheader("2.1 Advanced Pattern & Trend Analysis")
        self._pattern_trend_analysis()
        
        # Time Series Analysis
        if self._has_date_column():
            st.subheader("2.2 Time Series Analysis")
            self._time_series_analysis()
        
        # Correlation Analysis
        st.subheader("2.3 Correlation Analysis with Confidence Intervals")
        self._correlation_with_ci()
        
        # Distribution Analysis
        st.subheader("2.4 Distribution Analysis")
        self._distribution_analysis()
        
        # Multivariate Analysis
        st.subheader("2.5 Multivariate Analysis")
        self._multivariate_analysis()
        
        # Advanced Statistical Tests
        if PINGOUIN_AVAILABLE:
            st.subheader("2.6 Advanced Statistical Tests")
            self._advanced_statistical_tests()

    def _pattern_trend_analysis(self):
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
            self._rolling_analysis_with_ci(trend_var, confidence_level)
        
        self._linear_trend_analysis(trend_var, confidence_level)
        self._seasonal_analysis(trend_var)
        self._change_point_analysis(trend_var)
        
        if self._has_date_column() and PROPHET_AVAILABLE:
            self._prophet_forecasting(trend_var)

    def _rolling_analysis_with_ci(self, variable, confidence_level):
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
        ax1.set_title(f'Rolling Mean with {confidence_level:.0%} Confidence Interval - {variable}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.df[x_var], rolling_std, color='green', linewidth=2)
        ax2.set_title(f'Rolling Standard Deviation - {variable}')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        ci_width_mean = ((ci_upper - ci_lower).mean() / rolling_mean.mean() * 100) if rolling_mean.mean() != 0 else 0
        volatility_status = 'Stable' if rolling_std.mean() < rolling_mean.mean() * 0.1 else 'Volatile'
        trend_direction = self._get_trend_direction(rolling_mean)
        volatility_pattern = self._get_volatility_pattern(rolling_std)
        
        st.markdown(f"""
        <div class="trend-box">
        <h4>📈 Rolling Analysis Interpretation - {variable}:</h4>
        <ul>
            <li><strong>Trend Direction:</strong> {trend_direction}</li>
            <li><strong>Volatility Pattern:</strong> {volatility_pattern}</li>
            <li><strong>Confidence Interval Width:</strong> {ci_width_mean:.1f}% of mean value</li>
            <li><strong>Data Stability:</strong> {volatility_status}</li>
            <li><strong>Window Size:</strong> {window_size} observations used for rolling calculation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    def _linear_trend_analysis(self, variable, confidence_level):
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
        ax.set_title(f'Linear Trend Analysis with {confidence_level:.0%} Confidence Band - {variable}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
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

    def _seasonal_analysis(self, variable):
        if len(self.df) < 50:
            st.info("Insufficient data points for seasonal analysis (need ≥50 observations)")
            return
        
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
            
            seasonal_strength = max(0, 1 - (decomposition.resid.var() / decomposition.observed.var()))
            seasonal_strength_desc = 'Strong' if seasonal_strength > 0.6 else 'Moderate' if seasonal_strength > 0.3 else 'Weak'
            trend_clarity = 'Clear' if decomposition.trend.std() > decomposition.observed.std() * 0.1 else 'Subtle'
            residual_pattern = 'Random' if abs(decomposition.resid.skew()) < 1 else 'Systematic'
            
            st.markdown(f"""
            <div class="trend-box">
            <h4>🔄 Seasonal Analysis Interpretation - {variable}:</h4>
            <ul>
                <li><strong>Seasonal Strength:</strong> {seasonal_strength:.3f} ({seasonal_strength_desc})</li>
                <li><strong>Trend Clarity:</strong> {trend_clarity}</li>
                <li><strong>Residual Pattern:</strong> {residual_pattern}</li>
                <li><strong>Seasonal Period:</strong> {seasonal_period} time points</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Seasonal decomposition failed: {e}")

    def _change_point_analysis(self, variable):
        data = self.df[variable].dropna().values
        if len(data) < 20:
            return
        
        window = len(data) // 10
        rolling_mean = pd.Series(data).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(data).rolling(window=window, center=True).std()
        
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        change_points = np.where(z_scores > 2)[0]
        
        if len(change_points) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data, alpha=0.7, label='Data')
            ax.scatter(change_points, data[change_points], color='red', s=50, 
                     label='Potential Change Points', zorder=5)
            ax.set_title(f'Change Point Detection - {variable}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.info(f"**Detected {len(change_points)} potential change points in {variable}**")

    def _prophet_forecasting(self, variable):
        if not PROPHET_AVAILABLE:
            return
            
        st.subheader("Prophet Time Series Forecasting")
        
        if st.button("Run Prophet Forecast"):
            try:
                date_col = self.date_cols[0]
                prophet_df = self.df[[date_col, variable]].dropna().copy()
                prophet_df.columns = ['ds', 'y']
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                fig1 = model.plot(forecast)
                plt.title(f'Prophet Forecast for {variable}')
                st.pyplot(fig1)
                
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
                st.write("**Forecast Summary (Next 30 periods):**")
                last_forecast = forecast.tail(30)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Forecast", f"{last_forecast['yhat'].mean():.2f}")
                with col2:
                    st.metric("Forecast Range", 
                             f"{last_forecast['yhat_lower'].min():.2f} - {last_forecast['yhat_upper'].max():.2f}")
                with col3:
                    trend_change = last_forecast['trend'].iloc[-1] - last_forecast['trend'].iloc[0]
                    st.metric("Trend Change", f"{trend_change:.2f}")
                    
            except Exception as e:
                st.error(f"Prophet forecasting failed: {e}")

    def _correlation_with_ci(self):
        if len(self.numerical_cols) < 2:
            st.info("Need at least 2 numerical variables for correlation analysis")
            return
        
        # Create correlation heatmap
        st.write("**Correlation Heatmap**")
        corr_matrix = self.df[self.numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Interactive Plotly heatmap with confidence intervals
        st.write("**Interactive Correlation Matrix with Confidence Intervals**")
        
        corr_matrix_plotly = np.zeros((len(self.numerical_cols), len(self.numerical_cols)))
        p_value_matrix = np.zeros_like(corr_matrix_plotly)
        ci_lower_matrix = np.zeros_like(corr_matrix_plotly)
        ci_upper_matrix = np.zeros_like(corr_matrix_plotly)
        
        confidence_level = 0.95
        
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
                
                z = np.arctanh(corr)
                se = 1 / np.sqrt(len(valid_data) - 3)
                z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                ci_lower = np.tanh(z - z_crit * se)
                ci_upper = np.tanh(z + z_crit * se)
                
                ci_lower_matrix[i, j] = ci_lower
                ci_upper_matrix[i, j] = ci_upper
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix_plotly,
            x=self.numerical_cols,
            y=self.numerical_cols,
            colorscale='RdBu_r',
            zmid=0,
            hoverongaps=False,
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<br>CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>',
            customdata=np.dstack([ci_lower_matrix, ci_upper_matrix])
        ))
        
        fig.update_layout(
            title='Correlation Matrix with 95% Confidence Intervals',
            width=800,
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Significant correlations table
        significant_corrs = []
        for i, col1 in enumerate(self.numerical_cols):
            for j, col2 in enumerate(self.numerical_cols):
                if i < j and p_value_matrix[i, j] < 0.05:
                    significant_corrs.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': corr_matrix_plotly[i, j],
                        'P-value': p_value_matrix[i, j],
                        'CI Lower': ci_lower_matrix[i, j],
                        'CI Upper': ci_upper_matrix[i, j]
                    })
        
        if significant_corrs:
            sig_corr_df = pd.DataFrame(significant_corrs)
            st.write("**Statistically Significant Correlations (p < 0.05):**")
            st.dataframe(sig_corr_df.sort_values('Correlation', key=abs, ascending=False))

    def _distribution_analysis(self):
        if not self.numerical_cols:
            return
        
        selected_var = st.selectbox("Select variable for distribution analysis:", self.numerical_cols)
        confidence_level = st.slider("Confidence Level for Mean", 0.80, 0.99, 0.95, 
                                   key="dist_ci")
        
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
        ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, 
                   label=f'CI Lower: {ci_lower:.2f}')
        ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=2, 
                   label=f'CI Upper: {ci_upper:.2f}')
        ax1.fill_betweenx([0, ax1.get_ylim()[1]], ci_lower, ci_upper, alpha=0.2, color='orange')
        ax1.set_title(f'Distribution of {selected_var} with {confidence_level:.0%} CI')
        ax1.legend()
        
        sns.boxplot(y=data, ax=ax2)
        ax2.set_title(f'Box Plot of {selected_var}')
        
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
        if not PINGOUIN_AVAILABLE:
            return
            
        st.subheader("Advanced Statistical Tests (Pingouin)")
        
        test_type = st.selectbox(
            "Select Statistical Test:",
            ["ANOVA", "Repeated Measures ANOVA", "Bayesian T-test", 
             "Mann-Whitney U", "Partial Correlation"]
        )
        
        if test_type == "ANOVA":
            self._pingouin_anova()
        elif test_type == "Repeated Measures ANOVA":
            self._pingouin_rm_anova()
        elif test_type == "Bayesian T-test":
            self._pingouin_bayesian_ttest()
        elif test_type == "Mann-Whitney U":
            self._pingouin_mannwhitney()
        elif test_type == "Partial Correlation":
            self._pingouin_partial_corr()

    def _pingouin_anova(self):
        st.write("**One-Way ANOVA**")
        
        dv = st.selectbox("Dependent Variable:", self.numerical_cols, key="anova_dv")
        between = st.selectbox("Between-subjects factor:", self.categorical_cols, key="anova_between")
        
        if dv and between:
            try:
                anova_result = pg.anova(data=self.df, dv=dv, between=between, detailed=True)
                st.write("**ANOVA Results:**")
                st.dataframe(anova_result.round(4))
                
                p_value = anova_result['p-unc'].iloc[0]
                if p_value < 0.05:
                    st.markdown(f'<div class="significant">✅ Significant effect found (p = {p_value:.4f})</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="not-significant">❌ No significant effect (p = {p_value:.4f})</div>', 
                               unsafe_allow_html=True)
                    
                if p_value < 0.05 and len(self.df[between].unique()) > 2:
                    st.write("**Post-hoc Tests (Tukey HSD):**")
                    posthoc = pg.pairwise_tukey(data=self.df, dv=dv, between=between)
                    st.dataframe(posthoc.round(4))
                    
            except Exception as e:
                st.error(f"ANOVA failed: {e}")

    def _pingouin_rm_anova(self):
        st.write("**Repeated Measures ANOVA**")
        st.info("For repeated measures, ensure you have within-subjects factor")
        
        dv = st.selectbox("Dependent Variable:", self.numerical_cols, key="rm_anova_dv")
        within = st.selectbox("Within-subjects factor:", self.categorical_cols, key="rm_anova_within")
        subject = st.selectbox("Subject identifier:", self.categorical_cols, key="rm_anova_subject")
        
        if dv and within and subject:
            try:
                rm_anova = pg.rm_anova(data=self.df, dv=dv, within=within, subject=subject, detailed=True)
                st.write("**Repeated Measures ANOVA Results:**")
                st.dataframe(rm_anova.round(4))
            except Exception as e:
                st.error(f"Repeated Measures ANOVA failed: {e}")

    def _pingouin_bayesian_ttest(self):
        st.write("**Bayesian T-test**")
        
        variable = st.selectbox("Variable:", self.numerical_cols, key="bayes_var")
        group_var = st.selectbox("Grouping variable:", self.categorical_cols, key="bayes_group")
        
        if variable and group_var:
            groups = self.df[group_var].unique()
            if len(groups) == 2:
                group1 = self.df[self.df[group_var] == groups[0]][variable]
                group2 = self.df[self.df[group_var] == groups[1]][variable]
                
                try:
                    bayes_ttest = pg.ttest(group1, group2, paired=False, alternative='two-sided')
                    st.write("**Bayesian T-test Results:**")
                    st.dataframe(bayes_ttest.round(4))
                    
                    bf10 = bayes_ttest['BF10'].iloc[0]
                    if bf10 > 3:
                        st.markdown(f'<div class="significant">✅ Substantial evidence for H1 (BF10 = {bf10:.2f})</div>', 
                                   unsafe_allow_html=True)
                    elif bf10 < 1/3:
                        st.markdown(f'<div class="not-significant">✅ Substantial evidence for H0 (BF10 = {bf10:.2f})</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="not-significant">⚖️ Inconclusive evidence (BF10 = {bf10:.2f})</div>', 
                                   unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Bayesian t-test failed: {e}")

    def _pingouin_mannwhitney(self):
        st.write("**Mann-Whitney U Test (Non-parametric)**")
        
        variable = st.selectbox("Variable:", self.numerical_cols, key="mw_var")
        group_var = st.selectbox("Grouping variable:", self.categorical_cols, key="mw_group")
        
        if variable and group_var:
            groups = self.df[group_var].unique()
            if len(groups) == 2:
                group1 = self.df[self.df[group_var] == groups[0]][variable]
                group2 = self.df[self.df[group_var] == groups[1]][variable]
                
                try:
                    mw_test = pg.mwu(group1, group2, alternative='two-sided')
                    st.write("**Mann-Whitney U Test Results:**")
                    st.dataframe(mw_test.round(4))
                except Exception as e:
                    st.error(f"Mann-Whitney test failed: {e}")

    def _pingouin_partial_corr(self):
        st.write("**Partial Correlation**")
        
        var1 = st.selectbox("Variable 1:", self.numerical_cols, key="pcorr_var1")
        var2 = st.selectbox("Variable 2:", self.numerical_cols, key="pcorr_var2")
        covar = st.multiselect("Covariates to control for:", 
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
        if not self._has_date_column():
            st.info("No date column found for time series analysis")
            return
            
        date_col = self.date_cols[0]
        numerical_var = st.selectbox("Select numerical variable for time series:", self.numerical_cols)
        
        self.df = self.df.sort_values(date_col)
        
        st.subheader("Stationarity Analysis")
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

    def _has_date_column(self):
        return len(self.date_cols) > 0
    
    def _get_trend_direction(self, series):
        if len(series.dropna()) < 2:
            return "Insufficient data"
        
        first_val = series.dropna().iloc[0]
        last_val = series.dropna().iloc[-1]
        
        if last_val > first_val * 1.1:
            return "Strongly Increasing"
        elif last_val > first_val * 1.02:
            return "Moderately Increasing"
        elif last_val < first_val * 0.9:
            return "Strongly Decreasing"
        elif last_val < first_val * 0.98:
            return "Moderately Decreasing"
        else:
            return "Relatively Stable"
    
    def _get_volatility_pattern(self, std_series):
        if len(std_series.dropna()) < 2:
            return "Insufficient data"
        
        std_values = std_series.dropna()
        if std_values.iloc[-1] > std_values.iloc[0] * 1.1:
            return "Increasing"
        elif std_values.iloc[-1] < std_values.iloc[0] * 0.9:
            return "Decreasing"
        else:
            return "Stable"

    # Phase 3: Model Evaluation & Validation (COMPLETE IMPLEMENTATION)
    def model_evaluation_phase(self):
        st.markdown('<div class="phase-header"><h2>🤖 Phase 3: Model Evaluation & Validation</h2></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("Model Configuration")
        
        # Problem type selection
        problem_type = st.radio(
            "Select Problem Type:", 
            ["Classification", "Regression"],
            horizontal=True
        )
        
        # Target variable selection with dynamic options
        if problem_type == "Classification":
            target_options = [col for col in self.df.columns if self.df[col].nunique() <= 10 or self.df[col].dtype == 'object']
        else:
            target_options = self.numerical_cols
        
        if not target_options:
            st.error("No suitable target variables found for the selected problem type.")
            return
            
        target_var = st.selectbox("Select Target Variable:", target_options)
        
        if not target_var:
            st.warning("Please select a target variable.")
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
        
        # Cross-validation settings
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        if st.button("Train and Evaluate Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Prepare features
                    X, y = preprocessor.prepare_features(df_processed, target_var, problem_type)
                    
                    if X.empty:
                        st.error("No valid features available for modeling.")
                        return
                    
                    # Train-test split
                    test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, 
                        stratify=y if problem_type == "Classification" else None
                    )
                    
                    # Train and evaluate model
                    if problem_type == "Classification":
                        results = self._evaluate_classification_model(
                            model_choice, X_train, X_test, y_train, y_test, cv_folds
                        )
                    else:
                        results = self._evaluate_regression_model(
                            model_choice, X_train, X_test, y_train, y_test, cv_folds
                        )
                    
                    # Display results
                    self._display_model_results(results, problem_type, X_test, y_test)
                    
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")

    def _evaluate_classification_model(self, model_choice, X_train, X_test, y_train, y_test, cv_folds):
        """Evaluate classification models"""
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == "Support Vector Machine":
            model = SVC(random_state=42, probability=True)
        else:
            model = RandomForestClassifier(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'model': model,
            'model_name': model_choice,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': X_train.columns.tolist()
        }

    def _evaluate_regression_model(self, model_choice, X_train, X_test, y_train, y_test, cv_folds):
        """Evaluate regression models"""
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Support Vector Regression":
            model = SVR()
        else:
            model = RandomForestRegressor(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return {
            'model': model,
            'model_name': model_choice,
            'y_test': y_test,
            'y_pred': y_pred,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': X_train.columns.tolist()
        }

    def _display_model_results(self, results, problem_type, X_test, y_test):
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
            self._plot_classification_results(results, X_test, y_test)
        else:
            self._plot_regression_results(results)
        
        # Feature importance
        self._plot_feature_importance(results)
        
        # Model interpretation
        st.markdown(f"""
        <div class="model-box">
        <h4>🎯 Model Interpretation - {results['model_name']}:</h4>
        <ul>
            <li><strong>Model Performance:</strong> {'Excellent' if results['cv_mean'] > 0.8 else 'Good' if results['cv_mean'] > 0.6 else 'Fair' if results['cv_mean'] > 0.4 else 'Poor'}</li>
            <li><strong>Generalization:</strong> {'Good' if results['cv_std'] < 0.1 else 'Moderate' if results['cv_std'] < 0.15 else 'Poor'} (CV std: {results['cv_std']:.3f})</li>
            <li><strong>Recommendation:</strong> {'Ready for deployment' if results['cv_mean'] > 0.7 and results['cv_std'] < 0.1 else 'Needs improvement'}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    def _plot_classification_results(self, results, X_test, y_test):
        """Plot classification results"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            # ROC Curve (for binary classification)
            if results['y_pred_proba'] is not None and len(np.unique(results['y_test'])) == 2:
                fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'][:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)

    def _plot_regression_results(self, results):
        """Plot regression results"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['y_test'], results['y_pred'], alpha=0.6)
            ax.plot([results['y_test'].min(), results['y_test'].max()], 
                   [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
        
        with col2:
            # Residuals plot
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
                # Get feature importances
                importances = results['model'].feature_importances_
                feature_names = results['feature_names']
                
                # Create feature importance DataFrame
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(fi_df['feature'], fi_df['importance'])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance Plot')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display table
                st.write("**Feature Importance Scores:**")
                st.dataframe(fi_df.sort_values('importance', ascending=False).head(10))
                
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.warning(f"Could not compute feature importance: {str(e)}")

def main():
    st.markdown('<div class="main-header">📊 Advanced Statistical Analysis Suite</div>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    This application provides comprehensive statistical analysis with advanced pattern detection, 
    trend analysis, and confidence interval estimation for robust data insights.
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
        
        if st.button("Generate Sample Dataset"):
            with st.spinner("Generating sample data..."):
                df = DataLoader.generate_sample_data()
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
            ["Data Validation", "Exploratory Analysis", "Model Evaluation"],
            default=["Data Validation", "Exploratory Analysis", "Model Evaluation"]
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
            - Excel (.xlsx, .xls)
            
            **Data Requirements:**
            - First row should contain column headers
            - Missing values should be empty or marked as NA
            - Date columns will be automatically detected
            - Large datasets are supported (up to 200MB)
            
            **Recommended Structure:**
            - Numerical data for statistical analysis
            - Categorical data for grouping and segmentation
            - Time series data for trend analysis
            """)

if __name__ == "__main__":
    main()
