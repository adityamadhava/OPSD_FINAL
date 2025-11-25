"""
Streamlit Dashboard for Live Ingestion + Online Adaptation
Shows real-time monitoring of forecasts, anomalies, and KPIs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Live Load Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
COUNTRIES = ['DE', 'FR', 'ES']
LIVE_COUNTRY = 'DE'  # Preselected live country


@st.cache_data
def load_forecasts(country_code: str) -> pd.DataFrame:
    """Load online forecasts."""
    file_path = OUTPUT_DIR / country_code / f"{country_code}_online_forecasts.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


@st.cache_data
def load_anomalies(country_code: str) -> pd.DataFrame:
    """Load anomaly detection results."""
    # Try online anomalies first, then dev/test anomalies
    file_path = OUTPUT_DIR / country_code / f"{country_code}_online_anomalies.csv"
    if not file_path.exists():
        # Fallback to dev anomalies
        file_path = OUTPUT_DIR / country_code / f"{country_code}_anomalies_dev.csv"
    
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


@st.cache_data
def load_updates(country_code: str) -> pd.DataFrame:
    """Load online update log."""
    file_path = OUTPUT_DIR / country_code / f"{country_code}_online_updates.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def calculate_rolling_mase(forecasts_df: pd.DataFrame, window_hours: int = 168) -> float:
    """Calculate rolling 7-day MASE."""
    if len(forecasts_df) < window_hours:
        return np.nan
    
    recent = forecasts_df.tail(window_hours)
    errors = np.abs(recent['y_true'] - recent['yhat'])
    
    # Naive seasonal forecast (24h lag)
    if len(recent) >= 24:
        naive_errors = np.abs(recent['y_true'].values - recent['y_true'].shift(24).values)
        naive_mae = naive_errors[24:].mean()
        if naive_mae > 0:
            return errors.mean() / naive_mae
    
    return np.nan


def calculate_pi_coverage(forecasts_df: pd.DataFrame, window_hours: int = 168) -> float:
    """Calculate rolling 7-day 80% PI coverage."""
    if len(forecasts_df) < window_hours:
        return np.nan
    
    recent = forecasts_df.tail(window_hours)
    if 'lo' not in recent.columns or 'hi' not in recent.columns:
        return np.nan
    
    in_interval = (recent['y_true'] >= recent['lo']) & (recent['y_true'] <= recent['hi'])
    return in_interval.mean() * 100


def count_anomalies_today(anomalies_df: pd.DataFrame) -> int:
    """Count anomaly hours today."""
    if len(anomalies_df) == 0:
        return 0
    
    today = pd.Timestamp.now().normalize()
    today_data = anomalies_df[anomalies_df['timestamp'].dt.date == today.date()]
    
    if 'flag_z' in today_data.columns:
        return (today_data['flag_z'] == 1).sum()
    return 0


def get_last_update_time(updates_df: pd.DataFrame) -> str:
    """Get last update timestamp."""
    if len(updates_df) == 0:
        return "Never"
    
    last_update = updates_df['timestamp'].max()
    return last_update.strftime("%Y-%m-%d %H:%M:%S UTC")


def plot_live_series(forecasts_df: pd.DataFrame, days: int = 14):
    """Plot last N days of y_true and yhat."""
    if len(forecasts_df) == 0:
        return None
    
    # Get last N days
    cutoff = forecasts_df['timestamp'].max() - timedelta(days=days)
    recent = forecasts_df[forecasts_df['timestamp'] >= cutoff].copy()
    
    if len(recent) == 0:
        return None
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=recent['timestamp'],
        y=recent['y_true'],
        name='Actual (y_true)',
        line=dict(color='#1f77b4', width=2),
        mode='lines'
    ))
    
    # Forecasts
    fig.add_trace(go.Scatter(
        x=recent['timestamp'],
        y=recent['yhat'],
        name='Forecast (yhat)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        title=f"Live Series - Last {days} Days",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        hovermode='x unified',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_forecast_cone(forecasts_df: pd.DataFrame):
    """Plot next 24h forecast with 80% PI."""
    if len(forecasts_df) == 0:
        return None
    
    # Get most recent forecast (last 24 hours)
    recent = forecasts_df.tail(24).copy()
    
    if len(recent) == 0 or 'lo' not in recent.columns or 'hi' not in recent.columns:
        return None
    
    fig = go.Figure()
    
    # Prediction interval (shaded area)
    fig.add_trace(go.Scatter(
        x=pd.concat([recent['timestamp'], recent['timestamp'][::-1]]),
        y=pd.concat([recent['hi'], recent['lo'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% PI',
        showlegend=True
    ))
    
    # Forecast mean
    fig.add_trace(go.Scatter(
        x=recent['timestamp'],
        y=recent['yhat'],
        name='Forecast Mean',
        line=dict(color='#ff7f0e', width=3),
        mode='lines+markers'
    ))
    
    # Actual (if available)
    if 'y_true' in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent['timestamp'],
            y=recent['y_true'],
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Forecast Cone - Next 24 Hours",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        hovermode='x unified',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_anomaly_tape(forecasts_df: pd.DataFrame, anomalies_df: pd.DataFrame, days: int = 14):
    """Plot anomaly tape highlighting flagged hours."""
    if len(forecasts_df) == 0:
        return None
    
    # Get last N days
    cutoff = forecasts_df['timestamp'].max() - timedelta(days=days)
    recent_forecasts = forecasts_df[forecasts_df['timestamp'] >= cutoff].copy()
    
    if len(recent_forecasts) == 0:
        return None
    
    # Merge with anomalies
    if len(anomalies_df) > 0:
        recent_forecasts = recent_forecasts.merge(
            anomalies_df[['timestamp', 'flag_z', 'flag_cusum']],
            on='timestamp',
            how='left'
        )
        recent_forecasts['flag_z'] = recent_forecasts['flag_z'].fillna(0)
        if 'flag_cusum' in recent_forecasts.columns:
            recent_forecasts['flag_cusum'] = recent_forecasts['flag_cusum'].fillna(0)
    else:
        recent_forecasts['flag_z'] = 0
        recent_forecasts['flag_cusum'] = 0
    
    fig = go.Figure()
    
    # Base line (load)
    fig.add_trace(go.Scatter(
        x=recent_forecasts['timestamp'],
        y=recent_forecasts['y_true'],
        name='Load',
        line=dict(color='#1f77b4', width=2),
        mode='lines'
    ))
    
    # Highlight anomalies (z-score)
    z_anomalies = recent_forecasts[recent_forecasts['flag_z'] == 1]
    if len(z_anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=z_anomalies['timestamp'],
            y=z_anomalies['y_true'],
            name='Z-Score Anomaly',
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-up',
                line=dict(width=2, color='darkred')
            )
        ))
    
    # Highlight CUSUM anomalies (if available)
    if 'flag_cusum' in recent_forecasts.columns:
        cusum_anomalies = recent_forecasts[recent_forecasts['flag_cusum'] == 1]
        if len(cusum_anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=cusum_anomalies['timestamp'],
                y=cusum_anomalies['y_true'],
                name='CUSUM Anomaly',
                mode='markers',
                marker=dict(
                    color='orange',
                    size=10,
                    symbol='diamond',
                    line=dict(width=2, color='darkorange')
                )
            ))
    
    fig.update_layout(
        title=f"Anomaly Tape - Last {days} Days",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        hovermode='x unified',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def main():
    """Main dashboard function."""
    st.title("📊 Live Load Forecasting Dashboard")
    st.markdown("---")
    
    # Sidebar - Country selector
    st.sidebar.header("Configuration")
    selected_country = st.sidebar.selectbox(
        "Select Country",
        COUNTRIES,
        index=COUNTRIES.index(LIVE_COUNTRY) if LIVE_COUNTRY in COUNTRIES else 0
    )
    
    # Load data
    forecasts_df = load_forecasts(selected_country)
    anomalies_df = load_anomalies(selected_country)
    updates_df = load_updates(selected_country)
    
    if len(forecasts_df) == 0:
        st.error(f"No forecast data found for {selected_country}. Please run the online adaptation simulation first.")
        st.info("Run: `python src/online_adaptation.py`")
        return
    
    # KPI Tiles
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rolling_mase = calculate_rolling_mase(forecasts_df, window_hours=168)
        st.metric(
            "Rolling 7d MASE",
            f"{rolling_mase:.3f}" if not np.isnan(rolling_mase) else "N/A"
        )
    
    with col2:
        pi_coverage = calculate_pi_coverage(forecasts_df, window_hours=168)
        st.metric(
            "80% PI Coverage (7d)",
            f"{pi_coverage:.1f}%" if not np.isnan(pi_coverage) else "N/A"
        )
    
    with col3:
        anomaly_count = count_anomalies_today(anomalies_df)
        st.metric(
            "Anomaly Hours (Today)",
            f"{anomaly_count}"
        )
    
    with col4:
        last_update = get_last_update_time(updates_df)
        st.metric(
            "Last Update Time",
            last_update
        )
    
    st.markdown("---")
    
    # Main charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Live series
        fig_live = plot_live_series(forecasts_df, days=14)
        if fig_live:
            st.plotly_chart(fig_live, use_container_width=True)
        
        # Forecast cone
        fig_cone = plot_forecast_cone(forecasts_df)
        if fig_cone:
            st.plotly_chart(fig_cone, use_container_width=True)
    
    with col_right:
        # Anomaly tape
        fig_anomaly = plot_anomaly_tape(forecasts_df, anomalies_df, days=14)
        if fig_anomaly:
            st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Update log summary
        st.subheader("🔄 Update Log Summary")
        if len(updates_df) > 0:
            summary = updates_df.groupby('reason').agg({
                'duration_s': ['count', 'mean']
            }).round(2)
            summary.columns = ['Count', 'Avg Duration (s)']
            st.dataframe(summary, use_container_width=True)
            
            # Recent updates
            st.subheader("Recent Updates")
            recent_updates = updates_df.tail(10)[['timestamp', 'reason', 'duration_s']]
            st.dataframe(recent_updates, use_container_width=True, hide_index=True)
        else:
            st.info("No update log available.")
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Data source: {OUTPUT_DIR / selected_country}")


if __name__ == "__main__":
    main()

