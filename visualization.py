import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap using Plotly"""
        corr = self.data.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale='RdBu_r',
            aspect='equal'  # Changed to equal for better proportions
        )
        fig.update_layout(
            title={
                'text': "Feature Correlation Heatmap",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=700,  # Fixed width
            height=600,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_feature_distributions(self):
        """Plot distribution of numerical features"""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_cols:
            if col != 'target':
                fig = px.histogram(
                    self.data,
                    x=col,
                    color='target',
                    marginal='box',
                    title=f'Distribution of {col}',
                    template='plotly_white'  # Clean template
                )
                fig.update_layout(
                    width=800,
                    height=500,
                    bargap=0.1,
                    title={
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    margin=dict(l=40, r=40, t=60, b=40),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    def plot_target_distribution(self):
        """Plot target variable distribution"""
        fig = px.pie(
            self.data,
            names='target',
            title='Target Distribution',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            hole=0.3  # Added donut style
        )
        fig.update_layout(
            width=600,
            height=400,
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from the trained model"""
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            feat_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            template='plotly_white'
        )
        fig.update_layout(
            width=800,
            height=500,
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False,
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)