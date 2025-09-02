import numpy as np
import plotly.graph_objects as go

# Generate correlated 2D normal distribution
mean = [100, 160]  # Means for T1 and T2
cov = [[100, 80], [80, 100]]  # Covariance matrix

num_samples = 10000
samples = np.random.multivariate_normal(mean, cov, num_samples)
t1_samples = samples[:, 0]
t2_samples = samples[:, 1]

fig = go.Figure(
    data=go.Histogram2d(
        x=t1_samples,
        y=t2_samples,
        colorscale='Viridis',
        nbinsx=50,
        nbinsy=50,
        colorbar=dict(title='Counts')
    )
)

fig.update_layout(
    title="Correlated 2D Normal Distribution (Plotly)",
    xaxis_title="T1 (us)",
    yaxis_title="T2 (us)",
    width=700,
    height=600
)

fig.show()