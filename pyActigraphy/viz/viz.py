import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def double_plot(raw, freq='15min', binarize=False, span='48h'):
    """Double plot

    Parameters
    ----------
    raw: Instance of BaseRaw (or derived class)
        Raw object containing the actigraphy data
    freq: str, optional
        Data resampling frequency.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is '15min'.
    binarize: bool, optional
        If set to True, the data are binarized.
        Default is False.
    span: str, optional
        Time spanned per row of the double plot.
        Cf. #timeseries-offset-aliases in
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        Default is '48h' (i.e 2 days).
    Returns
    -------
    fig : Instance of plotly.graph_objects.Figure
        Figure containing the double plot

    """

    # Input data
    data = raw.resampled_data(freq=freq, binarize=binarize)

    # Time span for the double plot. Usually, 48h.
    td = pd.Timedelta(span)

    # Number of stacked periods (depends on the span)
    n_periods = raw.duration() // (td/2)

    # If the recording does not contain N full periods,
    # then pad the recording with 0.
    padding_index = pd.date_range(
        start=data.index[-1]+data.index.freq,
        periods=((n_periods+1)*(td/2)-raw.duration())/raw.data.index.freq,
        freq=freq
    )
    padding_series = pd.Series(
        index=padding_index, data=[0]*len(padding_index)
    )

    padded_data = pd.concat([data, padding_series])

    # Create figure
    fig = make_subplots(
        rows=n_periods, cols=1,
        vertical_spacing=0.4/n_periods,
        x_title='Date time',
        y_title='Counts/epoch',
    )

    for n in range(n_periods):
        start = padded_data.index[0] + n*(td//2)
        end = start + td
        fig.append_trace(
            go.Bar(
                x=padded_data[start:end].index.astype(str),
                y=padded_data[start:end],
                marker=dict(color="Blue"),
            ),
            row=n+1,
            col=1
        )
    fig.update_yaxes(range=[0, data.max()])
    fig.update_layout(title='Actigraphy data', height=1000, showlegend=False)
    return fig
