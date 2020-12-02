import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyActigraphy.io import BaseRaw


def double_plot(raw, freq='15min', binarize=False, threshold=0, span='48h'):
    """Double plot

    Create a stack plot where each line displays a specified time range
    (usually 48h) with a time overlap of half that range between lines.

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
    threshold: int, optional
        If binarize is set to True, data above this threshold are set to 1
        and to 0 otherwise.
        Default is 0.
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
    data = raw.resampled_data(
        freq=freq, binarize=binarize, threshold=threshold
    )

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


def daily_profile_plot(raw, *args, **kwargs):
    """Daily profile plot

    Create a plot of the average daily profile.

    Parameters
    ----------
    raw: Instance of BaseRaw (or derived class)
        Raw object containing the actigraphy data. If raw is a RawReader,
        the profiles of all BaseRaw elements are superimposed.
    *args
        Variable length argument list passed to the average_daily_activity
        function.
    **kwargs
        Arbitrary keyword arguements passed to the average_daily_activity
        function.

    Returns
    -------
    fig : Instance of plotly.graph_objects.Figure
        Figure containing the daily profile plot

    """
    # Plot title
    title = "Average daily activity profile"

    if isinstance(raw, BaseRaw):
        adapf = {raw.name: raw.average_daily_activity(*args, **kwargs)}
        showlegend = False
    else:
        adapf = raw.average_daily_activity(*args, **kwargs)
        showlegend = True

    return _profile_plot(adapf, title=title, showlegend=showlegend)


def _profile_plot(
    profiles,
    title,
    nticks=48,
    font_size=20,
    showlegend=False,
    height=900,
    width=1600
):
    r"""Profile plot

    Custom vizualisation of profile plots.

    Parameters
    ----------
    profiles: dict of pd.Series
        Profile time series.
    title: str
        Plot title
    nticks: int, optional
        Number of X axis ticks.
        Default is 48.
    font_size: int, optional
        Font size.
        Default is 20.
    showlegend: bool, optional
        If set to True, display legend.
        Default is False.
    height: int, optional
        Plot height
    width: int, optional
        Plot width

    Returns
    -------
    fig : Instance of plotly.graph_objects.Figure
        Figure containing the profile plot
    """

    layout = go.Layout(
        title=title,
        title_font=dict(size=font_size+4),
        xaxis=dict(
            title="Time of day",
            title_font=dict(size=font_size),
            nticks=nticks,
            ticks="outside",
            tickwidth=2,
            tickcolor='crimson',
            tickfont=dict(
                # family='Rockwell', color='crimson',
                size=font_size-4
            )
        ),
        yaxis=dict(
            title="Counts/period",
            title_font=dict(size=font_size)
        ),
        showlegend=showlegend,
        height=height,
        width=width
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=profile.index.to_pytimedelta().astype(str),
                y=profile,
                name=name,
                line_width=3
            ) for name, profile in profiles.items()
        ],
        layout=layout
    )

    return fig
