import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyActigraphy.io import BaseRaw


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def format_timedelta(td):
    if td < pd.Timedelta(0):
        return '-' + format_timedelta(-td)
    else:
        # Format positive timedeltas
        return strfdelta(td, "{hours:02}:{minutes:02}")  # ":{seconds:02}")


def double_plot(
    raw, freq='15min',
    binarize=False,
    threshold=0,
    span='48h',
    max_activity=None,
    bar_color="Blue",
    height=1000
):
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
    max_activity: float, optional
        If set to a value between 0 and 1, set the y-axis range to the maximal
        activity value multiplied by max_activity.
        Otherwise, set the y-axis range to the specified max_activity value.
        If set to None, the y-axis range is scaled between 0 and the maximal
        activity value found in the recording.
        Default is None.
    bar_color: str, optional
        Set bar colour. Available colours are listed in
        https://developer.mozilla.org/en-US/docs/Web/CSS/color_value
        Default is "Blue".
    height: int, optional
        Plot height.
        Default is 1000.

    Returns
    -------
    fig : Instance of plotly.graph_objects.Figure
        Figure containing the double plot

    """
    # Time span for the double plot. Usually, 48h.
    td = pd.Timedelta(span)

    # Input data
    data = raw.resampled_data(
        freq=freq, binarize=binarize, threshold=threshold
    )

    # If the recording does not contain N full periods,
    # starting from midnight, then pad the recording with NaN.

    # Timestamp set at midnight on the first day of the recording
    start = data.index[0].floor(freq='D')
    # start = data.index.normalize()[0]

    # timestamp set at midnight on the last day of the recording
    # NB: add an extra day for the last row of the double plot.
    end = data.index[-1].ceil(freq='D') + (td/2)

    # Pad data backwards/forwards with NaN up to midnigth by reindexing
    padded_data = data.reindex(
        pd.date_range(
            start=start,
            end=end,
            freq='15min'
        ),
        fill_value=pd.NA
    )

    # Number of stacked periods (depends on the span)
    # NB: remove the extra day added for the last row of the double plot
    n_periods = (padded_data.index.values.ptp() // (td/2)) - 1

    # Create figure
    fig = make_subplots(
        rows=n_periods, cols=1,
        vertical_spacing=0.08125/n_periods,
        x_title='Time of day [HH:MM]'
    )

    # Fill in subplots
    for n in range(n_periods):
        start = padded_data.index[0] + n*(td//2)
        end = start + td
        fig.append_trace(
            go.Bar(
                x=padded_data[start:end].index,
                y=padded_data[start:end],
                marker=dict(color=bar_color, line=dict(width=0)),
                name="Days {} & {}".format(n+1, n+2)
            ),
            row=n+1,
            col=1
        )

        # Set title for th y-axis of each subplot
        fig.update_yaxes(
            title_text=padded_data[start:end].index[0].strftime("%Y-%m-%d"),
            title_font=dict(size=height/110),
            # title_font=dict(size=12),
            row=n+1, col=1
        )

    # Add shading
    fig.add_vrect(
        row=1, col=1,
        x0=padded_data.index[0], x1=data.index[0],
        fillcolor="LightGrey", opacity=0.5,
        layer="below", line_width=0,
    )

    fig.add_vrect(
        row=n_periods, col=1,
        x0=data.index[-1], x1=padded_data.index[-1],
        fillcolor="LightGrey", opacity=0.5,
        layer="below", line_width=0,
    )

    # Format plot layout
    fig.update_layout(
        title='Actigraphy data', height=height, showlegend=False, bargap=0,
    )
    fig.update_xaxes(showticklabels=False)  # hide xticks for all subplots
    fig.update_xaxes(
        ticks="outside", tickwidth=2, tickcolor='crimson',
        ticklen=10, nticks=12, showticklabels=True,
        row=n_periods, col=1,  # show xticks on last subplot
    )
    fig.update_xaxes(
        tickvals=padded_data[start:end].index[::12],
        ticktext=list(
            pd.timedelta_range(
                start='0h',
                end=span,
                periods=len(padded_data[start:end].index[::12])
            ).map(format_timedelta)
        )
    )

    # Format Y axis
    if max_activity is not None:
        if max_activity < 1:
            max_activity = data.max()*max_activity
    else:
        max_activity = data.max()
    fig.update_yaxes(range=[0, max_activity])

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


def _scoring_plot(
    data,
    scoring,
    title,
    labels={'data': 'Activity', 'scoring': 'Scores'},
    fillarea=True,
    nticks=48,
    font_size=20,
    showlegend=False,
    height=800,
    width=1600
):
    r"""Scoring plot

    Custom vizualisation of scoring plots.

    Parameters
    ----------
    data: pd.Series
        Activity time series.
    scoring: pd.Series
        Scoring time series.
    title: str
        Plot title
    labels: dict, optional
        Dictionary of labels for the data and the scoring.
    fillarea: bool, optional
        If set to True, color the fill area below the scoring curve.
        Default is True.
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
            title="Date time",
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
        yaxis2=dict(
            title='Classification',
            title_font=dict(size=font_size),
            nticks=2,
            overlaying='y',
            side='right'
        ),
        showlegend=showlegend,
        height=height,
        width=width
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=data.index.astype(str),
                y=data,
                name=labels["data"],
                line_width=3
            ),
            go.Scatter(
                x=scoring.index.astype(str),
                y=scoring,
                yaxis='y2',
                name=labels["scoring"],
                line_width=1,
                line_dash='dash',
                fill='tozeroy' if fillarea else None
            )
        ],
        layout=layout
    )

    # Make both y axes coincide at zero.
    fig.update_yaxes(rangemode='tozero')

    return fig


def scoring_plot(
    raw, scoring, freq='15min', binarize=False, threshold=0, fillarea=True
):
    """Scoring plot

    Create a plot where activity counts and scoring are superimposed. If the
    resampling frequency is different from the original scoring frequency, the
    mean scores are calculated for each resampled period.

    Parameters
    ----------
    raw: Instance of BaseRaw (or derived class)
        Raw object containing the actigraphy data
    scoring: pandas.Series
        Scoring
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
    fillarea: bool, optional
        If set to True, color the fill area below the scoring curve.
        Default is True.
    Returns
    -------
    fig : Instance of plotly.graph_objects.Figure
        Figure containing the double plot

    """
    # Define scoring label
    labels = {'data': 'Activity', 'scoring': 'Scores'}
    if scoring.index.freq != pd.Timedelta(freq):
        labels['scoring'] = 'Mean scores'

    # Input data
    data = raw.resampled_data(
        freq=freq, binarize=binarize, threshold=threshold
    )
    scoring = scoring.resample(freq).mean()

    return _scoring_plot(
        data,
        scoring,
        title="Rest-activity scoring",
        labels=labels,
        fillarea=fillarea,
        showlegend=True
    )
