import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from globals import BIAS_DIMENSIONS, PLOTLY_DEFAULT_COLORS

import agg_data_utils as adu


def get_bar_trace(x_vals, y_vals, trace_config_dict={}):
    # Configure optional arguments for the bar trace
    showlegend = trace_config_dict.get('showlegend', False)
    marker_color = trace_config_dict.get('marker_color', PLOTLY_DEFAULT_COLORS[0])
    orientation = trace_config_dict.get('orientation', None)
    name = trace_config_dict.get('name', None)
    # Create bar trace
    bar_trace = go.Bar(
        x=x_vals,
        y=y_vals,
        showlegend=showlegend,
        marker_color=marker_color,
        orientation=orientation,
        name=name
    )

    return bar_trace


def get_probes_asns_counts_traces(probes_df, asns_df, top_x_lines=100, normalize=False):
    """
    Input:
        - probes_df (pd.DataFrame): Dataframe containing each probe ID and measurement ID combination as separate lines.
        - asns_df (pd.DataFrame): Dataframe containing each probe ID and measurement ID combination as separate lines.
        - top_x_lines (int): Top X most frequent probe IDs and ASNs to show. Default value is 100.
        - normalize (bool): Specifies if we want counts or frequencies in our plots (False for counts, True for frequencies). Defaults to counts.
    Output:
        - bar_traces (list[dict]): Contains the data and the title of each bar trace we plot.

    This function prepares the data for the top most frequent probes and ASNs. For each of them it creates bar traces and their corresponding titles for two different cases: All probes/ASNs and the top top_x_lines probes and ASNs.
    """
    if normalize:
        count_data_type = 'frequencies'
    else:
        count_data_type = 'counts'

    # Create bar traces and their titles for probe id and asn counts plots
    bar_traces = []
    for dfp, col in zip([probes_df, asns_df], ['probe_id', 'asn']):
        # Prepare plot data
        bar_data = adu.prepare_probeid_asn_counts_data(dfp, col, normalize)
        all_lines = bar_data.shape[0]
        # For each of probe IDs and ASNs we show all the counts and the top top_x_lines most frequent probe IDs/ASNs
        lines = [all_lines, top_x_lines]
        for show_lines in lines:
            # Get the plot trace
            x_vals = bar_data.iloc[:show_lines, 0].values
            y_vals = bar_data.iloc[:show_lines, 1].values
            bar_trace = get_bar_trace(x_vals, y_vals)
            # Trace title
            if show_lines == all_lines:
                title = f'{col} ' + count_data_type
            else:
                title = f'{col} ' + count_data_type + f' (showing top {show_lines}/{all_lines})'
            # Create the trace dictionary
            bar_trace_dict = {
                'title': title,
                'trace': bar_trace
            }
            bar_traces.append(bar_trace_dict)
    return bar_traces


def get_probes_asns_counts_fig(bar_traces):
    # Create subplots with the specified grid layout
    rows = 2
    cols = 2
    reordered_bar_traces = [
        bar_traces[0],  # All probe IDs' counts
        bar_traces[2],  # All ASNs' counts
        bar_traces[1],  # Top top_x_lines most frequent probe IDs' counts
        bar_traces[3]  # Top top_x_lines most frequent ASNs' counts
    ]
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[d['title'] for d in reordered_bar_traces])

    # Traces are appened into the figure in the same order as reordered_bar_traces
    fig.add_trace(bar_traces[0]['trace'], row=1, col=1)
    fig.add_trace(bar_traces[2]['trace'], row=1, col=2)
    fig.add_trace(bar_traces[1]['trace'], row=2, col=1)
    fig.add_trace(bar_traces[3]['trace'], row=2, col=2)

    fig.update_traces(marker_color=PLOTLY_DEFAULT_COLORS[2])
    fig.update_xaxes(visible=False, type='category', row=1, col=1)
    fig.update_xaxes(visible=False, type='category', row=1, col=2)
    fig.update_xaxes(visible=False, type='category', row=2, col=1)
    fig.update_xaxes(visible=False, type='category', row=2, col=2)

    # Update the layout
    fig.update_layout(
        width=1200,
        height=700
    )

    return fig


def get_num_probes_asns_per_meas_traces(df):
    trace_config_dict = {
        'orientation': 'h'
    }
    x_vals = df['num_probes'].values
    y_vals = list(df.index.astype(str))
    num_probes_per_meas_trace = get_bar_trace(x_vals, y_vals, trace_config_dict = trace_config_dict)
    num_probes_per_meas_trace_dict = {
        'trace': num_probes_per_meas_trace,
        'title': 'Number of probes per measurement'
    }
    x_vals = df['num_asns'].values
    num_asns_per_meas_trace = get_bar_trace(x_vals, y_vals, trace_config_dict = trace_config_dict)
    num_asns_per_meas_trace_dict = {
        'trace': num_asns_per_meas_trace,
        'title': 'Number of ASNs per measurement'
    }

    num_probes_asns_per_meas_traces = [num_probes_per_meas_trace_dict, num_asns_per_meas_trace_dict]

    return num_probes_asns_per_meas_traces


def get_num_probes_asns_per_meas_fig(num_probes_asns_per_meas_traces):
    rows = 1
    cols = 2
    fig = make_subplots(
        rows = rows,
        cols = cols,
        subplot_titles = [d['title'] for d in num_probes_asns_per_meas_traces],
        shared_xaxes = 'all'
    )

    fig.add_trace(num_probes_asns_per_meas_traces[0]['trace'], row = 1, col = 1)
    fig.update_yaxes(type = 'category', row = 1, col = 1)
    fig.add_trace(num_probes_asns_per_meas_traces[1]['trace'], row = 1, col = 2)
    fig.update_yaxes(type='category', row=1, col=2)

    fig.update_layout(
        width = 1000,
        height = 700
    )

    return fig


def get_radar_trace(data, name, color):
    r = list(data.values)
    r = [*r, r[0]]
    theta = list(data.index)
    theta = [*theta, theta[0]]

    trace = go.Scatterpolar(
        r=r,
        theta=theta,
        name=name,
        line_color=color
    )

    return trace


def get_radar_traces(radar_data_dict):
    radar_traces = []
    for name, data_dict in radar_data_dict.items():
        data = data_dict['data']
        color = data_dict['color']
        trace = get_radar_trace(data, name, color)
        radar_traces.append(trace)

    return radar_traces


def get_average_bias_per_sample_trace(avg_bias_data):
    bar_x_vals, bar_y_vals, trace_config_dict = adu.unpack_avg_bias_data(avg_bias_data)
    average_bias_per_sample_trace = get_bar_trace(bar_x_vals, bar_y_vals, trace_config_dict)
    return average_bias_per_sample_trace


def get_average_bias_per_sample_fig(average_bias_per_sample_trace):
    fig = go.Figure(average_bias_per_sample_trace)

    fig.update_layout(
        title="Average Bias per sample",
        xaxis_title="Sample",
        yaxis_title="Average Bias",
        height=500,
        width=600
    )

    return fig


def get_avg_bias_per_dim_radar_fig(radar_traces):
    fig = go.Figure()
    for trace in radar_traces:
        fig.add_trace(trace)

    fig.update_traces(opacity=0.6, fill='toself')

    theta = BIAS_DIMENSIONS
    theta_labels = [x.replace(' (', '<br>(') for x in [*theta, theta[0]]]

    fig.update_layout(
        height=500,
        width=700,
        title=f'Average Bias Distribution for Measurements and Random sample',
        font={
            'size': 10
        },
        polar={
            'radialaxis': {
                'visible': True,
                'range': [0, 1]
            },
            'angularaxis': {
                'rotation': 90,
                'ticktext': theta_labels
            }
        },

    )

    return fig


def get_avg_bias_per_dim_bar_traces(avg_bias_per_dim_data):
    sample_trace_config_dict = {
        'orientation': 'h',
        'name': 'Measurements',
        'marker_color': PLOTLY_DEFAULT_COLORS[0],
        'showlegend': True
    }
    sample_avg_bias_per_dim = avg_bias_per_dim_data['Measurements']['data']
    sample_bar_trace = get_bar_trace(
        sample_avg_bias_per_dim.sort_values().values,
        sample_avg_bias_per_dim.sort_values().index,
        sample_trace_config_dict
    )

    rand_trace_config_dict = {
        'orientation': 'h',
        'name': 'Random',
        'marker_color': PLOTLY_DEFAULT_COLORS[1],
        'showlegend': True
    }
    rand_avg_bias_per_dim = avg_bias_per_dim_data['Random']['data']
    rand_bar_trace = get_bar_trace(
        rand_avg_bias_per_dim.sort_values().values,
        rand_avg_bias_per_dim.sort_values().index,
        rand_trace_config_dict
    )

    bar_traces = [
        sample_bar_trace,
        rand_bar_trace
    ]
    return bar_traces


def get_avg_bias_per_dim_bar_fig(bar_traces):
    # Create the bar plot with custom colors
    fig = go.Figure()

    for bar_trace in bar_traces:
        fig.add_trace(bar_trace)

    # Customize the layout if needed
    fig.update_layout(
        title='Avg Bias per Dimension',
        xaxis_title='Avg Bias',
        yaxis_title='Dimension',
        height=700,
        width=600
    )

    return fig


def get_scatter_trace(scatter_data, x_col, y_col):
    scatter_trace = go.Scatter(
        x = scatter_data[x_col],
        y = scatter_data[y_col],
        text = scatter_data['hovertext']
    )
    return scatter_trace


def get_num_probes_avg_bias_meas_fig(scatter_trace):
    fig = go.Figure(scatter_trace)

    fig.update_layout(
        title_text = 'Number of probes vs Average bias for each measurement',
        xaxis_title = 'Number of probes',
        yaxis_title = 'Average bias',
        width = 800,
        height = 500
    )

    fig.update_xaxes(type="log")

    return fig


def get_cdf(arr, showlegend=False):
    """
    Returns cdf plot of arr
    """

    cdf = ECDF(arr)

    trace = go.Scatter(
        x=cdf.x,
        y=cdf.y,
        mode='lines',
        line=dict(color=PLOTLY_DEFAULT_COLORS[0]),
        name='Sample',
        showlegend=showlegend
    )

    return trace


def get_cdf_traces(data_df):
    all_traces = []
    data_cols = list(data_df.columns)
    # Get plot data
    for i in range(data_df.shape[1]):
        trace_dict = {}
        plot_data = data_df.iloc[:, i].reset_index(drop=True)
        if i == 0:
            showlegend = True
        else:
            showlegend = False

        # Get cdf trace
        cdf_trace = get_cdf(plot_data, showlegend=showlegend)

        trace_dict['title'] = data_cols[i]
        trace_dict['traces'] = cdf_trace
        all_traces.append(trace_dict)

    return all_traces


def create_plot_grid(all_traces, rows=5, cols=5, xrange=[0, 1], yrange=[0, 1], one_plot_per_subplot=True):
    # Create subplots with the specified grid layout
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[d['title'] for d in all_traces])

    # Iterate through each row and column index to add histograms to the subplots
    for i in range(rows):
        for j in range(cols):
            row = i + 1
            col = j + 1
            index = i * cols + j
            if index < len(all_traces):
                # Get plot data

                traces = all_traces[index]['traces']
                if one_plot_per_subplot:
                    fig.append_trace(traces, row=row, col=col)
                else:
                    for trace in traces:
                        fig.append_trace(trace, row=row, col=col)
                fig.update_xaxes(range=xrange, row=row, col=col)
                fig.update_yaxes(range=yrange, row=row, col=col)

    # Update the annotations (subplot titles) font size
    title_font_size = 12  # Adjust the font size as desired
    for i in range(rows * cols):
        if i < len(all_traces):
            fig.update_annotations(font_size=title_font_size,
                                   selector=dict(text=all_traces[i]['title'])
                                   )
    # Update the layout
    fig.update_layout(
        title_text="Bias Distribution across each Bias Dimension",
        width=1200,
        height=1200
    )

    return fig


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


def get_bias_causes_heatmap_trace(bias_causes_pivot):
    zmax = max(abs(bias_causes_pivot.min().min()), bias_causes_pivot.max().max())
    trace = go.Heatmap(
            df_to_plotly(bias_causes_pivot),
            colorbar={"title": "Difference from <br> all ASes (%)"},
            colorscale='Picnic',
            # reversescale=True,
            zmax = zmax,
            zmin = -zmax
    )
    return trace


def get_bias_causes_heatmap_fig(bias_causes_trace):
    fig = go.Figure(bias_causes_trace)

    fig.update_layout(
        xaxis_showgrid = True,
        yaxis_showgrid = False,
        xaxis=dict(type='category'),
        xaxis_title = 'Number of probes in sample',
        yaxis_title = 'Bias Dimensions:Bin',
        title_text="Bias causes for each measurement and all measurements",
        width = 1000,
        height = 500
    )

    fig.update_xaxes(
        tickson = 'boundaries',
        tickangle = -45
    )

    return fig

