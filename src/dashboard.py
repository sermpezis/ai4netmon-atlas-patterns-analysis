"""
To dos:
1) Figure out how to plot the top ASNs and Probes wrt the top most frequent ASNs and probes. Should you use separate plots
    for ASNs and probes as well as separate sliders?
2) Add option to select only specific bias dimensions?
3) Bias causes: make it so that when the user selects some value for k, then you don't have to re-run the entire bias
    causes script! (i.e. make it configurable at the plot level).
4) Add "Export data" option in various place (e.g. top ASNs and probes).
5) CDFs: Increase number of columns in the grid? Or add some text to the side? Think about it.

- Number of probes and ASNs put together in one plot?
"""

import streamlit as st
import re

from globals import BIAS_DIMENSIONS
import load_data as ld
import agg_data_utils as adu
import plot_utils as pu
import bias_causes as bc



def collect_numbers(s):
    return [int(i) for i in re.split("[^0-9]", s) if i != ""]


@st.cache_data
def st_load_main(input_meas_ids):
    df, probes_df, asns_df = ld.load_main(input_meas_ids)
    return df, probes_df, asns_df


@st.cache_data
def st_get_bias_analysis_data(df):
    avg_bias_per_dim_data, avg_bias_data = adu.get_bias_analysis_data(df)
    return avg_bias_per_dim_data, avg_bias_data


@st.cache_data
def st_get_bias_causes(input_meas_ids, asns_df):
    bias_causes_df = bc.get_bias_causes(input_meas_ids, asns_df)
    return bias_causes_df


# Initial input measurement IDs
initial_input = "1018338, 1004340, 1017820, 1019139, 1017005, 1019222, 1007976, 1035173, 1010732, 1036065"

# ----------------------------------------------------------------------------------------------------------------------
# Tab name and icon
st.set_page_config(
    page_title = 'Dashboard',
    page_icon = ':bar_chart:',
    layout = 'wide'
)

# Streamlit Dashboard title
st.title('RIPE Atlas Measurements Patterns Analysis')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.write("Welcome to our Streamlit dashboard on analyzing RIPE Atlas measurements!")

# Get input measurement IDs from the user
input_meas_ids_raw = st.text_input(
    label = "Please enter a set of RIPE Atlas measurement(s) IDs: ",
    value = initial_input
)
# Parse input measurement IDs
input_meas_ids = collect_numbers(input_meas_ids_raw)

# Create necessary datasets from input measurements
df, probes_df, asns_df = st_load_main(input_meas_ids)

# Use an expander so that the user can see the generated dataframe
with st.expander("Measurements' metadata and bias dataframe"):
    show_df = df.copy()
    show_df.index = show_df.index.map(str)
    st.dataframe(df)

# ----------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns([0.6, 0.4])

with col1:
    # Here we see the counts of all the ASNs and probe IDs that appear in our measurements. We also show the top `top_x_lines` most frequent ASNs and probe IDs for a closer look. The user can change the `top_x_lines` variable to whatever they wish. Furthermore, but changing the `normalize` variable to `True` the plots will show frequncies instead of counts.
    st.subheader('Top most frequent ASNs and probes')

    # How many of the top most frequent probes/ASNs to show
    top_x_lines = st.slider(
        ":calendar: Please select the top most frequent ASNs and probes you want to see",
        1, 500, 500
    )
    # Specify if we want counts or frequencies (False for counts, True for frequencies)
    normalize_option = st.radio(
        "Select type of data you want to see",
        ["Counts", "Frequencies"]
    )
    normalize = False if normalize_option == "Counts" else True
    # Create and plot the probe IDs and ASNs counts bar plots
    probes_asns_counts_bar_traces = pu.get_probes_asns_counts_traces(probes_df, asns_df, top_x_lines = top_x_lines, normalize = normalize)
    # probes_asns_counts_fig = pu.get_probes_asns_counts_fig(probes_asns_counts_bar_traces)
    probes_asns_counts_fig = pu.get_subset_probes_asns_counts_fig(probes_asns_counts_bar_traces)
    st.plotly_chart(probes_asns_counts_fig, use_container_width=True)
with col2:
    st.subheader("Number of probes and ASNs used per measurement")

    # Create and plot the number of probes/ASNs per measurement plots
    num_probes_asns_per_meas_traces = pu.get_num_probes_asns_per_meas_traces(df)
    num_probes_asns_per_meas_fig = pu.get_num_probes_asns_per_meas_fig(num_probes_asns_per_meas_traces)
    st.plotly_chart(num_probes_asns_per_meas_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader("Bias comparison: Measurements (average) vs random sample of probes")
#
# Here we first find the average number of probes $\hat{N}$ from all input measurements and then compare the average bias value of our input measurements with the average bias of a set of $\hat{N}$ randomly selected probes.
st.markdown("Here we first find the median number of probes $\hat{N}$ from all input measurements and then "
                "compare the average bias value of our input measurements with the average bias of a set of $\hat{N}$ "
                "randomly selected probes.")

# Get data for the bias analysis plots
avg_bias_per_dim_data, avg_bias_data = st_get_bias_analysis_data(df)
meas_avg_bias = avg_bias_data['Measurements']['data']
rand_avg_bias = avg_bias_data['Random']['data']

# meas_avg_bias = avg_bias_data['Measurements']['data']
# rand_avg_bias = avg_bias_data['Random']['data']
# st.metric("Measurements", meas_avg_bias)
# st.metric("Random", rand_avg_bias)

col1, col2 = st.columns([0.6, 0.4])
with col1:
    col11, col22 = st.columns([0.2, 0.80])
    with col11:
        # Avg bias bar plot
        average_bias_per_sample_trace = pu.get_average_bias_per_sample_trace(avg_bias_data)
        average_bias_per_sample_fig = pu.get_average_bias_per_sample_fig(average_bias_per_sample_trace)
        st.plotly_chart(average_bias_per_sample_fig, use_container_width=True)
    with col22:
        # Radar plot
        avg_bias_per_dim_radar_traces = pu.get_radar_traces(avg_bias_per_dim_data)
        avg_bias_per_dim_radar_fig = pu.get_avg_bias_per_dim_radar_fig(avg_bias_per_dim_radar_traces)
        st.plotly_chart(avg_bias_per_dim_radar_fig, use_container_width=True)

    st.markdown(":exclamation: A large difference of bias between the input measurements and the random sample, "
                "implies that random sampling can decrease our input measurements' bias. Furthermore, the bias "
                "breakdown by bias dimension can give us more detailed information about which dimensions' "
                "bias values can be decreased the most.")

with col2:
    # Avg bias per dimension grouped bar plot
    avg_bias_per_dim_bar_traces = pu.get_avg_bias_per_dim_bar_traces(avg_bias_per_dim_data)
    avg_bias_per_dim_bar_fig = pu.get_avg_bias_per_dim_bar_fig(avg_bias_per_dim_bar_traces)
    st.plotly_chart(avg_bias_per_dim_bar_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Number of probes vs Avg Bias per measurement")

    # Get and plot number of probes vs avg bias per measurement data
    scatter_data = adu.get_num_probes_avg_bias_meas_scatter_df(df)
    scatter_trace = pu.get_scatter_trace(scatter_data, 'num_probes', 'avg_meas_bias')
    num_probes_avg_bias_meas_fig = pu.get_num_probes_avg_bias_meas_fig(scatter_trace)
    st.plotly_chart(num_probes_avg_bias_meas_fig, use_container_width=True)

with col2:
    st.subheader("Bias Causes")
    st.markdown("In this section, we show the top `k` positive and negative bias causes for each measurement, "
                "as well as the top `k` (positive and negative) bias causes for the entire set of input measurements.")

    # Get all bias causes data
    bias_causes_df = st_get_bias_causes(input_meas_ids, asns_df)

    # Get top k positive and negative bias causes for each measurement as well as for all measurements
    # How many of the top positive and negative bias causes to show
    k = st.slider(
        "Please select the top positive and negative bias causes to show",
        1, 20, 5
    )
    bias_causes_pivot_df = bc.get_dashboard_bias_causes_data(input_meas_ids, bias_causes_df, k)

    # Plot the bias causes heatmap
    bias_causes_heatmap_trace = pu.get_bias_causes_heatmap_trace(bias_causes_pivot_df)
    bias_causes_heatmap_fig = pu.get_bias_causes_heatmap_fig(bias_causes_heatmap_trace)
    st.plotly_chart(bias_causes_heatmap_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
st.subheader("CDF of Bias per Bias dimension")


# Keep only bias dimensions
bias_df = df[BIAS_DIMENSIONS]
cdf_traces = pu.get_cdf_traces(bias_df)
cdfs_fig = pu.create_plot_grid(cdf_traces, one_plot_per_subplot=True)

# Show the plot
st.plotly_chart(cdfs_fig)

# ----------------------------------------------------------------------------------------------------------------------



