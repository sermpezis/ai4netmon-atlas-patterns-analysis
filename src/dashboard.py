import streamlit as st
import re

from globals import BIAS_DIMENSIONS
import load_data as ld
import agg_data_utils as adu
import plot_utils as pu
import bias_causes as bc


def collect_numbers(s):
    return [int(i) for i in re.split("[^0-9]", x) if i != ""]


@st.cache_data
def st_load_main(input_meas_ids):
    df, probes_df, asns_df = ld.load_main(input_meas_ids)
    return df, probes_df, asns_df


@st.cache_data
def st_bias_causes_main(input_meas_ids, asns_df, k=5):
    bias_causes_df, bias_causes_pivot_df = bc.bias_causes_main(input_meas_ids, asns_df, k=k)
    return bias_causes_df, bias_causes_pivot_df


@st.cache_data
def st_get_bias_analysis_data(df):
    avg_bias_per_dim_data, avg_bias_data = adu.get_bias_analysis_data(df)
    return avg_bias_per_dim_data, avg_bias_data


initial_input = "1018338, 1004340, 1017820, 1019139, 1017005, 1019222, 1007976, 1035173, 1010732, 1036065"

# ----------------------------------------------------------------------------------------------------------------------
# Streamlit Dashboard title
st.title('RIPE Atlas Measurements Patterns Analysis')

st.write("Welcome to our Streamlit dashboard on analyzing RIPE Atlas measurements!")

input_meas_ids_raw = st.text_input(
    label = "Please enter a set of RIPE Atlas measurement(s) IDs: ",
    value = initial_input
)

input_meas_ids = collect_numbers(input_meas_ids_raw)
# Create necessary datasets from input measurements
df, probes_df, asns_df = st_load_main(input_meas_ids)
# ----------------------------------------------------------------------------------------------------------------------
st.title('Top most frequent ASNs and probes')
# Here we see the counts of all the ASNs and probe IDs that appear in our measurements. We also show the top `top_x_lines` most frequent ASNs and probe IDs for a closer look. The user can change the `top_x_lines` variable to whatever they wish. Furthermore, but changing the `normalize` variable to `True` the plots will show frequncies instead of counts.


# How many of the top most frequent probes/ASNs to show
top_x_lines = st.slider(
    ":calendar: Please select the top most frequent ASNs and probes you want to see",
    1, 500, 100
)
# Specify if we want counts or frequencies (False for counts, True for frequencies)
normalize_option = st.radio(
    "Select type of data you want to see",
    ["Counts", "Frequencies"]
)
normalize = False if normalize_option == "Counts" else True
# Create and plot the probe IDs and ASNs counts bar plots
probes_asns_counts_bar_traces = pu.get_probes_asns_counts_traces(probes_df, asns_df, top_x_lines = top_x_lines, normalize = normalize)
probes_asns_counts_fig = pu.get_probes_asns_counts_fig(probes_asns_counts_bar_traces)
st.plotly_chart(probes_asns_counts_fig)

# ----------------------------------------------------------------------------------------------------------------------
st.title("Histograms of number of probes and asns used per measurement")


# Create and plot the number of probes/ASNs per measurement plots
num_probes_asns_per_meas_traces = pu.get_num_probes_asns_per_meas_traces(df)
num_probes_asns_per_meas_fig = pu.get_num_probes_asns_per_meas_fig(num_probes_asns_per_meas_traces)
st.plotly_chart(num_probes_asns_per_meas_fig)

# ----------------------------------------------------------------------------------------------------------------------
st.title("Bias comparison: Measurements (average) vs random sample of probes")
#
# Here we first find the average number of probes $\hat{N}$ from all input measurements and then compare the average bias value of our input measurements with the average bias of a set of $\hat{N}$ randomly selected probes.


# Get data for the bias analysis plots
avg_bias_per_dim_data, avg_bias_data = st_get_bias_analysis_data(df)


# Avg bias bar plot
average_bias_per_sample_trace = pu.get_average_bias_per_sample_trace(avg_bias_data)
average_bias_per_sample_fig = pu.get_average_bias_per_sample_fig(average_bias_per_sample_trace)
st.plotly_chart(average_bias_per_sample_fig)


# Radar plot
avg_bias_per_dim_radar_traces = pu.get_radar_traces(avg_bias_per_dim_data)
avg_bias_per_dim_radar_fig = pu.get_avg_bias_per_dim_radar_fig(avg_bias_per_dim_radar_traces)
st.plotly_chart(avg_bias_per_dim_radar_fig)


# Avg bias per dimension grouped bar plot
avg_bias_per_dim_bar_traces = pu.get_avg_bias_per_dim_bar_traces(avg_bias_per_dim_data)
avg_bias_per_dim_bar_fig = pu.get_avg_bias_per_dim_bar_fig(avg_bias_per_dim_bar_traces)
st.plotly_chart(avg_bias_per_dim_bar_fig)

# ----------------------------------------------------------------------------------------------------------------------
st.title("Number of probes vs Avg Bias per measurement")


# Get and plot number of probes vs avg bias per measurement data
scatter_data = adu.get_num_probes_avg_bias_meas_scatter_df(df)
scatter_trace = pu.get_scatter_trace(scatter_data, 'num_probes', 'avg_meas_bias')
num_probes_avg_bias_meas_fig = pu.get_num_probes_avg_bias_meas_fig(scatter_trace)
st.plotly_chart(num_probes_avg_bias_meas_fig)

# ----------------------------------------------------------------------------------------------------------------------
st.title("CDF of Bias per Bias dimension")


# Keep only bias dimensions
bias_df = df[BIAS_DIMENSIONS]
cdf_traces = pu.get_cdf_traces(bias_df)
cdfs_fig = pu.create_plot_grid(cdf_traces, one_plot_per_subplot=True)

# Show the plot
st.plotly_chart(cdfs_fig)

# ----------------------------------------------------------------------------------------------------------------------
st.title("Bias Causes")


# In this section, we show the top `k` positive and negative bias causes for each measurement, as well as the top `k` (positive and negative) bias causes for the entire set of input measurements. The user can change the value of the `k`.


# Get and plot bias causes
k = 5 # Feel free to change this value
bias_causes_df, bias_causes_pivot_df = st_bias_causes_main(input_meas_ids, asns_df, k=k)


# In the cell below we show the first 15 of the bias causes for the first measurement from the input measurements, so that we can have a sense of how all the bias causes look.


# Bias causes of first input measurement
# bias_causes_df[bias_causes_df['meas_id'] == str(input_meas_ids[0])].head(15)


# Plot the bias causes heatmap
bias_causes_heatmap_trace = pu.get_bias_causes_heatmap_trace(bias_causes_pivot_df)
bias_causes_heatmap_fig = pu.get_bias_causes_heatmap_fig(bias_causes_heatmap_trace)
st.plotly_chart(bias_causes_heatmap_fig)


"""
Add text for logging in the adu.get_bias_analysis_data() function
"""

