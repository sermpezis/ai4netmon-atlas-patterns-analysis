import streamlit as st
import re

from globals import BIAS_DIMENSIONS
import load_data as ld
import agg_data_utils as adu
import plot_utils as pu
import bias_causes as bc


# Table of Contents class
class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def subheader(self, text):
        self._markdown(text, "h2", " " * 2)

    def subsubheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()
        # key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


def collect_numbers(s):
    return [int(i) for i in re.split("[^0-9]", s) if i != ""]


@st.cache_data(ttl=300, show_spinner='Loading data...')
def st_load_main(input_meas_ids):
    df, probes_df, asns_df = ld.load_main(input_meas_ids)
    return df, probes_df, asns_df


@st.cache_data(ttl=300, show_spinner='Preparing bias comparison data...')
def st_get_bias_analysis_data(df):
    avg_bias_per_dim_data, avg_bias_data = adu.get_bias_analysis_data(df)
    return avg_bias_per_dim_data, avg_bias_data


@st.cache_data(ttl=300, show_spinner='Getting bias causes...')
def st_get_bias_causes(input_meas_ids, asns_df):
    bias_causes_df = bc.get_bias_causes(input_meas_ids, asns_df)
    return bias_causes_df


def convert_df_to_csv(df):
   return df.to_csv(index=True).encode('utf-8')


# ----------------------------------------------------------------------------------------------------------------------
# Tab name and icon
st.set_page_config(
    page_title = 'Dashboard',
    page_icon = ':bar_chart:',
    layout = 'wide'
)

with st.sidebar:
    st.title(":clipboard: Table of contents")
    toc = Toc()
    toc.placeholder()

    st.title(":information_source: About")
    st.info(
        """
        This dashboard was created within the context of the 
        [AI4NetMon](https://app-ai4netmon.csd.auth.gr/) project: a project dedicated to helping users discover bias 
        aspects in Internet measurement platforms, their own measurements as well as providing them with 
        recommendations on fixing them.
        """
    )

# Initial input measurement IDs
initial_input = "1018338, 1004340, 1017820, 1019139, 1017005, 1019222, 1007976, 1035173, 1010732, 1036065"

# ----------------------------------------------------------------------------------------------------------------------
# Streamlit Dashboard title
toc.title('RIPE Atlas Measurements Patterns Analysis')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.markdown(
    """
    Hi! :wave: Welcome to our interactive Streamlit dashboard for analyzing RIPE Atlas measurements! :bar_chart:
    
    In this dashboard you are able to see an analysis of your RIPE Atlas measurements. More specifically, you can see 
    things such as the top most frequent ASNs and probes in your measurements, how your measurements' bias compares to 
    the bias of random samples of probes (:point_up: which, by the way, have the lowest bias possible!), the causes of 
    bias in your measurements and more!
    
    In addition, you can interact with your data via our various widgets! :gear:
    
    To begin, you can simply input the measurement IDs of your measurements separated by commas in the box below.
    We have already provided an example case for you below! :heavy_check_mark:
    """
)

# Get input measurement IDs from the user
input_meas_ids_raw = st.text_input(
    label = ":pencil: Please enter a set of RIPE Atlas measurement(s) IDs:",
    value = initial_input
)
# Parse input measurement IDs
input_meas_ids = collect_numbers(input_meas_ids_raw)

# Create necessary datasets from input measurements
df, probes_df, asns_df = st_load_main(input_meas_ids)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
toc.title("General statistics of measurements")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader('Top most frequent ASNs and probes')

st.markdown(
    """
    :point_right: In this section you can see the top most frequent ASNs and Probe IDs in your measurements. You can select if you
    want to display their frequencies or counts. You can also select the top most frequent ASNs and probes you want
    to display and also download them as `.csv` files. Give it a go!
    """
)

# Specify if we want counts or frequencies (False for counts, True for frequencies)
normalize_option = st.radio(
    ":chart_with_upwards_trend: Select type of data you want to see",
    ["Counts", "Frequencies"]
)

data_type = "count" if normalize_option == "Counts" else "frequency"

fig_config_dict = {
    'x_axis_title': normalize_option,
    # 'height': 700
}

# Probes --------------------------------------------------
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.subheader(f"Probe ID {normalize_option.lower()} sorted by {data_type}")
    # How many of the top most frequent probes to show. Defaults to all
    max_probes = probes_df['probe_id'].unique().shape[0]
    probes_show_lines = st.slider(
        ":level_slider: Please select the top most frequent probes you want to see",
        1, min(100, max_probes), max_probes
    )

    # Plot probes counts bar plot
    probes_bar_trace_data_dict, probes_bar_data_df = pu.get_probes_asns_bar_trace(
        probes_df, 'probe_id', normalize_option, probes_show_lines
    )
    probes_bar_plot_fig = pu.get_bar_fig(probes_bar_trace_data_dict, fig_config_dict)
    st.plotly_chart(probes_bar_plot_fig, use_container_width=True)
with col2:
    st.subheader(f"Probe ID {normalize_option.lower()} data")
    st.dataframe(probes_bar_data_df, use_container_width=True, hide_index=True)
    # Option to download data
    csv = convert_df_to_csv(probes_bar_data_df)
    st.download_button(
        ":arrow_heading_down: Download data as .csv",
        csv,
        f"probe_id_{normalize_option.lower()}.csv",
        "text/csv",
        key='download-probes-csv'
    )

# ASNs -----------------------------------------------------
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.subheader(f"ASN {normalize_option.lower()} sorted by {data_type}")
    # How many of the top most frequent probes to show. Defaults to all
    max_asns = asns_df['asn'].unique().shape[0]
    asns_show_lines = st.slider(
        ":level_slider: Please select the top most frequent ASNs you want to see",
        1, min(100, max_asns), max_asns
    )

    # Plot ASNs counts bar plot
    asns_bar_trace_data_dict, asns_bar_data_df = pu.get_probes_asns_bar_trace(
        asns_df, 'asn', normalize_option, asns_show_lines
    )
    asns_bar_plot_fig = pu.get_bar_fig(asns_bar_trace_data_dict, fig_config_dict)
    st.plotly_chart(asns_bar_plot_fig, use_container_width=True)
with col2:
    st.subheader(f"ASN {normalize_option.lower()} data")
    st.dataframe(asns_bar_data_df, use_container_width=True)
    # Option to download data
    csv = convert_df_to_csv(asns_bar_data_df)
    st.download_button(
        ":arrow_heading_down: Download data as .csv",
        csv,
        f"probe_id_{normalize_option.lower()}.csv",
        "text/csv",
        key='download-asns-csv'
    )

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("Number of probes and ASNs used per measurement")
st.markdown(
    """
    :point_right: In this section, you can see the number of probes and ASNs per measurement for all your measurements. The y-axes of 
    the plots below are ordered by measurement ID.
    """
)

# Create and plot the number of probes/ASNs per measurement plots
num_probes_asns_per_meas_traces = pu.get_num_probes_asns_per_meas_traces(df)
num_probes_asns_per_meas_fig = pu.get_num_probes_asns_per_meas_fig(num_probes_asns_per_meas_traces)
st.plotly_chart(num_probes_asns_per_meas_fig, use_container_width=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
toc.title("Bias analysis of measurements")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("Bias comparison: Measurements (average) vs random sample of probes")

# Here we first find the average number of probes $\hat{N}$ from all input measurements and then compare the average bias value of our input measurements with the average bias of a set of $\hat{N}$ randomly selected probes.
st.markdown(
    """
    :point_right: In this section you can see information about the average bias distribution for each bias dimension as well as a comparison
    between the average bias of your measurements and a sample of randomly selected probes.
    
    :information_source: To get the random probes sample, first the median number of probes $\hat{N}$ from all your measurements is 
    calculated, then a random sample of $\hat{N}$ probes is created and finally it is compared to your measurements.
    
    :point_right: You can also choose how you want to visualize the average bias distributions: via a radar plot or a bar plot.
    """
)

# Get data for the bias analysis plots
avg_bias_per_dim_data, avg_bias_data = st_get_bias_analysis_data(df)
meas_avg_bias = avg_bias_data['Measurements']['data']
rand_avg_bias = avg_bias_data['Random']['data']


col1, col2 = st.columns([0.3, 0.7])
with col1:
    # Avg bias bar plot
    average_bias_per_sample_trace = pu.get_average_bias_per_sample_trace(avg_bias_data)
    average_bias_per_sample_fig = pu.get_average_bias_per_sample_fig(average_bias_per_sample_trace)
    st.plotly_chart(average_bias_per_sample_fig, use_container_width=True)

    st.info(
        """
        :exclamation: A large difference of bias between the input measurements and the random sample, implies that 
        random sampling can decrease your measurements' bias. Furthermore, the bias breakdown by bias dimension 
        can give you more detailed information about which dimensions' bias values can be decreased the most.
        """
    )
with col2:
    # Radar plot
    avg_bias_per_dim_radar_traces = pu.get_radar_traces(avg_bias_per_dim_data)
    avg_bias_per_dim_radar_fig = pu.get_avg_bias_per_dim_radar_fig(avg_bias_per_dim_radar_traces)
    # Avg bias per dimension grouped bar plot
    avg_bias_per_dim_bar_traces = pu.get_avg_bias_per_dim_bar_traces(avg_bias_per_dim_data)
    avg_bias_per_dim_bar_fig = pu.get_avg_bias_per_dim_bar_fig(avg_bias_per_dim_bar_traces)

    tab1, tab2 = st.tabs(["Radar plot", "Bar plot"])
    with tab1:
        st.plotly_chart(avg_bias_per_dim_radar_fig, use_container_width=True)
    with tab2:
        st.plotly_chart(avg_bias_per_dim_bar_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("Number of probes vs Avg Bias per measurement")
st.markdown(
    """
    :point_right: Below you can see a scatter plot, showing you the average bias per measurement vs its number of probes. Each dot in 
    the plot represents one of your measurements.
    """
)

# Get and plot number of probes vs avg bias per measurement data
scatter_data = adu.get_num_probes_avg_bias_meas_scatter_df(df)
scatter_trace = pu.get_scatter_trace(scatter_data, 'num_probes', 'avg_meas_bias')
num_probes_avg_bias_meas_fig = pu.get_num_probes_avg_bias_meas_fig(scatter_trace)
st.plotly_chart(num_probes_avg_bias_meas_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("Bias Causes")
st.markdown(
    """
    :point_right: In this section, you can see the top `k` positive and negative bias causes for each measurement, as well as the top `k` 
    (positive and negative) bias causes for the entire set of input measurements.
    """
)

# Get all bias causes data
bias_causes_df = st_get_bias_causes(input_meas_ids, asns_df)

# Get top k positive and negative bias causes for each measurement as well as for all measurements
# How many of the top positive and negative bias causes to show
k = st.slider(
    ":level_slider: Please select the top `k` positive and negative bias causes to show",
    1, 10, 5
)
bias_causes_pivot_df = bc.get_dashboard_bias_causes_data(input_meas_ids, bias_causes_df, k)

# Plot the bias causes heatmap
bias_causes_heatmap_trace = pu.get_bias_causes_heatmap_trace(bias_causes_pivot_df)
bias_causes_heatmap_fig = pu.get_bias_causes_heatmap_fig(bias_causes_heatmap_trace)
st.plotly_chart(bias_causes_heatmap_fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("CDF of Bias per Bias dimension")

st.markdown(
    """
    :point_right: In this subsection, you can click on the expander below to take a look at the CDF plots of the bias values for
    all bias dimensions.
    """
)

# Keep only bias dimensions
bias_df = df[BIAS_DIMENSIONS]
cdf_traces = pu.get_cdf_traces(bias_df)
cdfs_fig = pu.create_plot_grid(cdf_traces, one_plot_per_subplot=True)
with st.expander(":eyes: Show CDFs"):
    # Show the plot
    st.plotly_chart(cdfs_fig)

# ----------------------------------------------------------------------------------------------------------------------
toc.subheader("Data details")

st.markdown(
    """
    :point_right: Here you can check out the data themselves and also download them as a `.csv` file.
    """
)

# Use an expander so that the user can see the generated dataframe
# with st.expander(":mag_right: Measurements' metadata and bias dataframe"):
with st.expander(":eyes: Measurements' metadata and bias dataframe"):
    show_df = df.copy()
    show_df.index = show_df.index.map(str)
    st.dataframe(df)

    csv = convert_df_to_csv(df)
    st.download_button(
        ":arrow_heading_down: Download data as .csv",
        csv,
        "measuremet_data.csv",
        "text/csv",
        key='download-csv'
    )
# ----------------------------------------------------------------------------------------------------------------------
toc.generate()


