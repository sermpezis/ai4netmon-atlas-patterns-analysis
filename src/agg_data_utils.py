import time
import requests
import pandas as pd
from globals import BIAS_DIMENSIONS, PLOTLY_DEFAULT_COLORS


def get_all_asns(asns_df):
    return list(asns_df['asn'].unique())


def get_meas_asns(meas_id, asns_df):
    return list(asns_df[asns_df['meas_id'] == meas_id].loc[:, 'asn'].unique())


def prepare_probeid_asn_counts_data(df, col, normalize):
    # Convert probe_id or asn column to string
    df[col] = df[col].astype(str)
    # Get the number each probe id or asn appears
    bar_plot_data_df = df[col].value_counts(normalize=normalize).sort_values(ascending=True).reset_index()
    return bar_plot_data_df


def get_med_num_probes(df):
    return int(round(df['num_probes'].median(), 0))


def get_ai4netmon_atlas_random_sample_uri(N):
    return f'https://ai4netmon.csd.auth.gr/api/bias/randomAtlas/{N}'


def get_rand_avg_bias_per_dim(N):
    atlas_random_sample_uri = get_ai4netmon_atlas_random_sample_uri(N)
    response = requests.get(atlas_random_sample_uri).json()['Atlas'][f'{N}']
    return pd.Series(response)


def get_sample_avg_bias_per_dim(df):
    return df[BIAS_DIMENSIONS].reset_index(drop = True).mean()


def get_avg_bias(ser, ndec = 2):
    return round(ser.mean(), ndec)


def unpack_avg_bias_data(avg_bias_data):
    bar_x_vals, bar_y_vals = [], []
    trace_config_dict = {
        'marker_color': []
    }
    samples = list(avg_bias_data.keys())
    for i in range(len(samples)):
        sample = samples[i]
        bar_x_vals.append(sample)
        bar_y_vals.append(avg_bias_data[sample]['data'])
        trace_config_dict['marker_color'].append(PLOTLY_DEFAULT_COLORS[i])

    return bar_x_vals, bar_y_vals, trace_config_dict


def get_bias_analysis_data(df):
    bias_anal_data_start = time.time()
    # Median number of probes in input measurements
    N = get_med_num_probes(df)

    # Get the average bias per dimension for the set of random probes and our input measurements (sample)

    print(f"Getting average bias per dimension for a random sample of {N} RIPE Atlas probes...")
    rand_probe_start = time.time()
    rand_avg_bias_per_dim = get_rand_avg_bias_per_dim(N)
    rand_probe_end = time.time() - rand_probe_start
    print(f"Data retrieved in {rand_probe_end:.3f} seconds.")

    print(f"Getting average bias per dimension for input measurements...")
    meas_bias_start = time.time()
    sample_avg_bias_per_dim = get_sample_avg_bias_per_dim(df)
    meas_bias_end = time.time() - meas_bias_start
    print(f"Data retrieved in {meas_bias_end:.3f} seconds.")

    # Data related to the plot of the average bias per dimension
    avg_bias_per_dim_data = {
        'Measurements': {
            'data': sample_avg_bias_per_dim,
            'color': PLOTLY_DEFAULT_COLORS[0]
        },
        'Random': {
            'data': rand_avg_bias_per_dim,
            'color': PLOTLY_DEFAULT_COLORS[1]
        }
    }

    # Get the total average bias for the set of random probes and our input measurements (sample)
    sample_avg_bias = get_avg_bias(sample_avg_bias_per_dim)
    rand_avg_bias = get_avg_bias(rand_avg_bias_per_dim)

    # Data related to the plot of the average bias
    avg_bias_data = {
        'Measurements': {
            'data': sample_avg_bias,
            'color': PLOTLY_DEFAULT_COLORS[0]
        },
        'Random': {
            'data': rand_avg_bias,
            'color': PLOTLY_DEFAULT_COLORS[1]
        }
    }

    bias_anal_data_end = time.time() - bias_anal_data_start
    print(f"Bias analysis data completed in {bias_anal_data_end:.3f} seconds.")
    return avg_bias_per_dim_data, avg_bias_data


def get_num_probes_avg_bias_meas_scatter_df(df):
    scatter_data = pd.concat([df['num_probes'], df[BIAS_DIMENSIONS].mean(axis = 1)], axis = 1).rename(columns = {0: 'avg_meas_bias'}).sort_values(by = 'num_probes').reset_index()
    scatter_data['hovertext'] = 'Meas ID: ' + scatter_data['meas_id'].astype(str)
    return scatter_data


