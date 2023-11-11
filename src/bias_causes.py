import requests
import time
import pandas as pd


def get_meas_bias_causes(asn_list):
    """
    Input:
        - asn_list (list of <int>): List of ASNs.

    Output:
        - bias_causes (dict): Contains bias causes for the bias corresponding to the asn_list sample.

    This function uses an ai4netmon endpoint that returns the bias causes for the sample of ASNs that corresponds to
    the asn_list input. More specifically, the response is the following form:
        bias_causes = {
            'Custom list': {
                '<bias_dim_1>': {
                    '<bin_11>': 'x11%', '<bin_12>': 'x12%',...
                },
                '<bias_dim_2>': {
                    '<bin_21>': 'x21%', '<bin_22>': 'x22%',...
                },
                ...
            },
            '#ASNs not found': <int>
        }
    where:
        > <bias_dim_i>: is any bias dimension where we have bias for the current sample of ASNs.
        > <bin_ij>: is the j-th bin of the bias dimension <bias_dim_i> where there is a mismatch in the
                    distribution of values between our sample and the set of all ASes in RIPE Atlas.
        > 'xij%' <str>: Percentage difference calculated as <all ASes> - <asn_list>

    For example:
        bias_causes = {
            'Customer cone (#ASNs)': {
                '1.0-3.0': '6.164%',
                '3.0-9.0': '-3.7264%',
                '9.0-26.0': '-1.3839%'
            }
            'Customer cone (#prefixes)': {
                '1.0-4.0': '28.6553%',
                '4.0-14.0': '-17.5897%'
            }
        }

    Here, this tells us that in the 'Customer cone (#ASNs)' dimension, for the bin values 1-3 there is a 6% difference
    between all ASes and our sample (asn_list), i.e. our sample has 6% less values in that bin for that dimension
    compared to all ASes. For the bin 3-9, where the value is negative, the interpretation is similar: our sample has
    3.7% more values in that bin compared to all ASes. Therefore, is the user wants to make the measurement containing
    these ASNs less biased, they should add 6% more values in the 1-3 bin and 3% less values in the 3-9 bin for the
    'Customer cone (#ASNs)' dimension.
    """
    base_url = 'https://ai4netmon.csd.auth.gr/api/bias/cause/asn/'
    asn_list_str = '?asn=' + '&asn='.join([str(asn) for asn in asn_list])
    url = base_url + asn_list_str
    bias_causes = requests.get(url).json()
    return bias_causes


def get_sample_bias_causes(input_meas_ids, asns_df):
    all_bias_causes = {}
    for meas_id in input_meas_ids:
        print(f'Processing measurement with id {meas_id}.')
        meas_asns = asns_df[asns_df['meas_id'] == meas_id].loc[:, 'asn'].tolist()
        # print(meas_asns)
        # Skip any measurement that has no ASNs in it
        if len(meas_asns) > 0:
            print(f"{meas_id}: Getting measurement's bias causes...")
            meas_bias_causes_start = time.time()
            meas_bias_causes = get_meas_bias_causes(meas_asns)
            meas_bias_causes_end = time.time() - meas_bias_causes_start
            # print(f'{meas_id}: Got bias causes: {meas_bias_causes}')
            print(f'{meas_id}: Got bias causes.')
            print(f'{meas_id}: Time to get bias causes: {meas_bias_causes_end:.3f}')

            all_bias_causes[f'{meas_id}'] = meas_bias_causes
        else:
            print(f'{meas_id}: This measurement contains no ASNs so it is skipped for the bias causes calculation.')
        print('-' * 100)

    return all_bias_causes


def get_bias_df(meas_causes_dict):
    # Intialize lists that will hold the data of the final dataframe
    meas_id_list = []
    # num_probes_list = []
    bias_causes_list = []
    bias_causes_value_list = []
    # For each measurement we have calculated bias causes for
    for meas_id in meas_causes_dict.keys():
        meas_bias_causes = meas_causes_dict[meas_id]['Custom list']
        for dim in meas_bias_causes.keys():
            # Get bin data for current dimension
            bin_data_raw = meas_bias_causes[dim]

            # Clean bin data (remove '%' and convert to float (from str))
            bin_data = {k: float(v.replace('%', '')) for k, v in bin_data_raw.items()}

            # Add dimension information in the keys of bin_data
            dim_bin_data = {f"{dim}_{bin_}": value for bin_, value in bin_data.items()}

            # Add data to the lists that will be used for the final df creation
            for dim_bin, dim_bin_value in dim_bin_data.items():
                # meas_id_list.append(meas['meas_id'])
                meas_id_list.append(meas_id)
                # num_probes_list.append(int(meas["num_probes"]))
                bias_causes_list.append(dim_bin)
                bias_causes_value_list.append(dim_bin_value)

    bias_causes_df = pd.DataFrame({
        'meas_id': meas_id_list,
        # 'num_probes': num_probes_list,
        'bias_causes': bias_causes_list,
        'value': bias_causes_value_list
    })

    # Remove rows where "Country Influence" is found as a bias cause
    country_influence = 'Country influence'
    bias_causes_df = bias_causes_df[~bias_causes_df['bias_causes'].str.contains(country_influence)]

    return bias_causes_df


def aggregate_bias_causes(bias_causes_df):
    """
    Input:
        - bias_causes_df (pd.DataFrame): See get_bias_causes_df()'s description for details.
    Output:
        - agg_df (pd.DataFrame): Aggregated bias_causes_df

    Aggregates bias_causes_df so that we only have 1 of each dim_bin combination of bias_causes.
    """

    agg_df = (bias_causes_df[['bias_causes', 'value']].
              groupby('bias_causes')['value'].
              median().
              sort_values(ascending=False).
              reset_index())
    return agg_df


def get_meas_topk_causes(input_meas_ids, bias_causes_df, k=5):
    topk_meas_bias_causes = {}
    for meas_id in input_meas_ids:
        # Keep only current measurement's bias causes
        meas_bias_causes_df = bias_causes_df[bias_causes_df['meas_id'] == str(meas_id)].drop(
            columns=['meas_id']).sort_values('value')

        # Get top and bottom k bias_causes
        topk_meas_bias_causes_df = pd.concat([meas_bias_causes_df.head(k), meas_bias_causes_df.tail(k)])
        topk_meas_bias_causes[meas_id] = topk_meas_bias_causes_df
    return topk_meas_bias_causes


def get_topk_agg_bias_causes(bias_causes_df, k=5):
    # Aggregate bias_causes_df so that there are no duplicate bias_causes. agg_df is sorted wrt the 'value' column.
    agg_df = aggregate_bias_causes(bias_causes_df.drop(columns=['meas_id']))
    # Get top and bottom 'topk' bias_causes
    top_agg_df = pd.concat([agg_df.head(k), agg_df.tail(k)])

    return top_agg_df


def get_bias_causes_pivot(meas_bias_causes_dict, examined):
    total_bias_causes_df_list = []
    for meas in meas_bias_causes_dict:
        if meas in examined:
            bias_causes_df = meas_bias_causes_dict[meas]
            bias_causes_df['meas'] = meas
            total_bias_causes_df_list.append(bias_causes_df)
    total_bias_causes_df = pd.concat(total_bias_causes_df_list, ignore_index=True)

    bias_causes_pivot = pd.pivot_table(
        total_bias_causes_df,
        values='value',
        index='bias_causes',
        columns='meas'
    )
    return bias_causes_pivot


def get_bias_causes_data(meas_bias_causes_dict):
    measurements = list(meas_bias_causes_dict.keys())
    bias_causes_pivot = get_bias_causes_pivot(
        meas_bias_causes_dict,
        examined=measurements
    )
    return bias_causes_pivot


def get_bias_causes(input_meas_ids, asns_df):
    """
    This function gets all the bias causes for all measurement IDs in input_meas_ids
    """
    bias_causes_start = time.time()
    # Get bias causes for all input measurements
    all_bias_causes = get_sample_bias_causes(input_meas_ids, asns_df)

    # Create bias causes df
    bias_causes_df = get_bias_df(all_bias_causes)

    bias_causes_end = time.time() - bias_causes_start
    print(f'Bias causes calculation completed in {bias_causes_end:.3f} seconds.')
    return bias_causes_df


def get_dashboard_bias_causes_data(input_meas_ids, bias_causes_df, k):
    """
    This function keeps the top k positive and top k negative largest bias causes for each measurement ID in
    input_meas_ids, as well as for the entire set of input measurement IDs, and returns a pivot table that contains all
    that information.
    """
    # Get top k bias causes for each input measurement
    topk_meas_bias_causes_df = get_meas_topk_causes(input_meas_ids, bias_causes_df, k=k)
    # Get top k aggregated bias causes from the entire set if input measurements
    top_agg_df = get_topk_agg_bias_causes(bias_causes_df, k=k)
    topk_meas_bias_causes_df['total'] = top_agg_df

    bias_causes_pivot_df = get_bias_causes_data(topk_meas_bias_causes_df)
    return bias_causes_pivot_df


def bias_causes_main(input_meas_ids, asns_df, k = 5):
    # Create bias causes df
    bias_causes_df = get_bias_causes(input_meas_ids, asns_df)

    # Get top k positive and negative bias causes for each measurement as well as for all measurements
    bias_causes_pivot_df = get_dashboard_bias_causes_data(input_meas_ids, bias_causes_df, k)


    return bias_causes_df, bias_causes_pivot_df
