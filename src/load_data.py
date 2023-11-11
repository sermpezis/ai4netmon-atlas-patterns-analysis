import requests
import warnings
import time
import os
import pandas as pd
from globals import KEEP_FIELDS, PROBES_DATA_CSV_NAME, BIAS_DIMENSIONS
import agg_data_utils as adu
# AI4NetMon project imports
from project_files.data_aggregation_tools import load_aggregated_dataframe
from project_files.map_probes import map_probes
from project_files.calculate_bias_for_list import calc_bias

# ai4netmon-atlas-patterns\project_files\data_aggregation_tools.py:732
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
# ai4netmon-atlas-patterns\project_files\bias_utils.py:355
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
# C:\Users\<user>\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\arraylike.py:396
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")

# Contains data for each probe. Used to find the ASNs based on the probes of each measurement
if os.path.isfile(PROBES_DATA_CSV_NAME):
    PROBES_DF = pd.read_csv(PROBES_DATA_CSV_NAME)
    print('PROBES_DF loaded successfully.')
else:
    raise FileNotFoundError('probes_data.csv does not exist in the /data/ directory.')

# Get df containing info for each AS
print('Loading aggregated dataframe for ASNs...')
agg_df_load_start = time.time()
ASN_AGG_DF = load_aggregated_dataframe(preprocess=True)
agg_df_load_end = time.time() - agg_df_load_start
print(f'Aggregated dataframe loaded in {agg_df_load_end:.3f} seconds.')


def get_url(input_meas_ids):
    """
    Input:
        - input_meas_ids (list[int]): Input measurement IDs
    Output:
        - uri (str): URI for the GET requests to RIPE Atlas API

    Creates the URI that will be used to get information on the input measurement IDs through a GET call to the
    RIPE Atlas API. The fields we keep for each measurement is specified by the KEEP_FIELDS global variable
    """
    # Define base URI of the 'measurements' RIPE Atlas endpoint
    base_uri = 'https://atlas.ripe.net/api/v2/measurements'

    # Specify measurements IDs we want data for
    input_meas_ids_str = ','.join([str(x) for x in input_meas_ids])
    measurements_url = f'/?id__in={input_meas_ids_str}'

    # Specify which fields of the RIPE Atlas API response we want to keep
    fields = '&fields=' + (',').join(KEEP_FIELDS)

    # Construct final URI
    uri = base_uri + measurements_url + fields
    return uri


def get_input_meas_data(input_meas_uri):
    """
    Input:
        - input_mes_uri (str): URI for the endpoint of RIPE Atlas that gives us information for the input measurements IDs.
    Output:
        - reponse (dict): RIPE Atlas endoint response
    """
    return requests.get(input_meas_uri).json()


def get_input_meas_info(input_meas_ids):
    """
    Input:
        - input_mes_uri (str): URI for the endpoint of RIPE Atlas that gives us information for the input measurements IDs.
    Output:
        - reponse (dict): RIPE Atlas endoint response

        Given the list of measurement IDs we want want to get the data of from the RIPE Atlas API, first the appropriate URI is constructed and then the GET request is made.
    """

    # Create url for RIPE Atlas API to get info on the input measurements
    uri = get_url(input_meas_ids)

    # Make the GET request and get the response
    print(f"Getting data for {len(input_meas_ids)} measurement IDs: {input_meas_ids}")
    request_start = time.time()
    response = get_input_meas_data(uri)
    request_end = time.time() - request_start
    print(f"Time elapsed for the GET call to RIPE Atlas API: {request_end:.3f} seconds.")

    return response


def get_asn_af(meas_af):
    v4, v6 = 0, 0
    if meas_af == 4:
        v4 = 1
    elif meas_af == 6:
        v6 = 1
    else:
        raise ValueError(f'"af" field can be one of 4 or 6, encountered value {meas_af}.')

    return v4, v6


def get_asns_column(row):
    """
    Input:
        - row (pd.Series): A single row of the dataframe containing the data returned from the RIPE Atlas API.
                           Represents a single input measurement.
    Output:
        - meas_asns (list[int]): The ASNs used for the input measurement (row).

    Returns the ASNs that were used for the input measurement. Since a given probe can have different IPv4 and IPv6
    ASNs, we select the ASN that corresponds to the input measurmeent's address family. It can be the case that no ASN
    is found for a probe, if that probe does not have an ASN that corresponds to the measurement's addres family.

    For example, if the input measurement has address family equal to 4 and probe X of the input measurement has no
    IPv4 ASN, then no ASN will be returned for probe X.

    Will return an empty list if the input measurement:
        * has no probes
        * contains probes without ASNs corresponding to the input measurements address family
    """
    meas_id = row['meas_id']
    meas_probes_list = row['probes']

    # If there are no probes in the measurement
    if not meas_probes_list:
        meas_asns = []
        print(f'  {meas_id}: No probes found, continuing to the next measurement.')
    else:
        # Get the address family boolean for the measurement
        meas_af = row['af']
        v4, v6 = get_asn_af(meas_af)
        # Get the ASNs for the current measurement
        meas_asns, not_probes = map_probes(meas_probes_list, v4, v6, PROBES_DF)
        # If no ASNs are found
        if len(meas_asns) == 0:
            print(f'  {meas_id}: No ASNs found, continuing to the next measurement.')
        else:
            # print(f'  {meas_id}: Found ASNs {meas_asns}.')
            print(f'  {meas_id}: Found {len(meas_asns)} ASNs.')

    return meas_asns


def clean_asns_col(df):
    """
    Input:
        - df (pd.DataFrame): Measurements dataset.
    Output:
        - meas_with_asns_df (pd.DataFrame): Subset of measurements with ASNs.
        - meas_without_asns_df (pd.DataFrame): Subset of measurements with no ASNs.
    """
    no_asn_mask = df['num_asns'] == 0
    meas_with_asns_df, meas_without_asns_df = df[~no_asn_mask], df[no_asn_mask]
    return meas_with_asns_df, meas_without_asns_df


def create_meas_df(response_dict):
    """
    Input:
        - reponse_dict (dict): Dictionary containing the response of the GET call to the RIPE Atlas API.
    Output:
        - meas_with_asns_df (pd.DataFrame): Subset of measurements with ASNs.
        - meas_without_asns_df (pd.DataFrame): Subset of measurements with no ASNs.

    Creates the dataframe containing the measurement data from the RIPE Atlas API response. Calculates the ASNs for each measurement (if any), the number of ASNs for each measurement and it also removes measurements with no ASNs in them.
    """

    # Create df with the data from the API
    meas_data_df = (pd.DataFrame(response_dict['results'])
    .rename(
        columns={
            'participant_count': 'num_probes',
            'id': 'meas_id'
        }
    )
    )
    # Keep only probes ids for each probe entry in each measurement
    meas_data_df['probes'] = meas_data_df['probes'].apply(lambda l: [probe['id'] for probe in l])
    # Get the ASNs for each measurement based on the measurement's probes and af
    print("Calculating ASNs for each input measurement...")
    start_asn_calculation = time.time()
    meas_data_df['asns'] = meas_data_df.apply(get_asns_column, axis=1)
    end_asn_calculation = time.time() - start_asn_calculation
    print(f"ASN calculation completed succesfully in {end_asn_calculation:.3f} seconds.")
    # Get the number of ASNs for each measurement as a new column
    meas_data_df['num_asns'] = meas_data_df['asns'].apply(lambda l: len(l))
    # Remove measurements with no ASNs
    print("Removing measurements with no ASNs...")
    meas_data_df, no_asns_df = clean_asns_col(meas_data_df)
    print("Measurements with no ASNs removed.")
    return meas_data_df, no_asns_df


def generate_null_bias_data(meas_id):
    """
    Input:
        - meas_id (int): Input measurement ID
    Output:
        - bias_df (pd.DataFrame): Dataframe containing None for each bias dimension.
    """
    # Generate null data for each bias dimension
    data = {col: None for col in BIAS_DIMENSIONS}
    # Add the measurement ID to the null data
    data['meas_id'] = meas_id
    # Convert to df
    bias_df = pd.DataFrame.from_dict(data, orient='index', ).T  # .set_index('meas_id')
    bias_df['meas_id'] = bias_df['meas_id'].astype('int64')
    bias_df.set_index('meas_id', inplace=True)
    return bias_df


def get_bias_columns(row):
    """
    Input:
        - row (pd.Series): A row from the main dataframe, representing a single measurement.
    Output:
        - meas_bias_df (pd.DataFrame): Dataframe containing the bias data for the input measurement.

    For the input row (measurement), it calculates the bias values for each bias dimension based on the measurement's
    ASNs. If a measurement contains no ASNs, then the bias values for that measurements are set to None.
    """
    meas_id = row['meas_id']
    meas_asns = row['asns']
    if len(meas_asns) == 0:
        print(f'  {meas_id}: This measurement contains no ASNs. Bias values will be set to NaN.')
        meas_bias_df = generate_null_bias_data(meas_id)
    else:
        try:
            meas_bias_df = calc_bias(meas_asns, 'probes', 0, ASN_AGG_DF).transpose().rename(
                index={'bias': meas_id}).rename_axis('meas_id')
            # Keep only the bias dimensions we have defined
            meas_bias_df = meas_bias_df[BIAS_DIMENSIONS]
        except KeyError:
            # Can happen if some asn id in meas_asns does not exist in ASN_AGG_DF. Skip that measurement.
            print(
                f'  {meas_id}: This measurement contains ASNs that do not exist in ASN_AGG_DF. Bias values for this '
                f'measurement will be set to NaN.')
            meas_bias_df = generate_null_bias_data(meas_id)
    return meas_bias_df


def get_bias_df(meas_data_df):
    """
    Input:
        - meas_data_df (pd.DataFrame): Measurements' dataset.
    Output:
        - all_meas_bias_df (pd.DataFrame): Measurements' dataset with the bias values for each bias dimension.

    Calculates the bias values for each bias dimension, for each measurement in the input dataset.
    """
    print('Calculating the bias values for each measurement...')
    total_bias_calculation_start = time.time()

    bias_dfs = []
    # For each measurement in out dataset
    for idx, row in meas_data_df.iterrows():
        # Get dataframe containing the bias values for each bias dimension
        meas_bias_calculation_start = time.time()
        meas_bias_df = get_bias_columns(row)
        meas_bias_calculation_end = time.time() - meas_bias_calculation_start
        print(f"  {row['meas_id']}: Bias values calculated in {meas_bias_calculation_end:.3f} seconds.")
        bias_dfs.append(meas_bias_df)
    # Combine all the dataframes containing the bias values of each measurement into one
    all_meas_bias_df = pd.concat(bias_dfs)
    total_bias_calculation_end = time.time() - total_bias_calculation_start
    print(
        f"Bias calculation for all {meas_data_df.shape[0]} measurements completed in {total_bias_calculation_end:.3f} seconds.")
    return all_meas_bias_df


def extract_list_column(df, col_name):
    """
    Input:
        - df (pd.DataFrame): Dataframe containing the bias values for each measurement.
        - col_name (str): String specifying the column of df that we want to extract. Should correspond to a column
                          containing lists.
    Output:
        - df (pd.DataFrame): Same as input df but with column col_name removed.
        - col_list_df (pd.DataFrame): Dataframe containing the data of col_name.

    Extracts the col_name list-type column into a new df by exploding it together with the measurement id, returns it
    and also removes the column from the initial dataframe.
    """
    # Extract list colunmn as a new dataframe
    col_list_df = df[col_name].explode().reset_index()
    # Drop list column from initial dataframe
    df = df.drop([col_name], axis = 1)
    print(f' Removed column {col_name} to a new df since it was a list column.')
    return df, col_list_df


def extract_list_columns(df):
    """
    Input:
        - df (pd.DataFrame): Dataframe containing the bias values for each measurement.
    Output:
        - no_probes_asns_df (pd.DataFrame): Same as input df but with the probes and asns columns removed.
        - probes_df (pd.DataFrame): Dataframe containing as rows the ASNs and the measurement IDs in which they appear.
        - asns_df (pd.DataFrame): Dataframe containing as rows the probe IDs and the measurement IDs in which they
                                  appear.
    """
    df_ = df.copy().set_index('meas_id')
    # Extract probes into a new dataframe
    df, probes_df = extract_list_column(df_, 'probes')
    # Extract asns into a new dataframe
    df, asns_df = extract_list_column(df, 'asns')
    # Same as input df but without the probes and asns columns
    no_probes_asns_df = df.reset_index()

    probes_df = probes_df.rename(columns = {'probes': 'probe_id'}).astype(int)
    asns_df = asns_df.rename(columns = {'asns': 'asn'}).astype(int)

    return no_probes_asns_df, probes_df, asns_df


def get_final_df(meas_data_df, bias_df):
    """
    Input:
        - meas_data_df (pd.DataFrame): Dataframe containing the RIPE Atlas data for each input measurement.
        - bias_df (pd.DataFrame): Dataframe containing the bias values in each bias dimension for each input
                                  measurement.
    Output:
        - final_df (pd.DataFrame): Combination of the input dataframes into one.
    """
    final_df = meas_data_df.set_index('meas_id').join(bias_df)
    return final_df


def load_measurement_data(input_meas_ids):
    """
    Input:
        - input_meas_ids (list[int]): Input measurement IDs
    Output:
        - df (pd.DataFrame): Final dataframe containing each input measurement as a row, and as columns various metadata
                             about each measurement (specified by the KEEP_FIELDS global variable) as well as the bias
                             values for each bias dimension.
        - probes_df (pd.DataFrame): Dataframe containing as rows the ASNs and the measurement IDs in which they appear.
        - asns_df (pd.DataFrame): Dataframe containing as rows the probe IDs and the measurement IDs in which they
                                  appear.
    """
    # Make the GET request and get the response
    response = get_input_meas_info(input_meas_ids)
    # Convert response to dataframe
    meas_data_df, no_asns_df = create_meas_df(response)
    # Calculate bias for each measurement for all bias dimensions
    bias_df = get_bias_df(meas_data_df)
    # Explode probes and asns columns to get a df containing probe/asn id and measurement id
    meas_data_df, probes_df, asns_df = extract_list_columns(meas_data_df)
    df = get_final_df(meas_data_df, bias_df)
    return df, probes_df, asns_df


def load_main(input_meas_ids):
    if len(input_meas_ids) == 0:
        raise ValueError('No measurements given.')
    else:
        load_data_start = time.time()
        df, probes_df, asns_df = load_measurement_data(input_meas_ids)
        load_data_end = time.time() - load_data_start
        print(f"Data loading completed successfully in {load_data_end:.3f} seconds.")
        return df, probes_df, asns_df


def string_to_txt(s, file_path):
    with open(file_path, 'w') as f:
        f.write(s)


def export_all_asns(asns_df):
    all_asns_filepath = '../data/all_asns.txt'
    # Get list of all ASNs
    all_asns = adu.get_all_asns(asns_df)
    # Serialize into a string
    all_asns_str = ','.join([str(asn) for asn in all_asns])
    # Export the ASNs string to the specified file
    string_to_txt(all_asns_str, all_asns_filepath)
    print('All ASNs exported.')


def export_asns_per_meas(asns_df):
    # For each measurement ID in the input measurement IDs
    for meas_id in asns_df['meas_id'].unique():
        # Create the filename for the exported ASNs
        meas_asns_filepath = f'../data/{meas_id}_asns.txt'
        # Get current measurement's ASNs in a list
        meas_asns = adu.get_meas_asns(meas_id, asns_df)
        # If there are ASNs found
        if len(meas_asns) > 0:
            # Serialize the ASNs list into a string
            meas_asns_str = ','.join([str(asn) for asn in meas_asns])
        else:
            meas_asns_str = 'No ASNs found for this measurement ID.'
        # Export the ASNS string to the specified file
        string_to_txt(meas_asns_str, meas_asns_filepath)
        print(f"{meas_id}: Measurement's ASNS exported.")


if __name__ == "__main__":
    # Simple test case
    input_meas_ids = [1018338, 1004340, 1017820, 1019139, 1017005, 1019222, 1007976, 1035173, 1010732, 1036065]
    df, probes_df, asns_df = load_main(input_meas_ids)
    print(df)
    print('-'*200)
    print(probes_df.head())
    print('-' * 200)
    print(asns_df.head())
