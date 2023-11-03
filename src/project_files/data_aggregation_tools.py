import pandas as pd
import numpy as np
import json
# import pycountry_convert as pc
# from ..project_files.data_collectors import get_ripe_ris_data

#
# FILES_LOCATION = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/misc/'
# PATH_AS_RANK = FILES_LOCATION+'ASrank.csv'
# PATH_PERSONAL = FILES_LOCATION+'perso.txt'
# PATH_PEERINGDB = FILES_LOCATION+'peeringdb_2_dump_2021_07_01.json'
# AS_HEGEMONY_PATH = FILES_LOCATION+'AS_hegemony.csv'
# ALL_ATLAS_PROBES = FILES_LOCATION+'RIPE_Atlas_probes.json'
# ROUTEVIEWS_PEERS = FILES_LOCATION+'RouteViews_20220402.txt'
# AS_RELATIONSHIPS = FILES_LOCATION+'AS_relationships_20210701.as-rel2.txt'
# ASDB_PATH = 'https://asdb.stanford.edu/data/ases.csv'
# ORIGIN_PATH = FILES_LOCATION + 'origins.csv'
# TOP_PATH = FILES_LOCATION + 'top.csv'
# AGGREGATE_DATA_FNAME = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/aggregate_data/asn_aggregate_data_20230531.csv'
# ALL_DATASETS = ['AS_rank', 'personal', 'PeeringDB', 'AS_hegemony', 'Atlas_probes', 'RIPE_RIS', 'RouteViews', 'AS_relationships', 'top', 'origins', 'asdb_1', 'asdb_2']
#
#
# def cc2cont(country_code):
#     '''
#     Receives a country code ISO2 (e.g., 'US') and returns the corresponding continent name (e.g., 'North America').
#     Exceptions:
#         - if 'EU' is given as country code (it happened in data), then it is treated as the continent code
#         - if the country code is not found, then a None value is returned
#     :param  country_code:   (str) ISO2 country code
#     :return:    (str) continent name of the given country(-ies)
#     '''
#     if country_code in ['EU']:
#         continent_code = country_code
#     else:
#         try:
#             continent_code = pc.country_alpha2_to_continent_code(country_code)
#         except KeyError:
#             return None
#     continent_name = pc.convert_continent_code_to_continent_name(continent_code)
#     return continent_name
#
#
# def get_continent(country_code):
#     '''
#     Receives a series of country codes ISO2 (e.g., 'US') and returns the corresponding continent names (e.g., 'North America').
#     For NaN or None elements, it returns a None value
#     :param  country_code:   (pandas Series) ISO2 country codes
#     :return:    (list of str) continent names of the given countries
#     '''
#     continent_name = []
#     for cc in country_code.tolist():
#         if pd.isna(cc):
#             continent_name.append(None)
#         else:
#             continent_name.append( cc2cont(cc) )
#     return continent_name
#
# def create_df_from_AS_relationships():
#     """
#     Loads the CAIDA AS-relationships datasets from the source file. Returns a dataframe with index the ASN
#     and columns features derived from the graph; appends in the column names the prefix "AS_rel_".
#     The returned features are:
#         - "degree":    a column with the degree (i.e., #neighbors) of each AS
#
#     :return: A dataframe with index the ASN
#     """
#     G = create_graph_from_AS_relationships(AS_RELATIONSHIPS)
#     df = pd.DataFrame(G.degree(), columns=['asn','AS_rel_degree'])
#     df = df.set_index('asn')
#
#     return df
#
# def create_df_from_RouteViews():
#     """
#     Collects the list of RouteViews peers, and returns a dataframe with RouteViews peers ASNs
#     :return: A dataframe with index the ASN
#     """
#     df = pd.read_csv(ROUTEVIEWS_PEERS, delimiter="|")
#     df = df[['AS_NUMBER']]
#
#     df = df.drop_duplicates()
#     df['is_routeviews_peer'] = 1
#     df = df.set_index('AS_NUMBER')
#
#     return df
#
# def create_df_from_RIPE_RIS():
#     """
#     Collects the list of RIPE RIS peers, and returns a dataframe with the v4 and v6 RIS peers ASNs.
#     :return: A dataframe with index the ASN
#     """
#     ris_peer_ip2asn, _ = get_ripe_ris_data()
#     unique_asns = set(ris_peer_ip2asn.values())
#     unique_asns_v4 = set([asn for ip,asn in ris_peer_ip2asn.items() if ':' not in ip])
#     unique_asns_v6 = set([asn for ip,asn in ris_peer_ip2asn.items() if ':' in ip])
#
#     df = pd.DataFrame(columns=['is_ris_peer_v4', 'is_ris_peer_v6'], index=unique_asns)
#     df.loc[unique_asns_v4, 'is_ris_peer_v4'] = 1
#     df.loc[unique_asns_v6, 'is_ris_peer_v6'] = 1
#     df.index.name = 'asn'
#
#     return df
#
# def create_df_from_Atlas_probes():
#     """
#     Loads the list of RIPE Atlas probes, and returns a dataframe with the number of v4 and v6 probes per ASN (only for ASNs that have at least one probe).
#     :return: A dataframe with index the ASN
#     """
#     data = pd.read_json(ALL_ATLAS_PROBES, lines=True)
#     data = data[(data['status'] == 'Connected')]
#     s4 = data['asn_v4'].value_counts()
#     s6 = data['asn_v6'].value_counts()
#     df = pd.concat([s4, s6], axis=1)
#     df.index.name = 'asn'
#     df = df.rename(columns={'asn_v4': 'nb_atlas_probes_v4', 'asn_v6': 'nb_atlas_probes_v6'})
#
#     return df
#
#
# def create_df_from_AS_rank():
#     """
#     Loads the CAIDA AS-rank dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "AS_rank_".
#     :return: A dataframe with index the ASN
#     """
#     data = pd.read_csv(PATH_AS_RANK, sep=",")
#     new_columns = ['AS_rank_' + str(i) for i in data.columns]
#     data = data.set_axis(new_columns, axis='columns', inplace=False)
#     data.loc[(data['AS_rank_longitude'] == 0) & (data['AS_rank_latitude'] == 0), ['AS_rank_longitude',
#                                                                                   'AS_rank_latitude']] = None
#     data['AS_rank_continent'] = get_continent(data['AS_rank_iso'])
#     data['AS_rank_source'].replace('JPNIC','APNIC',inplace=True)    # fix to CAIDA's data: replace JPNIC (which is NIR) with APNIC (which is RIR)
#     data = data.set_index('AS_rank_asn')
#
#     return data
#
#
# def create_df_from_AS_hegemony():
#     """
#     Loads the AS hegemony dataset from the source file. Returns a dataframe with index the ASN, and a single column with the AS hegemony value of the AS
#     :return: A dataframe with index the ASN
#     """
#     data = pd.read_csv(AS_HEGEMONY_PATH, sep=",")
#     data = data.rename(columns={'hege': 'AS_hegemony'})
#     data = data.set_index('asn')
#
#     return data
#
#
# def create_df_from_personal():
#     """
#     Loads the bgp.tools personal AS dataset from the source file. Creates a dataframe with index the ASN of
#     the ASes that are personal use ASes; the dataframe has only one column with 1 for all rows
#     :return: A dataframe with index the ASN
#     """
#     data = pd.read_csv(PATH_PERSONAL, header=None)
#     data.columns = ['asn']
#     # keep only the digits of the ASNs
#     data['asn'] = data['asn'].apply(lambda x: int(x[2:]))
#     data['is_personal_AS'] = 1
#     data = data.set_index('asn')
#
#     return data
#
# def pdb_info_traffic_to_float(ds):
#     '''
#     Transforms the values of the PDB info_traffic field from str to float values.
#     The float values are measured in Mbps and taking the lower limit, e.g., the str '100-200Gbps' would be the float 100000
#     :param  ds: (pandas.Series) the Series that corresponds to the column 'info_traffic' of the dataframe (dtype: object, i.e., str)
#     :return:    (pandas.Series) the given Series tranformed to float values
#     '''
#     data = ds.copy()
#     data.replace('0-20Mbps', '1-20Mbps',inplace=True)
#     data.replace('100+Tbps', '100-Tbps',inplace=True)
#     data.replace('', np.nan,inplace=True)
#     traffic_str = [t for t in data.unique() if (isinstance(t,str)) and (len(t)>0)]
#     str2Mbps = {'Mbps':1, 'Gbps':1000, 'Tbps':1000000}
#     traffic_dict = {t: int(t.split('-')[0]) * str2Mbps[t[-4:]] for t in traffic_str}
#     for tr_str,tr_float in traffic_dict.items():
#         data.replace(tr_str,tr_float,inplace=True)
#     return data
#
# def create_df_from_PeeringDB():
#     """
#     Loads the PeeringDB dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "peeringDB_". The dataframe which contains only the features in the keep_keys list
#     :return: A dataframe with index the ASN
#     """
#     df = pd.read_json(PATH_PEERINGDB)
#     data = []
#     keep_keys = ['asn', 'info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
#                  'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
#     for row in df.net['data']:
#         net_row = [row.get(key) for key in keep_keys]
#         data.append(net_row)
#     df = pd.DataFrame(data, columns=keep_keys)
#     df['info_traffic'] = pdb_info_traffic_to_float(df['info_traffic'])
#     new_columns = ['peeringDB_' + str(i) for i in df.columns]
#     df = df.set_axis(new_columns, axis='columns', inplace=False)
#     df['is_in_peeringDB'] = 1
#     df = df.set_index('peeringDB_asn')
#
#     return df
#
# def create_dataframe_from_asdb(way):
#     """
#     Function that reads the csv files from Stanford's page to pandas Dataframe, keeps the first 3 columns of
#     and merges the two columns of categories into one, based on two methods, as a tuple and as a single string
#     separated by underscore, and returns a dataframe with index the ASN.
#     :param way: the way that the categories are finally stored
#     :return: the dataframe to be added to the final dataframe
#     """
#     data = pd.read_csv(ASDB_PATH)
#     data = data[['ASN', 'Category 1 - Layer 1', 'Category 1 - Layer 2']]
#     data['ASN'] = data['ASN'].str.split('AS', n=1).str.get(-1)
#     if way == 1:
#         # way 1 - merge the two categorical columns into one column that is now a tuple
#         data = data.drop(['Category 1 - Layer 2'], axis=1)
#         data = data.rename(columns={'ASN': 'asn'})
#         data = data.rename(columns={'Category 1 - Layer 1': 'asdb_Category 1 - Layer 1'})
#         data = data.set_index('asn')
#         return data
#     else:
#         # way 2 - merge the two categorical columns into one column that contains the two strings seperated by _
#         data["Categroy1 - Layer 1 and 2"] = data["Category 1 - Layer 1"] + "_" + data["Category 1 - Layer 2"]
#         data = data.drop(['Category 1 - Layer 1', 'Category 1 - Layer 2'], axis=1)
#
#         data = data.rename(columns={'ASN': 'asn'})
#         data = data.rename(columns={"Categroy1 - Layer 1 and 2": "asdb_Categroy1 - Layer 1 and 2"})
#
#         data = data.set_index('asn')
#         return data
#
#
# def create_df_from_cti_top():
#
#     data = pd.read_csv(TOP_PATH)
#     data = data.rename(columns={'ASN': 'asn'})
#     data = data.rename(columns={'prefix': 'top_prefix'})
#     data = data.set_index('asn')
#     data = data.drop(['Unnamed: 0'], axis=1)
#     return data
#
#
# def create_df_from_cti_origins():
#
#     data = pd.read_csv(ORIGIN_PATH)
#     data = data.rename(columns={'ASN': 'asn'})
#     data = data.rename(columns={'%country': 'origins_%country'})
#     data = data.set_index('asn')
#     data = data.drop(['Unnamed: 0'], axis=1)
#     return data
#
#
#
# def create_df_from(dataset):
#     """
#     Selects a method, based on the given dataset name, and creates the corresponding dataframe.
#     When adding a new method, take care to have as index the ASN and the column names to be of the format "dataset_name_"+"column_name" (e.g., the column "X" from the dataset "setA", should be "setA_X")
#     :param dataset: (type = string) name of the dataset to be loaded
#     :return: A dataframe with indexes the ASNs and columns the features loaded from the given dataset
#     """
#     if dataset == 'AS_rank':
#         data = create_df_from_AS_rank()
#     elif dataset == 'personal':
#         data = create_df_from_personal()
#     elif dataset == 'PeeringDB':
#         data = create_df_from_PeeringDB()
#     elif dataset == 'AS_hegemony':
#         data = create_df_from_AS_hegemony()
#     elif dataset == 'Atlas_probes':
#         data = create_df_from_Atlas_probes()
#     elif dataset == 'RIPE_RIS':
#         data = create_df_from_RIPE_RIS()
#     elif dataset == 'RouteViews':
#         data = create_df_from_RouteViews()
#     elif dataset == 'AS_relationships':
#         data = create_df_from_AS_relationships()
#     elif dataset == 'origins':
#         data = create_df_from_cti_origins()
#     elif dataset == 'top':
#         data = create_df_from_cti_top()
#     elif dataset == 'asdb_1':
#         data = create_dataframe_from_asdb(1)
#     elif dataset == 'asdb_2':
#         data = create_dataframe_from_asdb(2)
#     else:
#         raise Exception('Not defined dataset')
#     return data
#
#
# def create_dataframe_from_multiple_datasets(list_of_datasets=None):
#     """
#     Creates a dataframe for each given dataset, and concatenates all the dataframes in a common dataframe. The final/returned dataframe has the ASN as the index, and as columns all the columns from all datasets. It fills with NaN non existing values.
#     :param list_of_datasets:    (list) a list of str, where each string corresponds to a dataset to be loaded; If not set, loads all available datasets
#     :return:                    A dataframe with indexes the ASNs and columns the features loaded from each given dataset
#     """
#     if list_of_datasets is None:
#         # select all datasets
#         list_of_datasets = ALL_DATASETS
#     data = pd.DataFrame()
#     list_of_dataframes = []
#     for i in list_of_datasets:
#         list_of_dataframes.append(create_df_from(i))
#     final_df = pd.concat(list_of_dataframes, axis=1)
#     final_df.index.name = 'ASN'
#     return final_df
#
#
# def load_aggregated_dataframe(preprocess=False):
#     '''
#     Loads the aggregated dataframe
#     :param      preprocess:     (bollean) [optional] if set to True, it does some processing steps in the data (e.g., missing values)
#     :return:    (pandas.DataFrame) A dataframe with indexes the ASNs and columns the features of the different datasets
#     '''
#     df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
#     if preprocess:
#         df['is_personal_AS'].fillna(0, inplace=True)
#         df['is_in_peeringDB'].fillna(0, inplace=True)
#         # pdb_columns = [c for c in df.columns if c.startswith('peeringDB')]
#         # pdb_columns_categorical = [c for c in df.columns if c.startswith('peeringDB') and df[c].dtype=='O']
#         # df.loc[df[pdb_columns].isna().all(axis=1), pdb_columns_categorical] = 'Not in PDB'
#     return df
# import pandas as pd
# import numpy as np

import json
from datetime import date, datetime, timedelta
import time
import calendar
import warnings
import os
import pycountry_convert as pc
# from Analysis.aggregate_data import data_collectors as dc
from .graph_methods import create_graph_from_AS_relationships

# from Analysis.aggregate_data import graph_methods as gm
from collections import defaultdict
import urllib.request

# FILES_LOCATION = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/misc/'
# FILES_LOCATION = '/home/sofia/PycharmProjects/ai4netmon/data/misc/'
FILES_LOCATION = '../app/misc/'
today = date.today()
DATE = str(today).replace('-', '')
print(DATE)

# DATE = '20230117'

def get_path(date):
    PATH_AS_RANK = FILES_LOCATION + 'ASrank_{}.csv'.format(date)
    PATH_PERSONAL = FILES_LOCATION + 'bgp_perso_{}.txt'.format(date)
    PATH_PEERINGDB = FILES_LOCATION + 'PeeringDB_{}.json'.format(date)
    AS_HEGEMONY_PATH = FILES_LOCATION + 'AS_hegemony_{}.csv'.format(date)
    ALL_ATLAS_PROBES = FILES_LOCATION + 'RIPE_Atlas_probes_{}.json'.format(date)
    ALL_RIS_PEERS = FILES_LOCATION + 'RIPE_RIS_collectors_{}.json'.format(date)
    ROUTEVIEWS_PEERS = FILES_LOCATION + 'RouteViews_{}.txt'.format(date)
    AS_RELATIONSHIPS = FILES_LOCATION + 'AS_relationships_{}.txt'.format(date)
    ASDB_PATH = FILES_LOCATION + 'ASDB_{}.csv'.format(date)
    ORIGIN_PATH = FILES_LOCATION + 'CTI_origin_{}.csv'.format(date)
    TOP_PATH = FILES_LOCATION + 'CTI_top_{}.csv'.format(date)
    BGPTOOLS_PATH = FILES_LOCATION + 'bgptools_{}.csv'.format(date)
    BGPTOOLS_v4_PATH = FILES_LOCATION + 'bgptools_v4_{}.csv'.format(date)
    BGPTOOLS_v6_PATH = FILES_LOCATION + 'bgptools_v6_{}.csv'.format(date)
    return PATH_AS_RANK, PATH_PERSONAL, PATH_PEERINGDB, AS_HEGEMONY_PATH, ALL_ATLAS_PROBES, ALL_RIS_PEERS, ROUTEVIEWS_PEERS, AS_RELATIONSHIPS, ASDB_PATH, ORIGIN_PATH, TOP_PATH, BGPTOOLS_PATH, BGPTOOLS_v4_PATH, BGPTOOLS_v6_PATH


FILES_LOCATION_AGGREGATE = '../../data/aggregate_data/'
# AGGREGATE_DATA_FNAME = '/home/sofia/PycharmProjects/ai4netmon/data/aggregate_data/asn_aggregate_data_{}.csv'.format(DATE)
AGGREGATE_DATA_FNAME = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/aggregate_data/asn_aggregate_data.csv'

ALL_DATASETS = ['ASrank', 'bgp_perso', 'PeeringDB', 'AS_hegemony', 'RIPE_Atlas_probes', 'RIPE_RIS_collectors',
                'RouteViews', 'AS_relationships', 'CTI_top', 'CTI_origin', 'asdb_1', 'asdb_2', 'bgptools']


def cc2cont(country_code):
    '''
    Receives a country code ISO2 (e.g., 'US') and returns the corresponding continent name (e.g., 'North America').
    Exceptions:
        - if 'EU' is given as country code (it happened in data), then it is treated as the continent code
        - if the country code is not found, then a None value is returned
    :param  country_code:   (str) ISO2 country code
    :return:    (str) continent name of the given country(-ies)
    '''
    if country_code in ['EU']:
        continent_code = country_code
    else:
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
        except KeyError:
            return None
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name


def get_continent(country_code):
    '''
    Receives a series of country codes ISO2 (e.g., 'US') and returns the corresponding continent names (e.g., 'North America').
    For NaN or None elements, it returns a None value
    :param  country_code:   (pandas Series) ISO2 country codes
    :return:    (list of str) continent names of the given countries
    '''
    continent_name = []
    for cc in country_code.tolist():
        if pd.isna(cc):
            continent_name.append(None)
        else:
            continent_name.append(cc2cont(cc))
    return continent_name


def create_df_from_bgtools(BGPTOOLS_PATH):
    df = pd.read_csv(BGPTOOLS_PATH)
    df = df.set_index('ASN')
    asnList = list(df.index.values)
    return df, asnList


def create_df_from_AS_relationships(AS_RELATIONSHIPS):
    """
    Loads the CAIDA AS-relationships datasets from the source file. Returns a dataframe with index the ASN
    and columns features derived from the graph; appends in the column names the prefix "AS_rel_".
    The returned features are:
        - "degree":    a column with the degree (i.e., #neighbors) of each AS

    :return: A dataframe with index the ASN
    """
    G = create_graph_from_AS_relationships(AS_RELATIONSHIPS)
    df = pd.DataFrame(G.degree(), columns=['asn', 'AS_rel_degree'])
    df = df.set_index('asn')
    asnList = list(df.index.values)

    return df, asnList


def create_df_from_RouteViews(ROUTEVIEWS_PEERS):
    """
    Collects the list of RouteViews peers, and returns a dataframe with RouteViews peers ASNs
    :return: A dataframe with index the ASN
    """
    df = pd.read_csv(ROUTEVIEWS_PEERS, delimiter="|")
    df = df[['AS_NUMBER']]

    df = df.drop_duplicates()
    df['is_routeviews_peer'] = 1
    df = df.set_index('AS_NUMBER')
    asnList = list(df.index.values)

    return df, asnList


def create_df_from_RIPE_RIS(ALL_RIS_PEERS):
    """
    Collects the list of RIPE RIS peers, and returns a dataframe with the v4 and v6 RIS peers ASNs.
    :return: A dataframe with index the ASN
    """
    if FILES_LOCATION.startswith('http'):
        with urllib.request.urlopen(ALL_RIS_PEERS) as url:
            ris_dict = json.loads(url.read().decode())
    else:
        with open(ALL_RIS_PEERS, 'r') as f:
            ris_dict = json.load(f)
    unique_asns = set()
    unique_asns_v4 = set()
    unique_asns_v6 = set()
    for rrc, peers in ris_dict.items():
        unique_asns.update([int(p['asn']) for p in peers])
        unique_asns_v4.update([int(p['asn']) for p in peers if ':' not in p['ip']])
        unique_asns_v6.update([int(p['asn']) for p in peers if ':' in p['ip']])

    df = pd.DataFrame(columns=['is_ris_peer_v4', 'is_ris_peer_v6'], index=list(unique_asns))
    df.loc[unique_asns_v4, 'is_ris_peer_v4'] = 1
    df.loc[unique_asns_v6, 'is_ris_peer_v6'] = 1
    df.index.name = 'asn'
    asnList = list(df.index.values)

    return df, asnList


def create_df_from_Atlas_probes(ALL_ATLAS_PROBES):
    """
    Loads the list of RIPE Atlas probes, and returns a dataframe with the number of v4 and v6 probes per ASN (only for ASNs that have at least one probe).
    :return: A dataframe with index the ASN
    """
    if FILES_LOCATION.startswith('http'):
        with urllib.request.urlopen(ALL_ATLAS_PROBES) as url:
            probes = json.loads(url.read().decode())
    else:
        with open(ALL_ATLAS_PROBES, 'r') as f:
            probes = json.load(f)

    data = defaultdict(lambda: {'nb_atlas_probes_v4': 0, 'nb_atlas_probes_v6': 0, 'nb_atlas_anchors': 0})
    for prb in probes:
        if prb['status']['name'] == 'Connected':
            asn_v4 = prb['asn_v4']
            asn_v6 = prb['asn_v6']
            if asn_v4 is not None:
                data[asn_v4]['nb_atlas_probes_v4'] += 1
                if prb['is_anchor']:
                    data[asn_v4]['nb_atlas_anchors'] += 1
            if asn_v6 is not None:
                data[asn_v6]['nb_atlas_probes_v6'] += 1
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'asn'
    asnList = list(df.index.values)

    return df, asnList


def create_df_from_AS_rank(PATH_AS_RANK):
    """
    Loads the CAIDA AS-rank dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "AS_rank_".
    :return: A dataframe with index the ASN
    """
    data = pd.read_csv(PATH_AS_RANK, sep=",")
    new_columns = ['AS_rank_' + str(i) for i in data.columns]
    data = data.set_axis(new_columns, axis='columns', inplace=False)
    data.loc[(data['AS_rank_longitude'] == 0) & (data['AS_rank_latitude'] == 0), ['AS_rank_longitude',
                                                                                  'AS_rank_latitude']] = None
    data['AS_rank_continent'] = get_continent(data['AS_rank_iso'])
    data['AS_rank_source'].replace('JPNIC', 'APNIC',
                                   inplace=True)  # fix to CAIDA's data: replace JPNIC (which is NIR) with APNIC (which is RIR)
    data = data.set_index('AS_rank_asn')
    asnList = list(data.index.values)

    return data, asnList


def create_df_from_AS_hegemony(AS_HEGEMONY_PATH):
    """
    Loads the AS hegemony dataset from the source file. Returns a dataframe with index the ASN, and a single column with the AS hegemony value of the AS
    :return: A dataframe with index the ASN
    """
    data = pd.read_csv(AS_HEGEMONY_PATH, sep=",")
    data = data.rename(columns={'hege': 'AS_hegemony'})
    data = data.set_index('asn')
    asnList = list(data.index.values)

    return data, asnList


def create_df_from_personal(PATH_PERSONAL):
    """
    Loads the bgp.tools personal AS dataset from the source file. Creates a dataframe with index the ASN of
    the ASes that are personal use ASes; the dataframe has only one column with 1 for all rows
    :return: A dataframe with index the ASN
    """
    data = pd.read_csv(PATH_PERSONAL, header=None)
    data.columns = ['asn']
    # keep only the digits of the ASNs
    data['asn'] = data['asn'].apply(lambda x: int(x[2:]))
    data['is_personal_AS'] = 1
    data = data.set_index('asn')
    asnList = list(data.index.values)

    return data, asnList


def pdb_info_traffic_to_float(ds):
    '''
    Transforms the values of the PDB info_traffic field from str to float values.
    The float values are measured in Mbps and taking the lower limit, e.g., the str '100-200Gbps' would be the float 100000
    :param  ds: (pandas.Series) the Series that corresponds to the column 'info_traffic' of the dataframe (dtype: object, i.e., str)
    :return:    (pandas.Series) the given Series tranformed to float values
    '''
    data = ds.copy()
    data.replace('0-20Mbps', '1-20Mbps', inplace=True)
    data.replace('100+Tbps', '100-Tbps', inplace=True)
    data.replace('', np.nan, inplace=True)
    traffic_str = [t for t in data.unique() if (isinstance(t, str)) and (len(t) > 0)]
    str2Mbps = {'Mbps': 1, 'Gbps': 1000, 'Tbps': 1000000}
    traffic_dict = {t: int(t.split('-')[0]) * str2Mbps[t[-4:]] for t in traffic_str}
    for tr_str, tr_float in traffic_dict.items():
        data.replace(tr_str, tr_float, inplace=True)
    return data


def create_df_from_PeeringDB(PATH_PEERINGDB):
    """
    Loads the PeeringDB dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "peeringDB_". The dataframe which contains only the features in the keep_keys list
    :return: A dataframe with index the ASN
    """
    df = pd.read_json(PATH_PEERINGDB)
    data = []
    keep_keys = ['asn', 'info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
                 'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
    for row in df.net['data']:
        net_row = [row.get(key) for key in keep_keys]
        data.append(net_row)
    df = pd.DataFrame(data, columns=keep_keys)
    df['info_traffic'] = pdb_info_traffic_to_float(df['info_traffic'])
    new_columns = ['peeringDB_' + str(i) for i in df.columns]
    df = df.set_axis(new_columns, axis='columns', inplace=False)
    df['is_in_peeringDB'] = 1
    df = df.set_index('peeringDB_asn')
    asnList = list(df.index.values)

    return df, asnList


def create_dataframe_from_asdb(way, ASDB_PATH):
    """
    Function that reads the csv files from Stanford's page to pandas Dataframe, keeps the first 3 columns of
    and merges the two columns of categories into one, based on two methods, as a tuple and as a single string
    separated by underscore, and returns a dataframe with index the ASN.
    :param way: the way that the categories are finally stored
    :return: the dataframe to be added to the final dataframe
    """
    data = pd.read_csv(ASDB_PATH)
    data = data[['ASN', 'Category 1 - Layer 1', 'Category 1 - Layer 2']]
    # data['ASN'] = data['ASN'].str.split('AS', n=1).str.get(-1)
    data['ASN'] = data['ASN'].apply(lambda x: int(x.split('AS')[-1]))
    if way == 1:
        # way 1 - merge the two categorical columns into one column that is now a tuple
        data = data.drop(['Category 1 - Layer 2'], axis=1)
        data = data.rename(columns={'ASN': 'asn'})
        data = data.rename(columns={'Category 1 - Layer 1': 'ASDB_C1L1'})
        data = data.set_index('asn')
        asnList = list(data.index.values)

        return data, asnList
    elif way == 2:
        # way 2 - merge the two categorical columns into one column that contains the two strings seperated by _
        data["Categroy1 - Layer 1 and 2"] = data["Category 1 - Layer 1"] + "_" + data["Category 1 - Layer 2"]
        data = data.drop(['Category 1 - Layer 1', 'Category 1 - Layer 2'], axis=1)

        data = data.rename(columns={'ASN': 'asn'})
        data = data.rename(columns={"Categroy1 - Layer 1 and 2": "ASDB_C1L2"})

        data = data.set_index('asn')
        asnList = list(data.index.values)

        return data, asnList

    else:
        data = data.rename(columns={'ASN': 'asn'})
        data = data.set_index('asn')
        asnList = list(data.index.values)
        return 0, asnList


def create_df_from_cti_top(TOP_PATH):
    data = pd.read_csv(TOP_PATH)
    data = data.rename(columns={'ASN': 'asn'})
    data = data.rename(columns={'prefix': 'cti_top'})
    data = data.set_index('asn')
    data = data.drop(['Unnamed: 0'], axis=1)
    asnList = list(data.index.values)

    return data, asnList


def create_df_from_cti_origins(ORIGIN_PATH):
    data = pd.read_csv(ORIGIN_PATH)
    data = data.rename(columns={'ASN': 'asn'})
    data = data.rename(columns={'%country': 'cti_origin'})
    data = data.set_index('asn')
    data = data.drop(['Unnamed: 0'], axis=1)
    asnList = list(data.index.values)

    return data, asnList


def create_df_from(dataset, date):
    """
    Selects a method, based on the given dataset name, and creates the corresponding dataframe.
    When adding a new method, take care to have as index the ASN and the column names to be of the format "dataset_name_"+"column_name" (e.g., the column "X" from the dataset "setA", should be "setA_X")
    :param dataset: (type = string) name of the dataset to be loaded
    :return: A dataframe with indexes the ASNs and columns the features loaded from the given dataset
    """
    PATH_AS_RANK, PATH_PERSONAL, PATH_PEERINGDB, AS_HEGEMONY_PATH, ALL_ATLAS_PROBES, ALL_RIS_PEERS, ROUTEVIEWS_PEERS, AS_RELATIONSHIPS, ASDB_PATH, ORIGIN_PATH, TOP_PATH, BGPTOOLS_PATH, _, _ = get_path(
        date)

    if dataset == 'ASrank':
        data, asnlist = create_df_from_AS_rank(PATH_AS_RANK)
    elif dataset == 'bgp_perso':
        data, asnlist = create_df_from_personal(PATH_PERSONAL)
    elif dataset == 'PeeringDB':
        data, asnlist = create_df_from_PeeringDB(PATH_PEERINGDB)
    elif dataset == 'AS_hegemony':
        data, asnlist = create_df_from_AS_hegemony(AS_HEGEMONY_PATH)
    elif dataset == 'RIPE_Atlas_probes':
        data, asnlist = create_df_from_Atlas_probes(ALL_ATLAS_PROBES)
    elif dataset == 'RIPE_RIS_collectors':
        data, asnlist = create_df_from_RIPE_RIS(ALL_RIS_PEERS)
    elif dataset == 'RouteViews':
        data, asnlist = create_df_from_RouteViews(ROUTEVIEWS_PEERS)
    elif dataset == 'AS_relationships':
        data, asnlist = create_df_from_AS_relationships(AS_RELATIONSHIPS)
    elif dataset == 'CTI_origin':
        data, asnlist = create_df_from_cti_origins(ORIGIN_PATH)
    elif dataset == 'CTI_top':
        data, asnlist = create_df_from_cti_top(TOP_PATH)
    elif dataset == 'asdb_1':
        data, asnlist = create_dataframe_from_asdb(1, ASDB_PATH)
    elif dataset == 'asdb_2':
        data, asnlist = create_dataframe_from_asdb(2, ASDB_PATH)
    elif dataset == 'ASDB':
        data, asnlist = create_dataframe_from_asdb(3, ASDB_PATH)
    elif dataset == 'bgptools':
        data, asnlist = create_df_from_bgtools(BGPTOOLS_PATH)
    else:
        raise Exception('Not defined dataset')
    return data, asnlist


def create_dataframe_from_multiple_datasets(list_of_datasets=None):
    """
    Creates a dataframe for each given dataset, and concatenates all the dataframes in a common dataframe.
    The final/returned dataframe has the ASN as the index, and as columns all the columns from all datasets.
    It fills with NaN non-existing values.
    :param list_of_datasets:    (list) a list of str, where each string corresponds to a dataset to be loaded; If not set, loads all available datasets
    :return:                    A dataframe with indexes the ASNs and columns the features loaded from each given dataset
    """
    if list_of_datasets is None:
        # select all datasets
        list_of_datasets = ALL_DATASETS
    data = pd.DataFrame()
    list_of_dataframes = []
    for i in list_of_datasets:
        list_of_dataframes.append(create_df_from(i, DATE)[0])
    final_df = pd.concat(list_of_dataframes, axis=1)
    final_df.index.name = 'ASN'
    final_df.to_csv(AGGREGATE_DATA_FNAME)
    # return final_df


def load_aggregated_dataframe(preprocess=False):
    '''
    Loads the aggregated dataframe
    :param      preprocess:     (bollean) [optional] if set to True, it does some processing steps in the data (e.g., missing values)
    :return:    (pandas.DataFrame) A dataframe with indexes the ASNs and columns the features of the different datasets
    '''
    df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
    if preprocess:
        df['is_personal_AS'].fillna(0, inplace=True)
        df['is_bgptools_peer_v4'].fillna(0, inplace=True)
        df['is_bgptools_peer_v6'].fillna(0, inplace=True)
        df['is_in_peeringDB'].fillna(0, inplace=True)
        # pdb_columns = [c for c in df.columns if c.startswith('peeringDB')]
        # pdb_columns_categorical = [c for c in df.columns if c.startswith('peeringDB') and df[c].dtype=='O']
        # df.loc[df[pdb_columns].isna().all(axis=1), pdb_columns_categorical] = 'Not in PDB'
    return df

# df = load_aggregated_dataframe()
# print(df)
def show_file_size(size):
    """
    Convert size of file from bytes to kb and mb
    :param size: size of a file in bytes
    :return:
    """
    kb = round(size / 1024, 2)
    mb = round(kb / 1024, 2)
    #   print("The file size in -")
    #   print("Bytes: {}".format(size))
    #   print("Kilobytes (KB): {0:.2f}".format(kb))
    #     print("Megabytes (MB): {0:.2f}".format(mb))
    return kb, mb


from string import digits

# create comparasion df
COMPARE_DF = pd.DataFrame(columns=['Current month', 'dataset', 'Previous date size', 'Current date size',
                                   'Difference in size (current - previous)', 'Difference % in size (on prev)',
                                   'ASNs of previous month not in current month', 'prev ASNs list', 'curr ASNs list',
                                   'warnings'])


def compareFileDate(filenames_list):
    """
    Function that reads the dates as datatime objects, from the filenames of all collected datasets (misc),
    compares them to find which is the previous and the current date and returns the two dates.
    :param filenames_list: list that contains the names of the two files.
    :return: two dates (datetime objects)
    """
    date1 = ''.join(c for c in filenames_list[0] if c in digits)
    date2 = ''.join(c for c in filenames_list[1] if c in digits)

    datetime_object1 = datetime.strptime(date1, "%Y%m%d").date()
    datetime_object2 = datetime.strptime(date2, "%Y%m%d").date()

    if datetime_object1 < datetime_object2:
        prevMonth = datetime_object1
        nextMonth = datetime_object2
    else:
        prevMonth = datetime_object2
        nextMonth = datetime_object1
    return prevMonth, nextMonth


def compare_datasets(filename):
    """
    Function that compares the previous month and current month misc files for all collected datasets.
    It finds the two files (with different dates) in the /misc directory, that refer to a dataset and saves them in
    a list. It compares the dates, and then saves in df columns each info and comparison we want to keep
    (name of dataset, month of new dataset, size of each file, difference in size)
    :param filename: The name of dataset file in /misc
    :return: dataframe that contains all the comparisons (as columns) between the two existing files for each dataset.
    """
    df = pd.DataFrame()

    path = '../app/misc/'
    # diff_file = 'monthly_differences_{}.txt'.format(month)
    twoFilenames = []
    for fname in os.listdir(path):
        if filename in fname:
            twoFilenames.append(fname)
            _, file_extension = os.path.splitext(fname)
    print(twoFilenames)
    prevMonth, nextMonth = compareFileDate(twoFilenames)

    df['Current month'] = [calendar.month_name[nextMonth.month]]

    prevMonth = str(prevMonth).replace('-', '')
    nextMonth = str(nextMonth).replace('-', '')

    prevSize = os.stat(path + filename + '_' + prevMonth + file_extension).st_size
    nextSize = os.stat(path + filename + '_' + nextMonth + file_extension).st_size
    prev_kbsize, _ = show_file_size(prevSize)
    next_kbsize, _ = show_file_size(nextSize)

    if prev_kbsize > next_kbsize:
        warnings.warn('Previous month file bigger than current month')
        df['warnings'] = ['oops']

    diff_insize = str(round(next_kbsize - prev_kbsize, 2)) + ' KB'
    diff_insize_perc = str(round(((next_kbsize - prev_kbsize) / prev_kbsize) * 100, 2)) + '%'
    df['dataset'] = [filename]
    df['Previous date size'] = [str(prev_kbsize) + ' KB']
    df['Current date size'] = [str(next_kbsize) + ' KB']
    df['Difference in size (current - previous)'] = [diff_insize]
    df['Difference % in size (on prev)'] = [diff_insize_perc]

    # take asns of dataset
    prev_dsAsns = create_df_from(filename, prevMonth)[1]
    next_dsAsns = create_df_from(filename, nextMonth)[1]
    # print(prev_dsAsns)

    df['prev ASNs list'] = pd.Series([prev_dsAsns])
    df['curr ASNs list'] = pd.Series([next_dsAsns])

    c = np.setdiff1d(prev_dsAsns, next_dsAsns)

    numofasns = str(len(c)) + ' out of ' + str(len(prev_dsAsns))
    df['ASNs of previous month not in current month'] = [numofasns]

    return df


ALL_DATASETS_changes = ['AS_hegemony', 'AS_relationships', 'ASDB', 'ASrank', 'bgptools', 'bgp_perso', 'CTI_origin',
                        'CTI_top',
                        'PeeringDB', 'RIPE_Atlas_probes', 'RIPE_RIS_collectors', 'RouteViews']
#
# for ds in ALL_DATASETS_changes:
#     df = compare_datasets(ds)
#     COMPARE_DF = COMPARE_DF.append(df, ignore_index=True)
# #
# print(COMPARE_DF['Difference in size (current - previous)'], COMPARE_DF['Difference % in size (on prev)'])
# # create agg df, save it in db
# if COMPARE_DF['warnings'].isnull().all():
#     create_dataframe_from_multiple_datasets(ALL_DATASETS)
#     # save in db (replace previous)
# else:
#     warnings.warn('AGGREGATED DATAFRAME NOT SAVED IN DB!')
#
# COMPARE_DF.to_csv('df_changes.csv')

# if no warning at all
#
# create_dataframe_from_multiple_datasets(ALL_DATASETS)
# o, _ = create_df_from_RIPE_RIS('/home/sofia/PycharmProjects/ai4netmon/data/misc/RIPE_RIS_collectors_20230117.json')
# print(len(_))
