from .bias_utils import get_features_dict_for_visualizations, bias_score_dataframe
from .data_aggregation_tools import load_aggregated_dataframe
import pandas as pd

test = [1000000000000, 639854511, 31424, 50763, 47950, 206356, 48821, 15435, 47441, 31019, 48362, 198385, 34549, 20932,
        8758, 206499, 15605, 209263, 31742, 12307, 37497, 58057, 39351, 57381, 8492, 59414, 207044, 35619, 47692,
        202365, 25160, 59469, 34177, 59919, 328145, 49432, 49673, 50673, 42275, 29691, 47147, 49420, 50877, 12637,
        21232, 8222, 6939, 52873, 20612, 39122, 39821, 49515, 37239, 26073, 50300, 28634, 12859, 24961, 49788, 58299,
        37680, 210025, 16347, 29504, 267613, 206313, 25220, 52863, 2895, 8896, 196621, 49605, 57695, 20562, 4777, 51405,
        262757, 50304, 8455, 11708, 60501, 28571, 553, 29479, 59605, 8218, 328320, 264268, 12779, 35369, 3741, 59891,
        59689, 52720, 1103, 131477, 41722, 28186, 20912, 28917, 35266, 264911, 29686, 14907, 29680, 34288, 4608, 9002,
        1916, 3130, 37468, 2613, 34019, 38726, 1403, 16735, 3267, 205206, 14840, 513, 32097, 13237, 53070, 13030,
        397143, 6894, 2518, 3333, 42473, 1798, 37100, 24441, 209152, 2603, 28220, 6720, 25091, 8607, 53828, 25933,
        30132, 19151, 6881, 7713, 51185, 204355, 20080, 39120, 852, 5396, 56730, 25152, 263508, 3277, 20764, 3491, 680,
        43607, 19754, 14061, 1140, 28260, 15008, 4826, 38001, 5413, 263075, 15547, 21320, 393950, 54728, 198290, 17639,
        2152, 18106, 6667, 6423, 13786, 23106, 58308, 34872, 51088, 41095, 27446, 57463, 293, 263651, 8283, 34681,
        53767, 63956, 14361, 34224, 7660, 6762, 34854, 38883, 1299, 6082, 58511, 14630, 202194, 8220, 36351, 5392,
        34800, 37271, 28329, 6453, 15562, 64463, 49752, 22652, 38880, 263674, 43376, 4181, 24516, 37721, 1351, 7500,
        7575, 9304, 263152, 20514, 1239, 12350, 29222, 204526, 1280, 51873, 57111, 49134, 29140, 5602, 4739, 11686,
        328474, 8426, 2914, 37989, 263047, 36236, 204092, 55720, 39533, 701, 48571, 57866, 7195, 25227, 24482, 2497,
        61595, 11537, 28792, 57264, 57199, 6539, 286, 49697, 23673, 43578, 174, 56665, 137831, 1221, 45177, 59715,
        64475, 395152, 41327, 3303, 53013, 24875, 19782, 7018, 3257, 31313, 132825, 29467, 3561, 20811, 3549, 51907,
        204028, 64271, 13830, 20205, 3233]


def calc_bias(input_list, listType, nonexistprobes, df):
    # print(input_list)
    # select features for visualization
    FEATURE_NAMES_DICT = get_features_dict_for_visualizations()
    FEATURES = list(FEATURE_NAMES_DICT.keys())
    non_exist_asns = ['The below ASNs were not found in database, bias is calculated for the rest of them']
    # non_exist_asns = []
    ## load data
    # df = load_aggregated_dataframe(preprocess=True)  # Moved its loading in get_bias_data.py

    # create dict with sets
    network_sets_dict = dict()
    network_sets_dict['all'] = df

    # non_exist_asns.append(set(input_list) - set(df.index))
    # if asn in
    if isinstance(input_list[0], str):
        input_list = [int(value) if value.isdigit() else value for value in input_list]

    # check if given asn in the query does not exist in db
    non_exist_asns.append(list(set(input_list) - set(df.index)))

    for asn in non_exist_asns[1]:
        # print(asn)
        input_list.remove(asn)

    final_input_list_length = len(input_list)

    network_sets_dict['bias'] = df.loc[input_list]

    network_sets_dict_for_bias = {k: v[FEATURES] for k, v in network_sets_dict.items() if k != 'all'}

    params = {'method': 'kl_divergence', 'bins': 10, 'alpha': 0.01}
    bias_df,_,_,_ = bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

    # save to csv
    print_df = bias_df.copy()
    print_df.index = [n.replace('\n', '') for n in FEATURE_NAMES_DICT.values()]
    # print(print_df.round(2))
    print_df = print_df.fillna('')

    # print(print_df)

    # nonexist_length = len(non_exist_asns[1])
    # non_exist_df = pd.DataFrame(non_exist_asns[1], columns=['Not found ASNs'])

    # if nonexistprobes is not None:
    #     nonexistpr_length = len(nonexistprobes)
    #     non_exist_prdf = pd.DataFrame(nonexistprobes, columns=['Not found probes'])

    # df to dict
    # biaslist_dict = print_df.to_dict()

    # if listType == 'asn':
    #     found_metadata = {
    #         '#ASNs found': final_input_list_length,
    #         '#ASNs not found': nonexist_length}
    #
    # else:
    #     found_metadata = {
    #         '#probes found': final_input_list_length,
    #         '#probes not found': nonexistpr_length+nonexist_length}
    #
    # biaslist_dict.update(found_metadata)
    #
    # if listType == 'asn':
    #     if nonexist_length > 0:
    #         biaslist_dict.update(non_exist_df.to_dict('list'))
    # else:
    #     if nonexistpr_length > 0:
    #         biaslist_dict.update(non_exist_prdf.to_dict('list'))

    return print_df

    # pd.DataFrame(print_df).to_csv('greedy_hijack_RC_60_monitors_BIAS.csv')

