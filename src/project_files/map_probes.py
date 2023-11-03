import pandas as pd


def map_probes(probe_list, v4, v6, probes_df):
    allprobes = probes_df[['asn_v4', 'asn_v6', 'id']]
    probesd = allprobes.loc[probes_df['id'].isin(probe_list)]
    if v4 and not v6:
        probes = probesd[['asn_v4', 'id']]
        probes = probes.dropna()
    elif v6 and not v4:
        probes = probesd[['asn_v6', 'id']]
    else:
        probes = probesd[['asn_v4', 'asn_v6', 'id']]

    # keep the non-existing probes that the user has requested
    notprobes = list(set(probe_list).difference(probes['id'].to_list()))
    probdict = probes.set_index('id').T.to_dict('list')

    asns = list(probdict.values())
    flat_asns = [item for sublist in asns for item in sublist]
    flat_asns = [x for x in flat_asns if str(x) != 'nan']
    # asns_list = list(set(flat_asns))
    asns_list = list(flat_asns)
    return [int(x) for x in asns_list], notprobes

