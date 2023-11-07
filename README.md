# ai4netmon-atlas-patterns-analysis
This is a repository for analysis RIPE Atlas usage patterns, focusing on aspects related to bias.

More specifically, we have developed a python notebook (`main.ipynb`) that allows a user to input RIPE Atlas measurement IDs and get back an analysis of the bias and the patterns that can emerge from these measurements, based on some key plots. Namely, given a set of measurement IDs, the user can see:

1) The top most frequent ASNs and probes.
2) The number of probes and ASNs per measurement.
3) A bias comparison between the input measurements and a random sample of RIPE Atlas probes. The comparison is made with regard to:
    * The total average bias of the two samples.
    * The average bias per dimension between the samples.
4) The number of probes vs the average bias per measurement.
5) The bias CDF across all bias dimensions.
6) The top bias causes per measurement as well as for the entire input measurement set.

In addition, we also print all the ASNs that appear in the input measurements, as well as the ASNs for each input measurement, so the user can use them in order to get further information about their measurement set through the [AI4NetMon Web App](https://app-ai4netmon.csd.auth.gr/).
