# Measurement metadata we keep for each input measurement ID, from the RIPE ATLAS API GET call
KEEP_FIELDS = [
    'af',  # [4, 6] [Not for wifi] IPv4 of IPv6 Address family of the measurement
    'id',  # The unique identifier that RIPE Atlas assigned to this measurement
    'participant_count',  # Number of participating probes
    'probes',  # Probes involved in this measurement
    'target_asn',  # The number of the Autonomous System the IP address of the target belongs to
    'type',  # ["ping", "traceroute", "dns", "sslcert", "http", "ntp", "wifi"] The type of the measurement
]

# Name of the file where the data for all probes are saved
PROBES_DATA_CSV_NAME = '../data/probes_data.csv'

# Bias dimensions
BIAS_DIMENSIONS = [
    'RIR region',
    'Location (country)',
    'Location (continent)',
    'Customer cone (#ASNs)',
    'Customer cone (#prefixes)',
    'Customer cone (#addresses)',
    'AS hegemony',
    'Country influence (CTI origin)',
    'Country influence (CTI top)',
    '#neighbors (total)',
    '#neighbors (peers)',
    '#neighbors (customers)',
    '#neighbors (providers)',
    '#IXPs (PeeringDB)',
    '#facilities (PeeringDB)',
    'Peering policy (PeeringDB)',
    'ASDB C1L1',
    'ASDB C1L2',
    'Network type (PeeringDB)',
    'Traffic ratio (PeeringDB)',
    'Traffic volume (PeeringDB)',
    'Scope (PeeringDB)',
    'Personal ASN'
]
