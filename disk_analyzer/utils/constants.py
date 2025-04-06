# Data Collector constants
BATCHSIZE = 10_000
COLLECTOR_CFG = './analyzer_cfg.json'
STORAGE_PATH = './Data_collected'

# Data Analyzer constants
STATIC_STATS = [
    'data_size',
    'min_date',
    'max_date',
    'mean_lifetime',
    'max_lifetime',
    'na_rate',
    'truncated_rate',
    'survival_rate',
    'failure_rate',
    'double_failures',
    'mean_time_between_observ',
    'mean_observ_per_day'
]

STATIC_STATS_DESCRIPTION = {
    'data_size': 'number of observations in the data.',
    'min_date': 'date of the first observation.',
    'max_date': 'date of the last observation.',
    'mean_lifetime': 'mean lifetime of disks.',
    'max_lifetime': 'maximum lifetime of disks.',
    'na_rate': 'rate of missing data in each column.',
    'truncated_rate': 'rate of truncated disks.',
    'survival_rate': 'rate of survived disks.',
    'failure_rate': 'rate of failed disks.',
    'double_failures': 'rate of disks that have failed more than once.',
    'mean_time_between_observ': 'mean time between observations of a disk.',
    'mean_observ_per_day': 'mean number of observations per day.'
}


DYNAMIC_STATS = [
    'mean_lifetime',
    'max_lifetime',
    'na_rate',
    'survival_rate',
    'failure_rate',
    'mean_observ'
]

DYNAMIC_STATS_DESCRIPTION = {
    'mean_lifetime': 'mean lifetime of disks of disks alive at the day/month.',
    'max_lifetime': 'maximum lifetime of disks alive at the day/month.',
    'na_rate': 'rate of nans.',
    'survival_rate': 'rate of disks that have survived at the day/month.',
    'failure_rate': 'rate of disks that have failed at the day/month.',
    'mean_observ_per_day': 'mean number of observations.'
}
