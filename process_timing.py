#!/usr/bin/python
import csv
from collections import namedtuple
#
import numpy
import pylab
pylab.ion()
pylab.show()
#
import Cloudless.examples.DPMB.plot_utils as pu

filename = 'all_timing'
timing_list = []
_timing = namedtuple(
    'timing',
    ' num_features num_obs num_views num_clusters_list num_seconds '
    )
with open(filename) as fh:
    csv_reader = csv.reader(fh, delimiter=':')
    for line_els in csv_reader:
        desired_part = line_els[-1]
        desired_parts = desired_part.split(',')
        num_features = int(desired_parts[0])
        num_rows = int(desired_parts[1])
        num_views = int(desired_parts[2])
        num_clusters_list = map(int, desired_parts[3].split(';'))
        num_seconds = float(desired_parts[4])
        timing = _timing(num_features, num_rows, num_views, num_clusters_list, num_seconds)
        timing_list.append(timing)

def jitterify(in_vals, seed=0, magnitude=.03):
    random_state = numpy.random.RandomState(seed)
    random_vals = random_state.uniform(-magnitude, magnitude, size=len(in_vals))
    in_vals *= 1 + random_vals
    return in_vals

num_features_list = [timing.num_features for timing in timing_list]
num_obs_list = [timing.num_obs for timing in timing_list]
num_seconds_list = [timing.num_seconds for timing in timing_list]
#
unique_num_features = pylab.unique(num_features_list)
data = pylab.array(zip(num_features_list, num_obs_list, num_seconds_list))
colors_list = ['r', 'g', 'b', 'k', 'y', 'c']
pylab.figure()
for color, num_features in zip(colors_list, unique_num_features):
    use_bool = data[:, 0] == num_features
    num_obs_arr = data[use_bool, 1]
    num_seconds_arr = data[use_bool, 2]
    #
    num_obs_arr = jitterify(num_obs_arr)
    num_seconds_arr = jitterify(num_seconds_arr)
    pylab.scatter(num_obs_arr, num_seconds_arr, color=color, label=str(num_features), edgecolor='black')

pylab.legend()
pylab.title('time per iteration for various problem sizes\nlegend denotes number of features')
pylab.ylabel('seconds per iteration')
pylab.xlabel('num observations')
pu.legend_outside(bbox_to_anchor=(0.5, -.1))
pu.savefig_legend_outside('seconds_per_iter_vs_num_obs_' + filename)
