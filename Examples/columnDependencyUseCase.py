# Using CrossCat to Examine Column Dependencies

# Import packages/modules needed
import csv
import os
#
import numpy 
#
import tabular_predDB.python_utils.data_utils as du
import tabular_predDB.python_utils.sample_utils as su
import tabular_predDB.python_utils.plot_utils as pu
import tabular_predDB.settings as S
import tabular_predDB.CrossCatClient as ccc


# Load a data table from csv file. In this example, we use synthetic data
filename = os.path.join(S.path.examples_dir, 'flight_data_subset.csv')
filebase = 'flight_data_subset'
T, M_r, M_c = du.read_model_data_from_csv(filename, gen_seed=0)
T_array = numpy.asarray(T)
num_rows = len(T)
num_cols = len(T[0])
col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])
dataplot_filename = '{!s}_data'.format(filebase)


# Show column types of the data, the text of a few rows of the data, and plot all of the data
for colindx in range(len(col_names)):
    print 'Attribute: {0:30}   Model:{1}'.format(col_names[colindx],M_c['column_metadata'][colindx]['modeltype'])

with open(filename) as fh:
    csv_reader = csv.reader(fh)
    lines = [line for line in csv_reader]
header = lines[0]
get_column_width = lambda x: len(x) + 1
column_widths = map(get_column_width, header)
format_element = lambda (width, element): ('%' + str(width) + 's') % element
for line in lines[:6]:
    print ''.join(map(format_element, zip(column_widths, line[:-1])))

pu.plot_T(T_array, M_c, filename = dataplot_filename)

# Initialize CrossCat Engine and Build Model
engine = ccc.get_CrossCatClient('local', seed = 0)
X_L_list = []
X_D_list = []
numChains = 10
num_transitions = 10

for chain_idx in range(numChains):
    print 'Chain {!s}'.format(chain_idx)
    M_c_prime, M_r_prime, X_L, X_D = engine.initialize(M_c, M_r, T)
    X_L_prime, X_D_prime = engine.analyze(M_c, T, X_L, X_D, kernel_list=(),
                                          n_steps=num_transitions)
    X_L_list.append(X_L_prime)
    X_D_list.append(X_D_prime)


# Visualize clusters in one sample drawn from the model 
viewplot_filename = '{!s}_view'.format(filebase)
pu.plot_views(T_array, X_D_list[4], X_L_list[4], M_c, filename= viewplot_filename)


zplot_filename = '{!s}_feature_z'.format(filebase)
# Construct and plot column dependency matrix
su.do_gen_feature_z(X_L_list, X_D_list, M_c, zplot_filename, filename)
