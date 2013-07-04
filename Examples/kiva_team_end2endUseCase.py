import argparse, pylab, numpy, pdb
import tabular_predDB.python_utils.data_utils as du
import tabular_predDB.python_utils.sample_utils as su
import tabular_predDB.python_utils.plot_utils as pu
import tabular_predDB.CrossCatClient as ccc
import tabular_predDB.python_utils.file_utils as f_utils

from time import time


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='kiva_team_table.csv',
                    type=str)
parser.add_argument('--inf_seed', default=int(time()), type=int)
parser.add_argument('--gen_seed', default=int(time())+1, type=int)
parser.add_argument('--num_transitions', default=100, type=int)
parser.add_argument('--N_GRID', default=31, type=int)
parser.add_argument('--max_rows', default=25000, type=int)
parser.add_argument('--numDraws', default=25, type=int)
parser.add_argument('--numChains', default=50, type=int)
parser.add_argument('--output_path', default='kiva_team', type=str)

args = parser.parse_args()
filename = args.filename
inf_seed = args.inf_seed
gen_seed = args.gen_seed
num_transitions = args.num_transitions
N_GRID = args.N_GRID
max_rows = args.max_rows
numDraws = args.numDraws
numChains = args.numChains
output_path = args.output_path

engine = ccc.get_CrossCatClient('hadoop', seed=inf_seed, output_path=output_path)

# For Kiva Team table
cctypes = ['continuous','multinomial','multinomial','continuous','multinomial',
           'continuous','continuous','continuous']

# Load the data from table and sub-sample entities to max_rows
T, M_r, M_c = du.read_model_data_from_csv(filename, max_rows, gen_seed,cctypes = cctypes)
T_array = numpy.asarray(T)
num_rows = len(T)
num_cols = len(T[0])
col_names = numpy.array([M_c['idx_to_name'][str(col_idx)] for col_idx in range(num_cols)])
filebase = 'kiva_team_end2end_result'
for colindx in range(len(col_names)):
    print 'Attribute: {0:30}   Model:{1}'.format(col_names[colindx],M_c['column_metadata'][colindx]['modeltype'])

print 'Initializing ...'
# Call Initialize and Analyze
M_c, M_r, X_L_list, X_D_list = engine.initialize(M_c, M_r, T, n_chains = numChains)
completed_transitions = 0
step_size = 20
n_steps = min(step_size, num_transitions)
print 'Analyzing ...'
while (completed_transitions < num_transitions):
    X_L_list, X_D_list = engine.analyze(M_c, T, X_L_list, X_D_list, kernel_list=(),
                                        n_steps=n_steps)
    completed_transitions = completed_transitions+step_size
    print completed_transitions
    saved_dict = {'T':T, 'M_c':M_c, 'X_L_list':X_L_list, 'X_D_list': X_D_list}
    pkl_filename = 'kiva_flat_table_model_{!s}.pkl.gz'.format(str(completed_transitions))
    f_utils.pickle(saved_dict, filename = pkl_filename)

saved_dict = {'T':T, 'M_c':M_c, 'X_L_list':X_L_list, 'X_D_list': X_D_list}
pkl_filename = 'kiva_team_table_model_{!s}.pkl.gz'.format('last')
f_utils.pickle(saved_dict, filename = pkl_filename)
