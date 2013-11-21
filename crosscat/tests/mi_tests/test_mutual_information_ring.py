#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

# calculates mutual information of a 2 column data set with different 
# mutual information. Data is generated from a ring. As the width of the
# ring dicreases the mutual information increases.
# Crosscat is tested against the Kraskov et al (2004) estimator
# It will take a long time to run

import numpy
import pylab as pl
import crosscat.utils.inference_utils as iu
import crosscat.utils.data_utils as du
import crosscat.cython_code.State as State

from scipy.stats import pearsonr as pearsonr

import random
import math

def gen_ring_data(n, w, SEED=0):
	random.seed(SEED)
	numpy.random.seed(SEED)
	T = numpy.zeros((n,2))
	for i in range(n):
		r  = random.uniform(1-w,1) # radius of point
		a  = random.uniform(0,2*math.pi) # angle
		x  = r*math.sin(a)
		y  = r*math.cos(a) 
		T[i,0] = x
		T[i,1] = y

	return T

get_next_seed = lambda : random.randrange(32000)

ring_widths = [.1, .25, .5, .75, 1.0]
colors = ['r','b','c','m','k']
N = 500
n_samples = 10
n_data_sets = 5
pl.figure()
burn_in = 200

nr = 0

for w in ring_widths:
	kraskov_est = []
	crosscat_est = []
	for d in range(n_data_sets): 
		T = gen_ring_data( N, w, SEED=get_next_seed())

		mi_est = iu.mi([ [x] for x in T[:,0]], [ [y] for y in T[:,1]])

		print "num_samples: %i, W: %f, d: %i. " % (N, w, d+1)

		M_c = du.gen_M_c_from_T(T)
		X_Ls = []
		X_Ds = []

		for _ in range(n_samples):
			state = State.p_State(M_c, T)
			state.transition(n_steps=burn_in)
			X_Ds.append(state.get_X_D())
			X_Ls.append(state.get_X_L())

			kraskov_est.append(mi_est)
		
		MI, Linfoot = iu.mutual_information(M_c, X_Ls, X_Ds, [(0,1)], n_samples=200)

		for mi in MI[0]:
			crosscat_est.append(mi)

	color = colors[nr]
	pl.scatter(kraskov_est, crosscat_est, label="width=%1.2f"%w,
	 alpha=.7, s=80, c=color, edgecolors='none')

	nr += 1
	
xl = pl.get(pl.gca(),'xlim')
pl.ylim([0,xl[1]])
pl.xlim([0,xl[1]])

pl.xlabel('Kraskov et al, (2004) estimate')
pl.ylabel('Crosscat estimate')
pl.title('N=%i' % N)
# pl.legend()

pl.show()
