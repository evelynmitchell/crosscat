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

# Tests the error of the Kraskov et al (2004) mutual information estimator.
# Roughly recreate Figure 2 in Krasov et al (2004) 
# It will take a long time to run

import crosscat.utils.inference_utils as iu 
from numpy import array
from numpy import zeros
from numpy import mean
from numpy.random import multivariate_normal as mvnrnd
from math import log
import pylab as pl
import sys

mu = zeros(2)

true_mi = lambda rho: -.5*log(1-rho**2.0)

def gen_correlated_data(N, rho):
	sigma = array([[1, rho],[rho, 1]], dtype = float)
	X = mvnrnd(mu, sigma, N)
	x = [ [a] for a in X[:,0]]
	y = [ [a] for a in X[:,1]]
	return x, y

corr = [0.0, .3, .6, .9]
N = [50000, 1000, 750, 500, 350, 250, 150, 100, 75, 50, 25, 10]
x_axis = 1.0/array(N)
do_times = 500

print "Running tests. "
print "Averaged over %i runs." % do_times
print "Correlations: %s" % str(corr)
print "N's: %s" % str(N)
print " "

for r in corr:
	data = zeros((do_times,len(N)))
	t = 0;
	for n in N:
		sys.stdout.write("rho: %1.2f N: %i | " % (r, n))
		for i in range(do_times):
			x, y = gen_correlated_data(n, r)
			err = iu.mi(x, y, k=1) - true_mi(r)
			data[i,t] = err
			sys.stdout.write('.')
			sys.stdout.flush()
			if i % 10 == 0 and i > 0:
				sys.stdout.write('\b'*10)
				sys.stdout.write(" "*10)
				sys.stdout.write('\b'*10)
		t += 1
		sys.stdout.write('\r')
		sys.stdout.write(" "*100)
		sys.stdout.write('\r')

	y_data = mean(data,axis=0)
	assert len(y_data) == len(N)
	pl.plot(x_axis, mean(data,axis=0), label=str(r))

pl.legend()
pl.show()


