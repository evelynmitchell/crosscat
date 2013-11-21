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
from scipy.misc import logsumexp
from scipy.stats import pearsonr as pearsonr
import scipy.spatial as ss
from scipy.special import digamma,gamma
import numpy
import random
from math import log,pi,exp,e
import numpy.random as nr

import crosscat.cython_code.ContinuousComponentModel as CCM
import crosscat.cython_code.MultinomialComponentModel as MCM
import crosscat.utils.sample_utils as su


# convert mutual informatino to Linfoot
mutual_information_to_linfoot = lambda MI: (1.0-exp(-2.0*MI))**0.5

def mutual_information(M_c, X_Ls, X_Ds, Q, n_samples=1000):
    """ return the estimated mutual information for each pair of columns on Q given
        the set of samples in X_Ls and X_Ds. Q is a list of tuples where each tuple
        contains X and Y, the columns to compare. 
        Q = [(X_1, Y_1), (X_2, Y_2), ..., (X_n, Y_n)]
        Returns a list of list where each sublist is a set of MI's and Linfoots from
        each crosscat posterior sample. 
        See tests/test_mutual_information.py and 
        tests/test_mutual_information_vs_correlation.py for useage examples
    """
    assert(len(X_Ds) == len(X_Ls))
    n_postertior_samples = len(X_Ds)

    n_rows = len(X_Ds[0][0])
    n_cols = len(M_c['column_metadata'])

    MI = []
    Linfoot = []
    NMI = []

    get_next_seed = lambda: random.randrange(32767)

    for query in Q:
        assert(len(query) == 2)
        assert(query[0] >= 0 and query[0] < n_cols)
        assert(query[1] >= 0 and query[1] < n_cols)

        X = query[0]
        Y = query[1]

        MI_sample = []
        Linfoot_sample = []
        
        for sample in range(n_postertior_samples):
            
            X_L = X_Ls[sample]
            X_D = X_Ds[sample]

            MI_s = mutual_information_for_query(X, Y, M_c, X_L, X_D,
             get_next_seed, n_samples=n_samples)

            linfoot = mutual_information_to_linfoot(MI_s)
            
            MI_sample.append(MI_s)

            Linfoot_sample.append(linfoot)

        MI.append(MI_sample)
        Linfoot.append(Linfoot_sample)

         
    assert(len(MI) == len(Q))
    assert(len(Linfoot) == len(Q))

    return MI,  Linfoot

def mutual_information_for_query(X, Y, M_c, X_L, X_D, get_next_seed, n_samples=1000):
    """ estimates the mutual information for columns X and Y given the standard 
        algorithm. For a certain number of samples, chooses a cluster 
        (category) in the view, generates a sample from the cluster then 
        calculates the probability of the sample under the joint and marginal
        distributions. 
        Inputs:
            - X: interger. First column
            - Y: interger. Second column
            - M_c: M_c metadata struct. See docs for format
            - X_L: X_L metadata struct. See docs for format
            - X_D: X_D metadata struct. See docs for format
            - get_next_seed: a function or lambda that generates random 
              integers. Note that these integers are for C++ code, so keep the
              interger upper bound in mind or you might get a C++ error from
              passing a too-large integers
            - n_samples: integer, optional. The number of samples to takes
        Output: 
            - MI: float. the mutual information
    """
    
    get_view_index = lambda which_column: X_L['column_partition']['assignments'][which_column]

    view_X = get_view_index(X)
    view_Y = get_view_index(Y)

    # independent
    if view_X != view_Y:
        return 0.0

    # get cluster logps
    view_state = X_L['view_state'][view_X]
    cluster_logps = su.determine_cluster_crp_logps(view_state)
    cluster_crps = numpy.exp(cluster_logps) # get exp'ed values for multinomial
    n_clusters = len(cluster_crps)

    # get components models for each cluster for columns X and Y
    component_models_X = [0]*n_clusters
    component_models_Y = [0]*n_clusters
    for i in range(n_clusters):
        cluster_models = su.create_cluster_model_from_X_L(M_c, X_L, view_X, i)
        component_models_X[i] = cluster_models[X]
        component_models_Y[i] = cluster_models[Y]

    MI = 0.0    # mutual information

    for _ in range(n_samples):
        # draw a cluster 
        cluster_idx = numpy.nonzero(numpy.random.multinomial(1, cluster_crps))[0][0]

        # get a sample from each cluster
        x = component_models_X[cluster_idx].get_draw(get_next_seed())
        y = component_models_Y[cluster_idx].get_draw(get_next_seed())

        # calculate marginal logs
        Pxy = numpy.zeros(n_clusters)   # P(x,y), Joint distribution
        Px = numpy.zeros(n_clusters)    # P(x)
        Py = numpy.zeros(n_clusters)    # P(y)

        # get logp of x and y in each cluster. add cluster logp's
        for j in range(n_clusters):

            Px[j] = component_models_X[j].calc_element_predictive_logp(x)
            Py[j] = component_models_Y[j].calc_element_predictive_logp(y)
            Pxy[j] = Px[j] + Py[j] + cluster_logps[j]   # \sum_c P(x|c)P(y|c)P(c), Joint distribution
            Px[j] += cluster_logps[j]                   # \sum_c P(x|c)P(c)
            Py[j] += cluster_logps[j]                   # \sum_c P(y|c)P(c)    

        # pdb.set_trace()
        
        # sum over clusters
        Px = logsumexp(Px)
        Py = logsumexp(Py)
        Pxy = logsumexp(Pxy)

        # add to MI
        MI += Pxy - (Px + Py)

    # average
    MI /= float(n_samples)

    # ignore MI < 0
    if MI <= 0.0:
        MI = 0.0
        
    return MI

# Estimations are biased and shouldn't be used, this is just for testing purposes.
def mutual_information_for_query_estimate(X, Y, M_c, X_L, X_D, get_next_seed, n_samples=10000):
    """ estimates the mutual information for columns X and Y using further 
        estimation techniques. For a certain number of samples, chooses a 
        cluster (category) in the view, generates a sample from the cluster. 
        Once n_samples samples have been drawn the mutual information is 
        calculated via estimation. We currently use the techniqur outlined in
        Kraskov et al (2004).
        Inputs:
            - X: interger. First column
            - Y: interger. Second column
            - M_c: M_c metadata struct. See docs for format
            - X_L: X_L metadata struct. See docs for format
            - X_D: X_D metadata struct. See docs for format
            - get_next_seed: a function or lambda that generates random 
              integers. Note that these integers are for C++ code, so keep the
              interger upper bound in mind or you might get a C++ error from
              passing a too-large integers
            - n_samples: integer, optional. The number of samples to takes
        Output: 
            - MI: float. the mutual information
    """
    get_view_index = lambda which_column: X_L['column_partition']['assignments'][which_column]

    view_X = get_view_index(X)
    view_Y = get_view_index(Y)

    # independent
    if view_X != view_Y:        
        return 0.0

    # get cluster logps
    view_state = X_L['view_state'][view_X]
    cluster_logps = su.determine_cluster_crp_logps(view_state)
    cluster_crps = numpy.exp(cluster_logps)
    n_clusters = len(cluster_crps)

    # get components models for each cluster for columns X and Y
    component_models_X = [0]*n_clusters
    component_models_Y = [0]*n_clusters
    for i in range(n_clusters):
        cluster_models = su.create_cluster_model_from_X_L(M_c, X_L, view_X, i)
        component_models_X[i] = cluster_models[X]
        component_models_Y[i] = cluster_models[Y]

    MI = 0.0
    samples_x = [[0]]*n_samples
    samples_y = [[0]]*n_samples

    # draw the samples
    for i in range(n_samples):
        # draw a cluster 
        cluster_idx = numpy.nonzero(numpy.random.multinomial(1, cluster_crps))[0][0]

        x = component_models_X[cluster_idx].get_draw(get_next_seed())
        y = component_models_Y[cluster_idx].get_draw(get_next_seed())

        samples_x[i][0] = x
        samples_y[i][0] = y

    MI = mi(samples_x, samples_y)
    return MI

# Some utils for calculating entropy and mutual information from samples.
# log base changed from from 2 to e.
#
# python2.7
# Written by Greg Ver Steeg
# http://www.isi.edu/~gregv/npeet.html
#####CONTINUOUS ESTIMATORS

def entropy(x,k=3,base=e):
  """ The classic K-L k-nearest neighbor continuous entropy estimator
      x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert k <= len(x)-1, "Set k smaller than num. samples - 1"
  d = len(x[0])
  N = len(x)
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  tree = ss.cKDTree(x)
  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
  const = digamma(N)-digamma(k) + d*log(2)
  return (const + d*numpy.mean(map(log,nn)))/log(base)

def mi(x,y,k=3,base=e):
  """ Mutual information of x and y
      x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert len(x)==len(y), "Lists should have same length"
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  points = zip2(x,y)
  #Find nearest neighbors in joint space, p=inf means max-norm
  tree = ss.cKDTree(points)
  dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(len(x)) 
  return (-a-b+c+d)/log(base)

def cmi(x,y,z,k=3,base=e):
  """ Mutual information of x and y, conditioned on z
      x,y,z should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert len(x)==len(y), "Lists should have same length"
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  z = [list(p + intens*nr.rand(len(z[0]))) for p in z]
  points = zip2(x,y,z)
  #Find nearest neighbors in joint space, p=inf means max-norm
  tree = ss.cKDTree(points)
  dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  a,b,c,d = avgdigamma(zip2(x,z),dvec),avgdigamma(zip2(y,z),dvec),avgdigamma(z,dvec), digamma(k) 
  return (-a-b+c+d)/log(base)

def kldiv(x,xp,k=3,base=e):
  """ KL Divergence between p and q for x~p(x),xp~q(x)
      x,xp should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  assert k <= len(xp) - 1, "Set k smaller than num. samples - 1"
  assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
  d = len(x[0])
  n = len(x)
  m = len(xp)
  const = log(m) - log(n-1)
  tree = ss.cKDTree(x)
  treep = ss.cKDTree(xp)
  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
  nnp = [treep.query(point,k,p=float('inf'))[0][k-1] for point in x]
  return (const + d*numpy.mean(map(log,nnp))-d*numpy.mean(map(log,nn)))/log(base)

#####DISCRETE ESTIMATORS
def entropyd(sx,base=e):
  """ Discrete entropy estimator
      Given a list of samples which can be any hashable object
  """
  return entropyfromprobs(hist(sx),base=base)

def midd(x,y):
  """ Discrete mutual information estimator
      Given a list of samples which can be any hashable object
  """
  return -entropyd(zip(x,y))+entropyd(x)+entropyd(y)

def cmidd(x,y,z):
  """ Discrete mutual information estimator
      Given a list of samples which can be any hashable object
  """
  return entropyd(zip(y,z))+entropyd(zip(x,z))-entropyd(zip(x,y,z))-entropyd(z)

def hist(sx):
  #Histogram from list of samples
  d = dict()
  for s in sx:
    d[s] = d.get(s,0) + 1
  return map(lambda z:float(z)/len(sx),d.values())

def entropyfromprobs(probs,base=e):
#Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
  return -sum(map(elog,probs))/log(base)

def elog(x):
#for entropy, 0 log 0 = 0. but we get an error for putting log 0
  if x <= 0. or x>=1.:
    return 0
  else:
    return x*log(x)

#####MIXED ESTIMATORS
def micd(x,y,k=3,base=e,warning=True):
  """ If x is continuous and y is discrete, compute mutual information
  """
  overallentropy = entropy(x,k,base)

  n = len(y)
  word_dict = dict()
  for sample in y:
    word_dict[sample] = word_dict.get(sample,0) + 1./n
  yvals = list(set(word_dict.keys()))

  mi = overallentropy
  for yval in yvals:
    xgiveny = [x[i] for i in range(n) if y[i]==yval]  
    if k <= len(xgiveny) - 1:
      mi -= word_dict[yval]*entropy(xgiveny,k,base)
    else:
      if warning:
        print "Warning, after conditioning, on y=",yval," insufficient data. Assuming maximal entropy in this case."
      mi -= word_dict[yval]*overallentropy
  return mi #units already applied

#####UTILITY FUNCTIONS
def vectorize(scalarlist):
  """ Turn a list of scalars into a list of one-d vectors
  """
  return [(x,) for x in scalarlist]

def shuffle_test(measure,x,y,z=False,ns=200,ci=0.95,**kwargs):
  """ Shuffle test
      Repeatedly shuffle the x-values and then estimate measure(x,y,[z]).
      Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
      'measure' could me mi,cmi, e.g. Keyword arguments can be passed.
      Mutual information and CMI should have a mean near zero.
  """
  xp = x[:] #A copy that we can shuffle
  outputs = []
  for i in range(ns): 
    random.shuffle(xp)
    if z:
      outputs.append(measure(xp,y,z,**kwargs))
    else:
      outputs.append(measure(xp,y,**kwargs))
  outputs.sort()
  return numpy.mean(outputs),(outputs[int((1.-ci)/2*ns)],outputs[int((1.+ci)/2*ns)])

#####INTERNAL FUNCTIONS

def avgdigamma(points,dvec):
  #This part finds number of neighbors in some radius in the marginal space
  #returns expectation value of <psi(nx)>
  N = len(points)
  tree = ss.cKDTree(points)
  avg = 0.
  for i in range(N):
    dist = dvec[i]
    #subtlety, we don't include the boundary point, 
    #but we are implicitly adding 1 to kraskov def bc center point is included
    num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
    avg += digamma(num_points)/N
  return avg

def zip2(*args):
  #zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
  #E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
  return [sum(sublist,[]) for sublist in zip(*args)]
