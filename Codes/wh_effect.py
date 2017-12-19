import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib import mlab
from matplotlib import rc
import yaml
import argparse
import math

rc('text', usetex=True)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('input_params',help='name of yml input file')
  parser.add_argument('energy_w_file',help='data file with exchange energies')
  args = parser.parse_args()
  input_params = yaml.load(open(args.input_params).read())
  dt = input_params["dt"]
  ld = input_params["ld"]
  lx = input_params["lx"]
  eV_to_J = 1.602e-19
  ang_to_m = 1.0e-10
  ps_to_s = 1.0e-12
  energy_w_file = args.energy_w_file
  
  f1 = open(energy_w_file,'r')
  data = f1.readlines()
  nr = np.array(data).shape[0]
  e1,e2,w = [],[],[]
  for line in data:
    p = line.split()
    e1.append(float(p[0]))
    e2.append(float(p[1]))
    w.append(float(p[2]))
  e1v,e2v,wv = np.array(e1),np.array(e2),np.array(w)
  dTdx = 2.0
  e_avg,k,del_k = np.zeros((nr,1)),np.zeros((nr,1)),np.zeros((nr-1,1))
  e_avg[:,0] = [0.5*(abs(e1v[i])+abs(e2v[i])) for i in range(nr)]

  for i in range(nr):
    dq = float((e_avg[i,0]*eV_to_J))/float(2.0*dt*ld*ps_to_s*wv[i]*wv[i]*pow(ang_to_m,2))
    k[i,0] = (dq*0.5*ang_to_m)/dTdx
#
  print (k)
  del_k[:,0] = [100*abs(k[i+1,0]-k[i,0])/k[i,0] for i in range(nr-1)]
  print (del_k)
#  fig = plt.figure()
#  ax = fig.add_subplot(111)
#  ax.plot(del_k,'o')
#  fig.savefig('del_k.pdf')  
  
