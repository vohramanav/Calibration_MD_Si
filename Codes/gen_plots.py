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

fXX = open('XX.txt','r')
fYY = open('XX.txt','r')
fd = open('data_pce_surf.txt','r')
s = 40

dfXX,dfYY,dfd = fXX.readlines(),fYY.readlines(),fd.readlines()
XX = np.zeros((s,s))
YY = np.zeros((s,s))
dpce = np.zeros((s,s))

c = 0
for line in dfXX:
  p = line.split()
  XX[c,:] = [float(p[k]) for k in range(s)]
  c = c + 1

c = 0
for line in dfYY:
  p = line.split()
  YY[c,:] = [float(p[k]) for k in range(s)]
  c = c + 1

c = 0
for line in dfd:
  p = line.split()
  dpce[c,:] = [float(p[k]) for k in range(s)]
  c = c + 1

fig = plt.figure()
ax = fig.add_subplot()
plt.pcolor(XX,YY,dpce,cmap='RdBu')
fig.savefig('comp.pdf')


