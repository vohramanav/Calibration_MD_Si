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

def temp():
  parser = argparse.ArgumentParser()
  parser.add_argument('temp_file',help='name of temp input file')
  parser.add_argument('length',help='length of the system')
  parser.add_argument('temp_grad',help='applied temperature gradient')
  args = parser.parse_args()
  f2 = open(args.temp_file,'r')
  data2 = f2.readlines()
  c2_2,c4_2 = [],[]
  l = float(args.length)
  dTdx = float(args.temp_grad)

  for line in data2:
    if not line.startswith('#'):
      p = line.split()
      if (np.array(p).size > 3):
        c2_2.append(float(p[1]))
        c4_2.append(float(p[3]))
        c2_2v,c4_2v = np.array(c2_2),np.array(c4_2)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  line1, = ax1.plot(c2_2,c4_2,'-b',linewidth=2, label=r'$\mathrm{L~=~%3.2f\frac{dT}{dx}~=~%3.2f}$' %(l,dTdx))
  ax1.set_xlabel(r'$\mathrm{x/l}$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{Temperature~(K)}$',fontsize=15)
  #ax1.legend(['line1','line2'],[r'$\mathrm{\Delta T = 30}$',r'$\mathrm{\Delta T = 40}$'])
  ax1.legend()
  #plt.title(r'$\mathrm{Width, Height~(l): 27.15 \AA}$')
  fig.savefig('temp_plot.pdf')
  plt.close()

if __name__ == "__main__":
  temp()

