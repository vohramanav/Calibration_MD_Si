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
  f2 = open('tp_pm15.langevin','r')
  f3 = open('tp_pm20.langevin','r')
  f4 = open('tp_pm25.langevin','r')
  data2 = f2.readlines()
  data3 = f3.readlines()
  data4 = f4.readlines()
  c2_2,c4_2 = [],[]
  c2_3,c4_3 = [],[]
  c2_4,c4_4 = [],[]

  for line in data2:
    if not line.startswith('#'):
      p = line.split()
      if (np.array(p).size > 3):
        c2_2.append(float(p[1]))
        c4_2.append(float(p[3]))
        c2_2v,c4_2v = np.array(c2_2),np.array(c4_2)

  for line in data3:
    if not line.startswith('#'):
      p = line.split()
      if (np.array(p).size > 3):
        c2_3.append(float(p[1]))
        c4_3.append(float(p[3]))
        c2_3v,c4_3v = np.array(c2_3),np.array(c4_3)

  for line in data4:
    if not line.startswith('#'):
      p = line.split()
      if (np.array(p).size > 3):
        c2_4.append(float(p[1]))
        c4_4.append(float(p[3]))
        c2_4v,c4_4v = np.array(c2_4),np.array(c4_4)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  line1, = ax1.plot(c2_2,c4_2,'-b',linewidth=2, label=r'$\mathrm{\frac{dT}{dx} = 2.21}$')
  line2, = ax1.plot(c2_3,c4_3,'-m',linewidth=2, label=r'$\mathrm{\frac{dT}{dx} = 2.95}$')
  line3, = ax1.plot(c2_4,c4_4,'-g',linewidth=2, label=r'$\mathrm{\frac{dT}{dx} = 3.68}$')
  ax1.set_xlabel(r'$\mathrm{x/l}$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{Temperature~(K)}$',fontsize=15)
  #ax1.legend(['line1','line2'],[r'$\mathrm{\Delta T = 30}$',r'$\mathrm{\Delta T = 40}$'])
  ax1.legend()
  plt.title(r'$\mathrm{Length~(l): 27.15 \AA}$')
  fig.savefig('temp_plot.pdf')
  plt.close()

def cond_l(dt,ld,ly,lz,eV_to_kJpmol,ang_to_m,ps_to_s):
  f1 = open('energy_w.txt','r')
  e_data = f1.readlines()
  e1,e2,e3,dT = [],[],[],50
  for line in e_data:
    p = line.split()
    e1.append(float(p[0]))
    e2.append(float(p[1]))
    e3.append(float(p[2]))
  e1v,e2v,lx = np.array(e1),np.array(e2),np.array(e3)
  e_avg = np.zeros((1,e1v.shape[0]))
  e_avg[0,:] = [0.5*(abs(e1v[i]) + abs(e2v[i])) for i in range(e1v.shape[0])]
  k,dx = np.zeros((1,lx.shape[0])),np.zeros((1,lx.shape[0]))
  for i in range(lx.size):
    dq = float((e_avg[0,i]*eV_to_kJpmol))/float(2.0*dt*ld*ps_to_s*ly*lz*pow(ang_to_m,2))
    dx[0,i] = lx[i]*0.5*ang_to_m
    k[0,i] = float(dq*dx[0,i])/float(dT)

#  print (k)
# compute inverse of k and lx for the plot
  lx_inv,k_inv = np.reciprocal(lx),np.reciprocal(k)
  
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(lx_inv,k_inv[0,:],'o')
  ax1.set_xlabel(r'$\mathrm{L^{-1}}(\textrm{\AA}^{-1})$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{k^{-1}}(\textrm{mK/W})$',fontsize=15)
  ax1.legend([r'$\mathrm{\frac{dT}{dx} = 1.84~\frac{K}{\AA}}$'])
  fig.savefig('inv_cond_size.pdf')

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(lx,k[0,:],'o')
  ax1.set_xlabel(r'$\mathrm{Length}~(\textrm{\AA})$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{k}~(\textrm{W/mK})$',fontsize=15)
  ax1.legend([r'$\mathrm{\frac{dT}{dx} = 1.84~\frac{K}{\AA}}$'])
  fig.savefig('cond_size.pdf')

def cond_dT(dt,ld,ly,lz,eV_to_kJpmol,ang_to_m,ps_to_s):
  f1 = open('energy_dT.txt','r')
  e_data = f1.readlines()
  e1,e2,e3,lx = [],[],[],5*ly
  for line in e_data:
    p = line.split()
    e1.append(float(p[0]))
    e2.append(float(p[1]))
    e3.append(float(p[2]))
  e1v,e2v,dT = np.array(e1),np.array(e2),np.array(e3)
  dT_dx = np.zeros((1,dT.shape[0]))
  dT_dx[0,:] = [float(2.0*dT[i])/float(lx) for i in range(dT.shape[0])]
  e_avg = np.zeros((1,e1v.shape[0]))
  e_avg[0,:] = [0.5*(abs(e1v[i]) + abs(e2v[i])) for i in range(e1v.shape[0])]
  k,dx = np.zeros((1,dT.shape[0])),np.zeros((1,dT.shape[0]))
  for i in range(dT.size):
    dq = float((e_avg[0,i]*eV_to_kJpmol))/float(2.0*dt*ld*ps_to_s*ly*lz*pow(ang_to_m,2))
    dx[0,i] = lx*0.5*ang_to_m
    k[0,i] = float(dq*dx[0,i])/float(dT[i])
  
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.plot(dT_dx,k,'o',color='b')
  ax1.set_xlabel(r'$\mathrm{Thermal~Gradient,~\frac{dT}{dx}~(\frac{K}{\AA})}$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{k~(W/mK)}$',fontsize=15)
  ax1.legend([r'$\mathrm{Length: 27.15 \AA}$'])
  fig.savefig('cond_grad.pdf')

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  num_bins = 20
  n, bins, patches = plt.hist(np.transpose(k),num_bins,normed=True,facecolor='blue',alpha=0.75)
  ax1.set_xlabel(r'$\mathrm{k~(W/mK)}$',fontsize=15)
  plt.title(r'$\mathrm{Length: 27.15 \AA,~\frac{dT}{dx}~\in~[1.10, 4.42]~\frac{K}{\AA},~Number~of~Samples: 10}$')
  fig.savefig('hist_grad.pdf')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',help='name of yml input file')
  args = parser.parse_args()
  input_params = yaml.load(open(args.input_file).read())
  dt = input_params["dt"]
  ld = input_params["ld"]
  ly,lz = input_params["ly"],input_params["lz"]
  eV_to_kJpmol = 1.602e-19
  ang_to_m = 1.0e-10
  ps_to_s = 1.0e-12
  temp()
#  cond_size(dt,ld,ly,lz,eV_to_kJpmol,ang_to_m,ps_to_s)
#  cond_dT(dt,ld,ly,lz,eV_to_kJpmol,ang_to_m,ps_to_s)























