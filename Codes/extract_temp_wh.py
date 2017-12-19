import argparse
import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib import mlab
from matplotlib import rc

rc('text', usetex=True)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('lammps_file1',help='name of 1st lammps log-file')
  parser.add_argument('lammps_file2',help='name of 2nd lammps log-file')
  parser.add_argument('w1',help='width for case 1')
  parser.add_argument('w2',help='width for case 2')
  parser.add_argument('l',help='length')
  args = parser.parse_args()
  f1 = open(args.lammps_file1,'r')
  f2 = open(args.lammps_file1,'r')
  f3 = open(args.lammps_file2,'r')
  f4 = open(args.lammps_file2,'r')
  w1,w2,l = float(args.w1),float(args.w2),float(args.l)
  cte = 2.0

# Extract temperature files from the 1st lammps-log file
  data_1 = f2.readlines() # f2 is used to check the number of rows
  temp_nvt_1, temp_nve1_1 = np.zeros((40,2)),np.zeros((100,3))
  for i in range(np.array(data_1).shape[0]):
    row = f1.readline()
    if "{tnvt}" in row:
      for n_skip in range(4):
        line_skip = f1.readline()
      for k in range(temp_nvt_1.shape[0]):
        temp_data = f1.readline()
        p = temp_data.split()
        temp_nvt_1[k,0] = float(p[0])
        temp_nvt_1[k,1] = float(p[1])
      np.savetxt('temp_nvt_1.txt',temp_nvt_1,fmt=['%d','%3.2f'])
    
    if "{t1nve}" in row:
      for n_skip in range(4):
        line_skip = f1.readline()
      for k in range(temp_nve1_1.shape[0]):
        temp_data = f1.readline()
        p = temp_data.split()
        temp_nve1_1[k,0] = float(p[0])
        temp_nve1_1[k,1] = float(p[2])
        temp_nve1_1[k,2] = float(p[3])
      np.savetxt('temp_nve1_1.txt',temp_nve1_1,fmt=['%d','%3.2f','%3.2f'])
#    break

# Extract temperature files from the 2nd lammps-log file
  data_2 = f4.readlines() # f4 is used to check the number of rows
  temp_nvt_2, temp_nve1_2 = np.zeros((40,2)),np.zeros((100,3))
  for i in range(np.array(data_2).shape[0]):
    row = f3.readline()
    if "{tnvt}" in row:
      for n_skip in range(4):
        line_skip = f3.readline()
      for k in range(temp_nvt_2.shape[0]):
        temp_data = f3.readline()
        p = temp_data.split()
        temp_nvt_2[k,0] = float(p[0])
        temp_nvt_2[k,1] = float(p[1])
      np.savetxt('temp_nvt_2.txt',temp_nvt_2,fmt=['%d','%3.2f'])
    
    if "{t1nve}" in row:
      for n_skip in range(4):
        line_skip = f3.readline()
      for k in range(temp_nve1_2.shape[0]):
        temp_data = f3.readline()
        p = temp_data.split()
        temp_nve1_2[k,0] = float(p[0])
        temp_nve1_2[k,1] = float(p[2])
        temp_nve1_2[k,2] = float(p[3])
      np.savetxt('temp_nve1_2.txt',temp_nve1_2,fmt=['%d','%3.2f','%3.2f'])
#    break

  dT = (l/4.0)*cte
  thi = 300 + dT
  tlo = 300 - dT

# compute Frobenius norm of the teemperature fluctuations
  dtemp_nvt_1 = temp_nvt_1[10:,1] - 300
  dtemp_nvt_2 = temp_nvt_2[10:,1] - 300
  dtemp_nve1_h1 = temp_nve1_1[10:,1] - (thi)
  dtemp_nve1_h2 = temp_nve1_2[10:,1] - (thi)
  dtemp_nve1_l1 = temp_nve1_1[10:,2] - (tlo)
  dtemp_nve1_l2 = temp_nve1_2[10:,2] - (tlo)
  rn_dnvt_1 = np.linalg.norm(dtemp_nvt_1)/float(dtemp_nvt_1.shape[0])
  rn_dnvt_2 = np.linalg.norm(dtemp_nvt_2)/float(dtemp_nvt_2.shape[0])
  rn_dnve1_h1 = np.linalg.norm(dtemp_nve1_h1)/float(dtemp_nve1_h1.shape[0])
  rn_dnve1_h2 = np.linalg.norm(dtemp_nve1_h2)/float(dtemp_nve1_h2.shape[0])
  rn_dnve1_l1 = np.linalg.norm(dtemp_nve1_l1)/float(dtemp_nve1_l1.shape[0])
  rn_dnve1_l2 = np.linalg.norm(dtemp_nve1_l2)/float(dtemp_nve1_l2.shape[0])
  
  return temp_nvt_1,temp_nvt_2,temp_nve1_1,temp_nve1_2,w1,w2,l,rn_dnvt_1,rn_dnvt_2,rn_dnve1_h1,rn_dnve1_h2,rn_dnve1_l1,rn_dnve1_l2,thi,tlo

def plots(temp_nvt_1,temp_nvt_2,temp_nve1_1,temp_nve1_2,w1,w2,l,rn_dnvt_1,rn_dnvt_2,rn_dnve1_h1,rn_dnve1_h2,rn_dnve1_l1,rn_dnve1_l2,thi,tlo):
  xc = np.linspace(np.amin(temp_nvt_1[:,0]),np.amax(temp_nvt_1[:,0]),50)
  yc = np.linspace(300,300,num=50)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  line1, = ax1.plot(temp_nvt_1[:,0],temp_nvt_1[:,1],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w1,rn_dnvt_1))
  line2, = ax1.plot(temp_nvt_2[:,0],temp_nvt_2[:,1],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w2,rn_dnvt_2))
  line3, = ax1.plot(xc,yc,'--',color='k')
  ax1.set_xlabel(r'$\mathrm{Number~of~Steps}$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{Temperature~(K)}$',fontsize=15)
  ax1.legend(fontsize=12,loc=4)
  plt.title(r'$\mathrm{Length: %dL_c}$' %(l))
  fig.savefig('nvt_temp.pdf')
  
  xc = np.linspace(np.amin(temp_nve1_1[:,0]),np.amax(temp_nve1_1[:,0]),50)
  ycl = np.linspace(tlo,tlo,num=50)
  ych = np.linspace(thi,thi,num=50)
  
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  line1, = ax1.plot(temp_nve1_1[:,0],temp_nve1_1[:,1],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w1,rn_dnve1_h1))
  line2, = ax1.plot(temp_nve1_2[:,0],temp_nve1_2[:,1],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w2,rn_dnve1_h2))
  line3, = ax1.plot(xc,ych,'--',color='k')
  line4, = ax1.plot(temp_nve1_1[:,0],temp_nve1_1[:,2],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w1,rn_dnve1_l1))
  line5, = ax1.plot(temp_nve1_2[:,0],temp_nve1_2[:,2],lw=2.0,label=r'$\mathrm{W = %dL_c~[%3.2f]}$' %(w2,rn_dnve1_l2))
  line6, = ax1.plot(xc,ycl,'--',color='k')
  ax1.set_xlabel(r'$\mathrm{Number~of~Steps}$',fontsize=15)
  ax1.set_ylabel(r'$\mathrm{Temperature~(K)}$',fontsize=15)
  ax1.legend(fontsize=10,loc='best')
  plt.title(r'$\mathrm{Length: %dL_c}$' %(l))
  fig.savefig('nve_temp.pdf')
  
if __name__ == "__main__":
  temp_nvt_1,temp_nvt_2,temp_nve1_1,temp_nve1_2,w1,w2,l,rn_dnvt_1,rn_dnvt_2,rn_dnve1_h1,rn_dnve1_h2,rn_dnve1_l1,rn_dnve1_l2,thi,tlo = main()
  plots(temp_nvt_1,temp_nvt_2,temp_nve1_1,temp_nve1_2,w1,w2,l,rn_dnvt_1,rn_dnvt_2,rn_dnve1_h1,rn_dnve1_h2,rn_dnve1_l1,rn_dnve1_l2,thi,tlo)
