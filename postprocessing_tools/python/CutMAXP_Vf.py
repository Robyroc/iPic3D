
import numpy as np

# Uncommnt to make plots only on file
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.mlab as mlab # for setting up the data
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import scipy
import h5py
import os
import sys

#from mpi4py import MPI

""" core routines
"""

#==============================================================================
#  Read Variable
#==============================================================================
# dim = 0,2 makes cuts in the planes
# dim = any other values  does max pressure plane
def read_variables(f, dim):
    global rho,p, b, gb, db, gyro_radius, vth, e, ex, ey, ez
    global bx, by, bz, bz, az, jx, jy, jz, j, rhoc, rho0, rho1
    global efx, efy, efz, kT, efpar, efper1, efper2
    global efx_frame, efy_frame, efz_frame, efx_ram
    global vfx, vfy, vfz
    global spec, uth, vth, wth, vthpar,vthper1, vthper2, small
    global pxx, pyy, pzz, pxy, pxz, pyz, ppar, pper11, pper22
    global jx, jy, jz, qom, rhoc_avg, divE_avg
    global jdote, jdote_par, jdote_perp, lorentz
    global jdotesm, jdotesm_par, jdotesm_perp, lorentzsm
    global zmaxp, ymaxp
    global x_yzcut, dx, dy, dz
    global lamb
    global inertial_legth,gyro_radius,wc,wp,dx,dy,dz
    

    Pxx = np.array(f['/Step#0/Block/Pxx_1/0'])
    Pyy = np.array(f['/Step#0/Block/Pyy_1/0'])
    Pzz = np.array(f['/Step#0/Block/Pzz_1/0'])
    Nz=Pxx.shape[0]
    Ny=Pxx.shape[1]
    Nx=Pxx.shape[2]
    
    Rho1 = np.array(f['/Step#0/Block/rho_1/0'])
    Bx = np.array(f['/Step#0/Block/Bx/0'])
    #zmaxp = maxpZ(Pxx+Pyy+Pzz, Nx, Ny, Nz)
    variable=scipy.ndimage.gaussian_filter(Bx**2, 1, mode='nearest')
    zmaxp = maxpZ(variable, Nx, Ny, Nz)
        
    ymaxp = maxpY(variable, Nx, Ny, Nz)
    #zmaxp = maxpZ(1/Bx**2, Nx, Ny, Nz)
    #ymaxp = maxpY(1/Bx**2, Nx, Ny, Nz)
    
    if dim == 0:
        Ncut = int(Nz/2.)
    elif dim == 1:
        Ncut = int(Ny/2.)
    else:
        Ncut = int(x_yzcut / Lx * Nx)  
    print('Ncut=',Ncut)
    #Ncut=0
    
    Pxx = np.array(f['/Step#0/Block/Pxx_'+spec+'/0'])
    Pyy = np.array(f['/Step#0/Block/Pyy_'+spec+'/0'])
    Pzz = np.array(f['/Step#0/Block/Pzz_'+spec+'/0'])
    Pxy = np.array(f['/Step#0/Block/Pxy_'+spec+'/0'])
    Pxz = np.array(f['/Step#0/Block/Pxz_'+spec+'/0'])
    Pyz = np.array(f['/Step#0/Block/Pyz_'+spec+'/0'])
    
    pxx = cut2D(Pxx, Ncut, dim, zmaxp, ymaxp)
    pyy = cut2D(Pyy, Ncut, dim, zmaxp, ymaxp)
    pzz = cut2D(Pzz, Ncut, dim, zmaxp, ymaxp)
    pxy = cut2D(Pxy, Ncut, dim, zmaxp, ymaxp)
    pxz = cut2D(Pxz, Ncut, dim, zmaxp, ymaxp)
    pyz = cut2D(Pyz, Ncut, dim, zmaxp, ymaxp)
    
    Bx = np.array(f['/Step#0/Block/Bx/0'])
    #Bx += np.array(f['/Step#0/Block/Bx_ext/0'])
    By = np.array(f['/Step#0/Block/By/0'])
    #By += np.array(f['/Step#0/Block/By_ext/0'])
    Bz = np.array(f['/Step#0/Block/Bz/0'])
    #Bz += np.array(f['/Step#0/Block/Bz_ext/0'])


    
    dx = Lx/(Nx-1.)
    dy = Ly/(Ny-1.)
    dz = Lz/(Nz-1.)

    
    print('Nx,Ny,Nz',Nx, Ny, Nz)
    print(dx, dy, dz)
    

    
    
    bx = cut2D(Bx, Ncut, dim, zmaxp, ymaxp)
    by = cut2D(By, Ncut, dim, zmaxp, ymaxp)
    bz = cut2D(Bz, Ncut, dim, zmaxp, ymaxp)
    

    b = np.sqrt(pow(bx,2) + pow(by,2) + pow(bz,2))+small


    
    az = vecpot(dx,dz,bx,bz)

    gbx,gby = np.gradient(b)
    
    #gb = np.sqrt(pow(gbx,2) + pow(gby,2))/b

    Ex = np.array(f['/Step#0/Block/Ex/0'])
    Ey = np.array(f['/Step#0/Block/Ey/0'])
    Ez = np.array(f['/Step#0/Block/Ez/0'])
    ex = cut2D(Ex, Ncut, dim, zmaxp, ymaxp)
    ey = cut2D(Ey, Ncut, dim, zmaxp, ymaxp)
    ez = cut2D(Ez, Ncut, dim, zmaxp, ymaxp)
    exsm=scipy.ndimage.gaussian_filter(ex, smooth, mode='nearest')
    eysm=scipy.ndimage.gaussian_filter(ey, smooth, mode='nearest')
    ezsm=scipy.ndimage.gaussian_filter(ez, smooth, mode='nearest')
    e = np.sqrt(pow(ex,2) + pow(ey,2) + pow(ez,2))
    esm = np.sqrt(pow(exsm,2) + pow(eysm,2) + pow(ezsm,2))
    edotb = ex*bx+ey*by+ez*bz
    esmdotb = exsm*bx+eysm*by+ezsm*bz

   
    Jx = np.array(f['/Step#0/Block/Jx_'+spec+'/0'])
    Jy = np.array(f['/Step#0/Block/Jy_'+spec+'/0'])
    Jz = np.array(f['/Step#0/Block/Jz_'+spec+'/0'])
    jx = cut2D(Jx, Ncut, dim, zmaxp, ymaxp)
    jy = cut2D(Jy, Ncut, dim, zmaxp, ymaxp)
    jz = cut2D(Jz, Ncut, dim, zmaxp, ymaxp)
    j=np.sqrt(jx*jx+jy*jy+jz*jz)



    Rho = np.array(f['/Step#0/Block/rho_'+spec+'/0'])
    Rho0 = np.array(f['/Step#0/Block/rho_'+'0'+'/0'])
    Rho1 = np.array(f['/Step#0/Block/rho_'+'1'+'/0'])
    rho0 = cut2D(Rho0, Ncut, dim, zmaxp, ymaxp)
    rho1 = cut2D(Rho1, Ncut, dim, zmaxp, ymaxp)
    rho = cut2D(Rho, Ncut, dim, zmaxp, ymaxp)
    rho += small
    

    jdote, jdote_par, jdote_perp = compute_energies(ex,ey,ez)
    jdotesm, jdotesm_par, jdotesm_perp = compute_energies(exsm,eysm,ezsm)
    
    # Energy flux
    EFx = np.array(f['/Step#0/Block/EFx_'+spec+'/0'])
    EFy = np.array(f['/Step#0/Block/EFy_'+spec+'/0'])
    EFz = np.array(f['/Step#0/Block/EFz_'+spec+'/0'])
    efx = cut2D(EFx, Ncut, dim, zmaxp, ymaxp)
    efy = cut2D(EFy, Ncut, dim, zmaxp, ymaxp)
    efz = cut2D(EFz, Ncut, dim, zmaxp, ymaxp)
    efx_frame = np.copy(efx)
    efy_frame = np.copy(efy)
    efz_frame = np.copy(efz)
    
    #
    # Get moving frame
    #
    Vfx = np.array(f['/Step#0/Block/Vfx/0'])
    Vfy = np.array(f['/Step#0/Block/Vfy/0'])
    Vfz = np.array(f['/Step#0/Block/Vfz/0'])
    vfx = cut2D(Vfx, Ncut, dim, zmaxp, ymaxp)
    vfy = cut2D(Vfy, Ncut, dim, zmaxp, ymaxp)
    vfz = cut2D(Vfz, Ncut, dim, zmaxp, ymaxp)
 
    # To get the total energy flux we need also the energy of the frame motion
    efx_ram = compute_energy_flux()
    
    #
    # Compute pressure (i.e. remove the residual mean velocity 
    # even once the moving frame is included. 
    # Pressure should not have either the moving frame speed 
    # nor the average speed in the moving frame.
    #
    #
    
    
    # by definition of pressure we do not want to add the frame speed
    # we need still to remove the mean speed in the moving frame
    compute_pressure()
    

    uth = np.sqrt(abs(qom*pxx/rho))
    vth = np.sqrt(abs(qom*pyy/rho))
    wth = np.sqrt(abs(qom*pzz/rho))
    vthpar = np.sqrt(abs(qom*ppar/rho))
    vthper1 = np.sqrt(abs(qom*pper11/rho))
    vthper2 = np.sqrt(abs(qom*pper22/rho))
  
    p = (pxx + pyy + pzz)/3.0
    kT = p/np.abs(rho)
    
    
    # Add frame motion in the current
    jx += vfx * rho
    jy += vfy * rho
    jz += vfz * rho 
    
    
    
    #Lambda = np.array(f['/Step#0/Block/Lambda/0'])
    #lamb = cut2D(Lambda, Ncut, dim, zmaxp, ymaxp)
    
    emin=np.nanpercentile(e[e>0.0],1)
    emin=np.nanpercentile(e[e>0.0],1)/10
    lorentz = np.sqrt(e*e*b*b -edotb**2+1e-15)/(e**2+1e-15)
    lorentzsm = np.sqrt(esm*esm*b*b -esmdotb**2+1e-15)/(esm**2+1e-15)
    lorentz[lorentz==np.nan]=1e5

    c=1
    wc = np.abs(qom)*b/c
    wp = np.sqrt(rho*4*np.pi*qom)
    p=(pxx+pyy+pzz)/3.0
    vth=np.sqrt(np.abs(p/(rho+1e-10)*qom))
    gyro_radius = vth/wc
    inertial_legth = c/wp
#    vth=np.sqrt(np.abs(p/(rho+1e-10)*qom))
#    gyro_radius=vth/wc

    #Rhoc_avg = np.array(f['/Step#0/Block/rho_avg/0'])
    #DivE_avg = np.array(f['/Step#0/Block/divE/0'])
    #rhoc_avg = Rhoc_avg[Ncut][:][:]
    #divE_avg = DivE_avg[Ncut][:][:]

    return True

def generate_cmap(n_bins,sim_color):
    from matplotlib.colors import LinearSegmentedColormap
    b=[0,0,1]
    k=[0,0,0]
    r=[1,0,0]
    g=[0,1,0]
    y=[1,1,0]
    c=[0,1,1]
    m=[1,0,1]
    w=[1,1,1]
    if(sim_color):
        colors = [k,b,c,w,g,y,r]  # R -> G -> B
    else:
        colors = [w,y,r,m,b,k]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm
def cut2DmaxpZ(Vx, Nx, Ny, Nz, zmaxp):
    vect = np.zeros((Ny,Nx))
    
    for i in range(0,Nx):
        for j in range(0,Ny):
            k=int(zmaxp[j,i])
            vect[j,i] = Vx[k,j,i]
    return vect

def cut2DmaxpY(Vx, Nx, Ny, Nz, ymaxp):
    vect = np.zeros((Nz,Nx))
    
    for i in range(0,Nx):
        for j in range(0,Nz):
            k=int(ymaxp[j,i])
            vect[j,i] = Vx[j,k,i]
    return vect

def cut2D(V, Ncut, dim, zmaxp, ymaxp):
    Nz = V.shape[0]
    Ny = V.shape[1]
    Nx = V.shape[2]
    
    #bx = np.zeros((Ny,Nx))
    
    if dim == 0 :
        vect = V[Ncut,:,:]
    elif dim == 1:  
        vect = V[:,Ncut,:]
    elif dim == 2:
        vect = V[:,:,Ncut]
    elif dim == -1:   
        vect = np.zeros((Ny,Nx))
        for i in range(0,Nx):
            for j in range(0,Ny):
                k=int(zmaxp[j,i])
                vect[j,i] = V[k,j,i]
    elif dim == -2:   
        vect = np.zeros((Nz,Nx))
        for i in range(0,Nx):
            for j in range(0,Nz):
                k=int(ymaxp[j,i])
                vect[j,i] = V[j,k,i]

    return vect
def maxpZ(V, Nx, Ny, Nz):
    iz = np.linspace(0, Nz, Nz)
    
    zmaxp = np.zeros((Ny,Nx))
    
    for i in range(0,Nx):
        for j in range(0,Ny):
            range_z = range(int(Nz/3),int(2*Nz/3))
            zmaxp[j,i] = range_z[0]+np.argmin(V[range_z,j,i])
            #zmaxp[j,i] = np.sum(V[range_z,j,i]*range_z)/np.sum(V[range_z,j,i])
            #print(zmaxp[j,i])
    return zmaxp

def maxpY(V, Nx, Ny, Nz):
    iy = np.linspace(0, Ny, Ny)
    
    ymaxp = np.zeros((Nz,Nx))
    
    for i in range(0,Nx):
        for j in range(0,Nz):
            range_y = range(int(Ny/3),int(2*Ny/3))
            ymaxp[j,i] = np.argmax(V[j,range_y,i])
            ymaxp[j,i] = np.sum(V[j,range_y,i]*range_y)/np.sum(V[j,range_y,i])
            #print(zmaxp[j,i])
    return ymaxp
  
def compute_energies(ax,ay,az):
    b2D = 1e-10 + bx*bx + by*by;
    b = b2D + bz*bz;
    perp1x = by/np.sqrt(b2D);
    perp1y = -bx/np.sqrt(b2D);
    perp1z = 0;
    perp2x = bz*bx/np.sqrt(b*b2D);
    perp2y = bz*by/np.sqrt(b*b2D);
    perp2z = -np.sqrt(b2D/b);
                
    jpar = (bx*jx + by*jy + bz*jz )/np.sqrt(b);
    epar = (bx*ax + by*ay + bz*az )/np.sqrt(b);
    
    jdote = jx*ax+jy*ay+jz*az
    jdote_par = jpar*epar
    jdote_perp = jdote - jpar*epar
    return jdote, jdote_par, jdote_perp

def compute_pressure():
    global pxx, pyy, pzz, pxy, pxz, pyz, jx, jy, jz, rho, bx, by, bz, qom, small
    global pani, pagy, pagy2, pper11, pper22, pper12, ppar
    pxx = (pxx - jx*jx / (rho+small) ) /qom;
    pyy = (pyy - jy*jy / (rho+small) ) /qom;
    pzz = (pzz - jz*jz / (rho+small) ) /qom;
    pxy = (pxy - jx*jy / (rho+small) ) /qom;
    pxz = (pxz - jx*jz / (rho+small) ) /qom;
    pyz = (pyz - jy*jz / (rho+small) ) /qom;
    b2D = 1e-10 + bx*bx + by*by;
    b = b2D + bz*bz;
    perp1x = by/np.sqrt(b2D);
    perp1y = -bx/np.sqrt(b2D);
    perp1z = 0;
    perp2x = bz*bx/np.sqrt(b*b2D);
    perp2y = bz*by/np.sqrt(b*b2D);
    perp2z = -np.sqrt(b2D/b);
                
    ppar = bx*pxx*bx + 2*bx*pxy*by + 2*bx*pxz*bz;
    ppar +=  by*pyy*by + 2*by*pyz*bz;
    ppar +=  bz*pzz*bz;
                
    ppar = ppar/b;
                
    pper11 = by*pxx*by - 2*by*pxy*bx + bx*pyy*bx;
    pper11 = pper11/b2D;
                
    pper22 = perp2x*pxx*perp2x + 2*perp2x*pxy*perp2y + 2*perp2x*pxz*perp2z;
    pper22 += perp2y*pyy*perp2y + 2*perp2y*pyz*perp2z;
    pper22 += perp2z*pzz*perp2z;
    
    pper12 = perp1x*pxx*perp2x + perp1x*pxy*perp2y + perp1x*pxz*perp2z;
    pper12 += perp1y*pxy*perp2x + perp1y*pyy*perp2y + perp1y*pyz*perp2z;
    pper12 += perp1z*pxz*perp2x + perp1z*pyz*perp2y + perp1z*pzz*perp2z;
    
    pani = np.log10(ppar/(pper11+pper22+small))
    lam1 = pper11+pper22+np.sqrt((pper11-pper22)**2+4*pper12**2)
    lam2 = pper11+pper22-np.sqrt((pper11-pper22)**2+4*pper12**2)
    pagy = 2.0*(lam1-lam2)/(lam1+lam2+small)
    pagy2 = 2.0*np.sqrt((pper11-pper22)**2+4*pper12**2)/(pper11+pper22+small)
    
    return True

def compute_energy_flux():
    global pxx, pyy, pzz, pxy, pxz, pyz, jx, jy, jz, rho, bx, by, bz, qomr
    global efx, efy, efz, efpar, efper1, efper2
    efx += (vfx * vfx + vfy * vfy + vfz *vfz) * jx /qom 
    efx += 2 * (vfx * pxx + vfy * pxy + vfz * pxz ) /qom 
    efx += vfx * (pxx + pyy + pzz) /qom 
    efx += vfx * rho * (vfx * vfx + vfy * vfy + vfz * vfz) /qom 
    efx_ram = vfx * rho * (vfx * vfx + vfy * vfy + vfz * vfz) /qom
    efx += 2 * vfx * (vfx * jx + vfy * jy + vfz * jz) /qom
    
    efy += (vfx * vfx + vfy * vfy + vfz *vfz) * jy /qom  
    efy += 2 * (vfx * pxy + vfy * pyy + vfz * pyz ) /qom 
    efy += vfy * (pxx + pyy + pzz) /qom 
    efy += vfy * rho * (vfx * vfx + vfy * vfy + vfz * vfz) /qom 
    efy += 2 * vfy * (vfx * jx + vfy * jy + vfz * jz) /qom
        
    efz += (vfx * vfx + vfy * vfy + vfz *vfz) * jz /qom  
    efz += 2 * (vfx * pxz + vfy * pyz + vfz * pzz ) /qom 
    efz += vfz * (pxx + pyy + pzz) /qom 
    efz += vfz * rho * (vfx * vfx + vfy * vfy + vfz * vfz) /qom 
    efz += 2 * vfz * (vfx * jx + vfy * jy + vfz * jz) /qom
    
    b2D = 1e-10 + bx*bx + by*by;
    b = b2D + bz*bz;
    perp1x = by/np.sqrt(b2D);
    perp1y = -bx/np.sqrt(b2D);
    perp1z = 0;
    perp2x = bz*bx/np.sqrt(b*b2D);
    perp2y = bz*by/np.sqrt(b*b2D);
    perp2z = -np.sqrt(b2D/b);
  
    efpar = (efx * bx + efy * by + efz *bz) / np.sqrt(b)
    efper1 = perp1x * efx + perp1y * efy + perp1z * efz
    efper2 = perp2x * efx + perp2y * efy + perp2z * efz
    
    return efx_ram  

def vecpot(dx,dy,bx,by):
    ny = bx.shape[0]
    nx = bx.shape[1]
    nymezzo = ny-1 #int(np.ceil(ny/2))
    #print('vecpot',nx,ny)
    az=np.zeros((ny,nx))
    ny = az.shape[0]
    nx = az.shape[1]
    #print('vecpotaz',nx,ny)
    for i in range(1,nx):
      az[nymezzo,i] = az[nymezzo,i-1]- (by[nymezzo,i-1]+by[nymezzo,i])*dx/2.0

    for ind in range(nymezzo+1,ny):
        for j in range(0,nx):
            az[ind,j]=az[ind-1,j]+ (bx[ind,j] + bx[ind-1,j])*dy/2.0
    
    for ind in range(nymezzo-1,0,-1):
        for j in range(0,nx):
            az[ind,j]=az[ind+1,j]- (bx[ind+1,j] + bx[ind,j])*dy/2.0

    return az
  
def vecpot2(dx,dy,bx,by):
    nx = bx.shape[0]
    ny = bx.shape[1]
    nymezzo = int(np.ceil(ny/2))
    #print('vecpot',nx,ny)
    az=np.zeros((nx,ny))
    for i in range(1,nx):
      az[i][nymezzo] = az[i-1][nymezzo]- (bx[i-1][nymezzo]+bx[i][nymezzo])*dy/2.0

    for ind in range(nymezzo+1,ny):
        for j in range(0,nx):
            az[j,ind]=az[j][ind-1]+ (by[j][ind-1] + by[j][ind-1])*dx/2.0
    
    for ind in range(nymezzo-1,-1,-1):
        for j in range(0,nx):
            az[j,ind]=az[j][ind+1]- (by[j][ind+1] + by[j][ind])*dx/2.0
    return az
  
def cross_product(Vx, Vy, Vz, Bx, By, Bz):
    Zx = (Vy* Bz - Vz* By)
    Zy = (Vz* Bx - Vx* Bz)
    Zz = (Vx* By - Vy* Bx)
    return Zx, Zy, Zz

import math
def round_to_n(x, n):
    if not x: return 0
    power = -int(math.floor(math.log10(abs(x)))) + (n - 1)
    factor = (10 ** power)
    return round(x * factor) / factor

#==============================================================================
#  Plot Variable
#==============================================================================
def plot_stream(log,dim, time_label, Lx, Ly, varin, bx, by, bz, name_variable, minval, maxval, sm, sim_color,unit):
    global directory, plane, xc, yc, zc, rc
    
   
    if dim ==0:
    # code plan XY
        plane='XY'
        x_c = xc
        y_c = yc
        xmin = 0
        xmax = Lx
        ymin = 0
        ymax = Ly
        Nx = bx.shape[1]
        Ny = bx.shape[0]
        b0 = bx
        b1 = by
        Y, X = np.mgrid[0:Ly:Ny*1j,0:Lx:Nx*1j]
        labelx = '$x/R_H$'
        labely = '$y/R_H$'
    elif dim == 1:
    #code plan XZ
        plane='XZ'
        x_c = xc
        y_c = zc
        xmin = 0
        xmax = Lx
        ymin = 0
        ymax = Lz
        Nx = bx.shape[1]
        Nz = bx.shape[0]
        b0 = bx
        b1 = bz
        Y, X = np.mgrid[0:Lz:Nz*1j,0:Lx:Nx*1j]
        labelx = 'x'
        labely = 'z'
    elif dim == 2:
        plane='YZ'
        x_c = yc
        y_c = zc
        xmin = 0
        xmax = Ly
        ymin = 0
        ymax = Lz
        b0 = by
        b1 = bz
        Ny = bx.shape[1]
        Nz = bx.shape[0]
        Y, X = np.mgrid[0:Lz:Nz*1j,0:Ly:Ny*1j]
        labelx = 'y'
        labely = 'z'
    elif dim == -1:
    #code plan MAXP ALONG CODE Z
        plane ='MAXPZ'
        x_c = xc
        y_c = yc
        Nx = bx.shape[1]
        Ny = bx.shape[0]
        b0 = bx
        b1 = by
        xmin = 0
        xmax = Lx
        ymin = 0
        ymax = Ly
        Y, X = np.mgrid[0:Ly:Ny*1j,0:Lx:Nx*1j]
        labelx = 'x'
        labely = 'y'
    elif dim == -2:
    # code plan MAXP ALONG CODE Y
        plane ='MAXPY'
        x_c = xc
        y_c = zc
        Nx = bx.shape[1]
        Ny = bx.shape[0]
        b0 = bx
        b1 = bz
        xmin = 0
        xmax = Lx
        ymin = 0
        ymax = Lz   
        Y, X = np.mgrid[0:Lz:Nz*1j,0:Lx:Nx*1j]
        labelx = 'x'
        labely = 'z'
    
    #xc_loc = np.sum(X*lamb,)/np.sum(lamb) 
    #yc_loc = np.sum(Y*lamb)/np.sum(lamb) 
    #rc_loc = 2*np.sqrt(np.sum((X-xc_loc)**2*lamb)/np.sum(lamb))
    #rc_loc = np.amax(np.abs(X[lamb>1e-5]-xc_loc))
    #print('xc = ', x_c,'  xcl = ', xc_loc)
    #print('yc = ', y_c,'  ycl = ', yc_loc)
    #print('rc = ', rc,'  rcl = ', rc_loc)
    
    #color = generate_cmap(256,sim_color)
    if(sim_color):
        color = matplotlib.cm.seismic
        color_contour='g'
        color.set_bad(color='yellow')
    else:
        #color = generate_cmap(256,sim_color)
        color = matplotlib.cm.nipy_spectral #jet #'viridis'
        color_contour='w'
        color.set_bad(color='magenta')
    
    U = b0.copy()

    mask = np.zeros(U.shape, dtype=bool)
    mask[np.where(pow(X-x_c,2)+pow(Y-y_c,2)<pow(rc,2))] = True
    if dim != 2:
        U[np.where(pow(X-x_c,2)+pow(Y-y_c,2)<pow(rc,2))] = np.nan
        U = np.ma.array(U, mask=mask)
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    strm = ax.streamplot(X, Y, U, b1, density=1.5, linewidth=0.5, arrowsize=0.5, color=color_contour,zorder=1)
   
    levels = 20
    
    variable=varin.copy()
    if sm:
        variable=scipy.ndimage.gaussian_filter(variable,sigma=smooth, mode='nearest')
        

        
    #variable[np.where(pow(X-xc,2)+pow(Y-yc,2)<pow(rc,2))] = np.nan
    
    #variable = np.ma.array(U, mask=mask)


    if minval==0 and maxval==0:
            nskip = 50
            minval = round_to_n(np.nanpercentile(variable,1),4)
            maxval = round_to_n(np.nanpercentile(variable,99),4)
            #avg = np.mean(variable)
            #std = np.std(variable).max()
        #minval = round_to_n(avg - 3*std,4)
        #maxval = round_to_n(avg + 3*std,4)
            if(sim_color):
                maxval = np.max([np.abs(minval), np.abs(maxval)])
                minval = - maxval

    #print(name_variable, minval,maxval)

    minval = round_to_n(minval,2)
    maxval = round_to_n(maxval,2)

    if minval==0 and maxval==0:
        return False
    if log:
        CF = ax.imshow(variable, cmap=color, norm=LogNorm(vmin=minval,vmax=maxval),
           extent=(xmin, xmax, ymin, ymax),
           interpolation='lanczos', origin='lower',zorder=0)
    else:
        CF = ax.imshow(variable, cmap=color, vmin=minval, vmax=maxval,
           extent=(xmin, xmax, ymin, ymax),
           interpolation='lanczos', origin='lower',zorder=0)
    ax.set_xlabel(labelx, fontsize=12)
    ax.set_ylabel(labely, fontsize=12)
    plt.title(name_variable+'  frame='+time_label)
# plot the contour lines
#   add mask
    #CL = plt.imshow(~mask, extent=(0, Lx, 0, Ly), alpha=0.5,
    #      interpolation='nearest', cmap='gray', aspect='auto')
    if dim != 2: 
        # circ = Circle((x_c,y_c),rc,facecolor='#800000') #granata
        circ = Circle((x_c,y_c),rc/1.4*1.2,facecolor='#009B7D',alpha=.6)
        circ.set_zorder(2)
        #ax.add_patch(circ)
        #ax.imshow(hermes,extent=(x_c -rc/1.4, xc+rc/1.4, y_c-rc/1.4, y_c+rc/1.4),zorder=3)
#        circ = Circle((x_c,y_c),rc/1.4,facecolor='#9B009B')
#        ax.add_patch(circ)
# using gray scale
    #levels = 60
    #CL = plt.contour(az, levels,
    #             linewidths=0.3,
    #             extent=(0,Lx,0,Ly),linestyles='solid',colors='k')#cmap=cm.gray)

# plot color bars for both contours (filled and lines)
#CB = plt.colorbar(CL, extend='both')
    mn=minval      # colorbar min value
    mx=maxval       # colorbar max value
    md=round_to_n((mx+mn)/2,4)                     # colorbar midpoint value
    #print('round',mn,md,mx)
    CBI = plt.colorbar(CF, ticks=[mn, md,mx],orientation='vertical',shrink=0.75)
    CBI.set_ticklabels([mn,md,mx])
    CBI.ax.set_title(unit, rotation=0)

# Plotting the second colorbar makes
# the original colorbar look a bit out of place,
# so let's improve its position.

#l,b,w,h = plt.gca().get_position().bounds
#ll,bb,ww,hh = CB.ax.get_position().bounds
#CB.ax.set_position([ll, b, ww, h])



    
    # Immagine
    #im = plt.imread('/Users/giovannilapenta/Desktop/hermes.png')
    #plt.imshow(im)
    if dim==0:
       plt.ylim((Ly/2-Lx,Ly/2+Lx))
    plt.xlim(xmin+10*9.6/102.86, xmax-10*9.6/102.86)
    plt.ylim(ymin+10*9.6/102.86, ymax-10*9.6/102.86)
    #plt.show()
    fname=directoryOUT+plane+'_'+name_variable+time_label+'.png' # .svg also works very well
    #plt.savefig(fname, dpi=1200)
    fig.savefig(fname, dpi=300,  bbox_inches="tight")
    plt.clf()
    return maxval

def plotter_moment(dim, spec, ispec, qom): #, kTn, kTramn):
    small=1e-50*qom;
    plot_stream(False,dim, time_label,Lx, Ly, uth*code_V, bx, by, bz, 'vthx'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, vth*code_V, bx, by, bz, 'vthy'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, wth*code_V, bx, by, bz, 'vthz'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, vthpar*code_V, bx, by, bz, 'vth_par'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, vthper1*code_V, bx, by, bz, 'vth_per1'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, vthper2*code_V, bx, by, bz, 'vth_per2'+spec,0,vthn, False, False,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, 2*np.abs(vthper1-vthper2)/(vthper2+vthper1+small), bx, by, bz, 'vth_agy'+spec,0,1, False, False,'')
    plot_stream(False,dim, time_label,Lx, Ly, np.log10(vthpar/(vthper2+vthper1+small)), bx, by, bz, 'vth_ani'+spec,-1,1, False, False,'')
        
    firehose = 1-(ppar-pper11)*4*np.pi/b/b
    plot_stream(False,dim, time_label,Lx, Ly, firehose, bx, by, bz, 'firehose'+spec,-10,10, True, True,'')
    plot_stream(False,dim, time_label,Lx, Ly, ppar, bx, by, bz, 'ppar'+spec,0,0, False, False,'')
    plot_stream(False,dim, time_label,Lx, Ly, pper11, bx, by, bz, 'pper11'+spec,0,0, False, False,'')
    plot_stream(False,dim, time_label,Lx, Ly, pper22, bx, by, bz, 'pper22'+spec,0,0, False, False,'')
    
    
    plot_stream(False,dim, time_label,Lx, Ly, jx/rho*code_V, jx/rho, jy/rho, jz/rho, 'vx'+spec,-vn,vn, logicals[ispec], True,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, jy/rho*code_V, jx/rho, jy/rho, jz/rho, 'vy'+spec,-vn,vn, logicals[ispec], True,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, jz/rho*code_V, jx/rho, jy/rho, jz/rho, 'vz'+spec,-vn,vn, logicals[ispec], True,'Km/s')
    plot_stream(False,dim, time_label,Lx, Ly, jx*code_J, bx, by, bz, 'jx'+spec,-jn,jn, logicals[ispec], True,'$nA/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jy*code_J, bx, by, bz, 'jy'+spec,-jn,jn, logicals[ispec], True,'$nA/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jz*code_J, bx, by, bz, 'jz'+spec,-jn,jn, logicals[ispec], True,'$nA/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, j*code_J, bx, by, bz, 'j'+spec,0,jn, logicals[ispec], False,'$nA/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdote*code_J*code_E*1e-3, bx, by, bz, 'jE'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdote_par*code_J*code_E*1e-3, bx, by, bz, 'jE_par'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdote_perp*code_J*code_E*1e-3, bx, by, bz, 'jE_perp'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdotesm*code_J*code_E*1e-3, bx, by, bz, 'jEsm'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdotesm_par*code_J*code_E*1e-3, bx, by, bz, 'jEsm_par'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, jdotesm_perp*code_J*code_E*1e-3, bx, by, bz, 'jEsm_perp'+spec,-jEn,jEn, logicals[ispec], True,'$nW/m^3$')
    plot_stream(False,dim, time_label,Lx, Ly, efx*code_J*code_E*code_space*1e-9, efx, efy, efz, 'efx'+spec,-efxn,efxn, False, True,'$mW/m^2$')
    plot_stream(False,dim, time_label,Lx, Ly, efy*code_J*code_E*code_space*1e-9, efx, efy, efz, 'efy'+spec,-efyn,efyn, False, True,'$mW/m^2$')
    plot_stream(False,dim, time_label,Lx, Ly, efz*code_J*code_E*code_space*1e-9, efx, efy, efz, 'efz'+spec,-efzn,efzn, False, True,'$mW/m^2$')
    plot_stream(False,dim, time_label,Lx, Ly, efx_frame*code_J*code_E*code_space*1e-9, efx_frame, efy_frame, efz_frame, 'efx_frame'+spec,-efxn,efxn, False, True,'$mW/m^2$')
    plot_stream(False,dim, time_label,Lx, Ly, efy_frame*code_J*code_E*code_space*1e-9, efx_frame, efy_frame, efz_frame, 'efy_frame'+spec,-efyn,efyn, False, True,'$mW/m^2$')
    plot_stream(False,dim, time_label,Lx, Ly, efz_frame*code_J*code_E*code_space*1e-9, efx_frame, efy_frame, efz_frame, 'efz_frame'+spec,-efzn,efzn, False, True,'$mW/m^2$')
    
    plot_stream(False,dim, time_label,Lx, Ly, efx_ram*code_J*code_E*code_space*1e-9, efx_frame, efy_frame, efz_frame, 'efx_ram'+spec,-efxn,efxn, False, True,'$mW/m^2$')
     
    plot_stream(False,dim, time_label,Lx, Ly, np.abs(rho)*code_n, bx, by, bz, 'rho'+spec,0,code_n*0, False, False,'$cm^{-3}$')
    #plot_stream(False,dim, time_label,Lx, Ly, np.log10(np.abs(rho*code_n)), bx, by, bz, 'log10rho'+spec,-2,1, False, False)
    plot_stream(False,dim, time_label,Lx, Ly, pani, bx, by, bz, 'pani'+spec,-1,1, True, True,'')
    plot_stream(False,dim, time_label,Lx, Ly, pagy, bx, by, bz, 'pagy'+spec,0,1, True, False,'')
    plot_stream(False,dim, time_label,Lx, Ly, kT*code_T, bx, by, bz, 'kT'+spec,0,kTn, False, False,'')
    
    plot_stream(False,dim, time_label,Lx, Ly, np.log10(gyro_radius/dx+1e-15), bx, by, bz, 'rsodx_log'+spec,-2.,2., False, True,'')
    plot_stream(False,dim, time_label,Lx, Ly, np.log10(inertial_legth/dx+1e-15), bx, by, bz, 'dsox_log'+spec,-2,2, False, True,'')
    plot_stream(False,dim, time_label,Lx, Ly, -np.log10(dtc*wp), bx, by, bz, 'odtwp'+spec,-2,2, False, True,'')
    plot_stream(False,dim, time_label,Lx, Ly, -np.log10(dtc*wc), bx, by, bz, 'odtwc'+spec,-2,2, False, True,'')
    
    plot_stream(True,dim, time_label,Lx, Ly,np.abs(qom)*(gyro_radius/dx)**2, bx, by, bz, 'qom_rs'+spec,1,1000, False, False,'')
    plot_stream(True,dim, time_label,Lx, Ly, np.abs(qom)*(inertial_legth/dx)**2, bx, by, bz, 'qom_ds'+spec,1,1000, False, False,'')
 
    return True

def gsmx(x):
    xgsm=-x/Lx*(Xgsmrange[1]-Xgsmrange[0])+Xgsmrange[1];
    return xgsm
def gsmy2z(y):
    zgsm= y/Ly*(Zgsmrange[1]-Zgsmrange[0])+Zgsmrange[0];
    return zgsm
def gsmz2y(z):
    ygsm= z/Lz*(Ygsmrange[1]-Ygsmrange[0])+Ygsmrange[0];
    return ygsm

#==============================================================================
#  Main program
#==============================================================================
##0 6000 ogni 100

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#print('My MPI rank is ',rank)
#size = comm.Get_size()
#print('My MPI size is ',size)

directory = './barrier/'
directoryOUT = directory
# os.system('mkdir '+ directoryOUT)
#ini=850
#nt=np.int(sys.argv[2])
#ini=np.int(sys.argv[1])
#dt=np.int(sys.argv[3])


Lx=102.86
Ly=128.58
Lz=Ly
ff= 1.7144
xc = 21.2*ff
yc = 37.5*ff
zc = yc
rc = 6.25*ff

rc *=1.2

x_yzcut = 5.5
x_yzcut = 6.0

Lx = 9.6
Ly=12
Lz=12
rc *=9.6/102.86
xc *=9.6/102.86
yc *= 9.6/102.86
zc *= 9.6/102.86
xc = 212./600.*Lx
yc = Ly/2
zc = Lz/2
rc = 25./300.*Ly
rc *= 1.4

#Mercury2022_3
Lx = 32.5262
Ly = 40.6577
Lz=Ly
xc = 11.4926
yc = Ly/2
zc = yc
rc = 4.2
dtc = 1.

#Dayside
Lx = 14
Ly=8
Lz=8
xc = Lx/2
yc = Ly/2
zc = Lz/2
rc = 0
dtc = 1. #not sure
x_yzcut = 8.0

#rc/=2

#Generic
Xgsmrange= [-Lx, 0]
Zgsmrange= [-Ly/2, Ly/2]
Ygsmrange= [-Lz/2, Lz/2]

# IN SI
code_time = 1/1.3166e+03
code_space = 2.2771e+05
code_n=1000000
code_J=4.8033e-05
code_V=299792251.7596
code_T=1.5033e-10
code_E=4120.4228
code_B=1.3744e-05

# new case
code_n=1e7
code_J=4.8033e-04
code_E=13029.9211
code_B=4.3463e-5
code_space = 7.2007e4
code_time = 1/4.1634e3

# E in mV/m
code_E *=1e3
#B in nT
code_B *=1e9
# J in nA/m^2
code_J *= 1e9
code_V = 2.99792e+08
# V in Km/s
code_V *= 1e-3
#n in p/cm^3
code_n *=1e-6
#kTin keV
code_T *=1e-3/1.6e-19
smooth=1

ns=2
qomsp=[-1836, 1]
#vthn=[0.015, 0.0021]
vthns=0*np.array([0.03, 0.0005]) *code_V
# to have it in mW/m^2
efxns=0*np.array([1e-9,1e-9]) *code_J*code_E*code_space*1e-9
efyns=0*np.array([1e-9,1e-9]) *code_J*code_E*code_space*1e-9
efzns=0*np.array([1e-9,1e-9]) *code_J*code_E*code_space*1e-9
bn = 0*0.002*code_B
en=0*1e-5*code_E
vns = 0*np.array([0.004,0.002])*code_V
jns = 0*np.array([0,0])
jEns = 0*np.array([4.1,2.1])
logicals=[False, False]
kTns=0*np.array([0.5,.5])
# label of species
#spec='0' # 0=electrons, 1=ions
#qom=qomsp[0]
#small=1e-50*qom;

#hermes = image = plt.imread('hermes.png')

#times = list(range(nt-rank*dt, ini-dt,-dt*size))

#times=[1,100,1000, 1600, 2400, 3000, 3500, 4800 ] # 1000, 1700]
#times=[1,500,1000, 1700]
#times=[1600]
times=[1]
#for it in range(rank,len(times),size):

for i in times:
    #i=times[it]
    print('cycle',i)
    it=i
    if(it==0):
         it=1
    # time_label = str(it).zfill(6)
    
    print(directory + 'proc0.hdf')
    print(os.getcwd())

    filename = directory + 'proc0.hdf'
    
    filename = directory+'ForceFree-Fields_' + time_label +'.h5'
    #print filename
    f = h5py.File(filename, 'r')
    for dim in [0,1,2]:#[0,-1,1]:
             
        for ispec in range(0,ns):

            qom = qomsp[ispec]
            vthn = vthns[ispec]
            efxn = efxns[ispec]
            efyn = efyns[ispec]
            efzn = efzns[ispec]
            vn = vns[ispec]
            jn = jns[ispec]
            jEn = jEns[ispec]
            kTn = kTns[ispec]
            small=1e-50*qom;
            spec=str(ispec).zfill(1) # 0=electrons, 1=ions
            read_variables(f,dim)
            #dim = 0 # XY plane
            #dim = 1 # XZ plane
            #dim = 2 # YZ plane
            #dim = -1 # MAX P plane
        
            plotter_moment(dim, spec, ispec, qom)#, kTn, kTramn)
    
        
        # #plot_stream(False,dim, time_label,Lx, Ly, lamb, bx, by, bz, 'lambda',0,0, False, False,'')
        # plot_stream(False,dim, time_label,Lx, Ly, ex*code_E, bx, by, bz, 'ex',-en,en, False, True,'mV/m')
        # plot_stream(False,dim, time_label,Lx, Ly, ey*code_E, bx, by, bz, 'ey',-en,en, False, True,'mV/m')
        # plot_stream(False,dim, time_label,Lx, Ly, ez*code_E, bx, by, bz, 'ez',-en,en, False, True,'mV/m')
        # plot_stream(False,dim, time_label,Lx, Ly, bx*code_B, bx, by, bz, 'bx',-bn,bn, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, by*code_B, bx, by, bz, 'by',-bn,bn, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, bz*code_B, bx, by, bz, 'bz',-bn,bn, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, vfx, bx, by, bz, 'vfx',0,0, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, vfy, bx, by, bz, 'vfy',0,0, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, vfz, bx, by, bz, 'vfz',0,0, False, True,'nT')
        # plot_stream(False,dim, time_label,Lx, Ly, 1.0/(lorentz), bx, by, bz, 'lorentz',0,0, False, False,'')
        # plot_stream(False,dim, time_label,Lx, Ly, 1.0/(lorentzsm), bx, by, bz, 'lorentzsm',0,0, False, False,'')
        # #vexbx, vexby, vexbz = cross_product(ex, ey, ez, bx, by, bz)
        # #plot_stream(False,dim, time_label,Lx, Ly, -vexbx/b/b, bx, by, bz, 'Vexbx',0,0, False, True)
        # #plot_stream(False,dim, time_label,Lx, Ly, vexby/b/b, bx, by, bz, 'Vexbz',0,0, False, True)
        # #plot_stream(False,dim, time_label,Lx, Ly, vexbz/b/b, bx, by, bz, 'Vexby',0,0, False, True)
        # if dim == -2:
        #     plot_stream(False,dim, time_label,Lx, Ly, zmaxp, bx, by, bz, 'ymaxp',0,0, False, False,'')
        # if dim == -1:
        #     plot_stream(False,dim, time_label,Lx, Ly, ymaxp, bx, by, bz, 'zmaxp',0,0, False, False,'')


#

#os.system("rsync -r "+directory+"/*.png gianni@buteo.colorado.edu:hermes/XY/"+directory)
print('done')
