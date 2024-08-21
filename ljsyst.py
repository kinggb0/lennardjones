######################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.animation import FuncAnimation
#####################################################
#Initialize the particles 
def initialize(npart,length,radius,v0):

   #should have ~sqrt(npart) particles / direction
   points = int(np.ceil(np.sqrt(npart)))
   dl = length / points #grid spacing to fit all particles
   #slightly off the walls so that you are not out of the box
   x = np.linspace(radius + dl/2, length - radius - dl/2, points) 
   sites = list(product(x,x)) #makes tuples like (x0,x0), (x1, x0) ... (x0,xn), (x1,x0), (x1, x1), ... (x1,xn) ... (xn,xn)
   pos = np.asarray(sites[:npart]) #takes the first npart sites generated
   
   #initialize npart random directions
   theta = 2. *np.pi * np.random.uniform(size=npart)
   vx, vy = v0*np.cos(theta), v0*np.sin(theta) # 
   vel = np.stack((vx,vy),axis=-1)

   #return the positions and velocities of the particles
   return pos, vel

#compute length of vector
def norm(vect):
   
   # sqrt (sum_i v_i**2)
   return np.sqrt(np.sum(vect**2.))

#LJ force, \vect{F}_{LJ}(r) = (24/r**8 - 48/r**14) * \vect{r}
def force(rij,cut):

   r = norm(rij)
    
   #If particles are within a cutoff distance, compute LJ force
   if (r < cut):
      fij = (24./r**8. - 48./r**14.)*rij 
   else:
      fij = np.zeros(2) #F(rij) > cutoff) = 0

   return fij

#compute the forces on all of the particles
def computeforces(pos,vel,npart,L,cut):

   forces = np.zeros((npart,2))

   for i in range(npart):
      for j in range(i+1,npart):
         rij = pos[j] - pos[i] #difference vector
         rij = rij - np.rint(rij/L) * L #apply PBC
         fij = force(rij,cut)
         forces[i] += fij #add force to particle i
         forces[j] -= fij #add opposite force to particle j         

   return forces

#update positions and velocities using velocity Verlet algorithm
def update(pos,vel,dt,npart,L,cut,mass):
   
   #get forces at current time
   forces = computeforces(pos,vel,npart,L,cut)
   #update positions with velocity Verlet
   pos = pos + vel*dt + dt**2./(2.*mass) * forces
   pos = pos - np.floor(pos/L)*L #apply PBC
   #forces at next time step
   nextforces = computeforces(pos,vel,npart,L,cut)
   vel = vel + 0.5 * dt / mass * (nextforces + forces)

   return pos, vel

def pot(pos,npart,L,cut):

   eng = 0.

   for i in range(npart):
      for j in range(i+1,npart):
         rij = pos[j] - pos[i]
         rij = rij - np.rint(rij/L) * L #apply PBC
         r = norm(rij)
         if ( r < cut ):
            eng += 4.*(1./r**12. - 1./r**6.)
   return eng

def kinetic(mass,vel):

   return mass/2. * np.sum(vel**2.)

#compute kinetic energy
def energy(mass,vel,pos,npart,L,cut):

   return kinetic(mass,vel) + pot(pos,npart,L,cut)

def mb(v,temp,mass):

   return  (mass/temp) * v * np.exp(-v**2./(2.*temp))

#####################################################

npart = 400 #number of particles 
mass = 1 #1 kg, units are gonna be weird for now
radius = 0.2 #m, radius of the particles for vizulization
cutoff = 2.5 #m, LJ cutoff distance 
length = 20. #m, large enough to comfortably fit the particles
dt = 0.005 #s, time step
v0 = 5. #m/s, initial velocity
niter = 400 #number of iterations for the simulation
duration = 10  #s, irl time for animation

#####################################################
### RUN THE SIMULATION ##############################
#####################################################
allpos = np.zeros((niter+1,npart,2)) #store all positions in the sim
allvel = np.zeros((niter+1,npart,2)) #store all velocities in the sim
alleng = np.zeros(niter+1) #store all total energies
allkin = np.zeros(niter+1) #store all kinetic energies
allpot = np.zeros(niter+1) #store all potential energies

pos,vel = initialize(npart,length,radius,v0)

allpos[0] = pos
allvel[0] = vel
allkin[0] = kinetic(mass,vel)
allpot[0] = pot(pos,npart,length,cutoff)
alleng[0] = allkin[0] + allpot[0]

for i in range(1,niter+1):

   pos, vel = update(pos,vel,dt,npart,length,cutoff,mass)
   allpos[i],allvel[i] = pos,vel
   allpot[i] = pot(pos,npart,length,cutoff)
   allkin[i] = kinetic(mass,vel)
   alleng[i] = allpot[i] + allkin[i]
   print("energy = %.5f" % alleng[i])
   if( np.abs(alleng[i] - alleng[i-1]) > 0.1*alleng[i-1]): 
      print("WARNING: ENERGY MAY NOT BE CONSERVED IN THIS SIMULATION")


steps = np.arange(0,niter+1,1)

#plot energy, KE, and PE as a function of time to check energy conservation
fig1 = plt.figure(figsize=(12,8))
plt.scatter(steps,allkin,label='Kinetic')
plt.scatter(steps,allpot,label='Potential')
plt.scatter(steps,alleng,label='Energy')
plt.xlabel(r'Time step',fontsize=50)
plt.ylabel('Energy (arb)',fontsize=50)
plt.tick_params(axis='both',direction='in',length=4,width=1,labelsize=35)
plt.legend(loc='best',fontsize='35')
plt.tight_layout()
plt.show()

#compute temperature and maxwell-boltzmann distribution
temp = np.mean(allkin[200:])/npart #kB*T
vrange = np.arange(0.,10.,0.01)
mbdist = mb(vrange,temp,mass)

#####################################################
### MAKE THE ANIMATION ##############################
#####################################################

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))

def animategas(frame):
   ax1.clear()
   ax2.clear()
   for i in range(npart):
      x,y = allpos[frame,i,0], allpos[frame,i,1] #get x and y of particle i
      circle = plt.Circle((x,y), radius, fill = True) #make a circle of a given radius
      ax1.add_artist(circle)

   ax1.set_xlabel('$x$',fontsize=30)
   ax1.set_ylabel('$y$',fontsize=30) 
   ax1.set_xlim([0.,length])
   ax1.set_ylim([0.,length])
   ax1.set_xticks([])
   ax1.set_yticks([])

   speeds = [norm(velocity) for velocity in allvel[frame]]
  
   ax2.hist(speeds,bins=50,range=(0,10),density=True,label='Speeds') #histogram velocity dist normalized to 1
   ax2.plot(vrange,mbdist,lw=4,label='Maxwell-Boltzmann')
   ax2.set_xlabel('$v$ (arb)',fontsize=30)
   ax2.set_ylabel('$f(v)$',fontsize=30) 
   ax2.tick_params(axis='both',direction='in',length=4,width=1,labelsize=30)
   ax2.legend(loc='best',fontsize=15)

ani = FuncAnimation(fig, animategas, frames=niter+1, interval=1000*duration/(niter+1))
plt.show()

