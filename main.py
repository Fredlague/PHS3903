
import numpy as np
import matplotlib.pyplot as plt
import meshzoo as mz
import math

def front_cond(points): #les max et min y = 0 non glissement
    for i in range(0,len(points)):
        if abs(points[i][1]) >= 0.495: #attention cette valeur peut changer en fonction du maillage
            points[i].append(0)
            points[i].append(0)
    return points

def ini_fluide(points,speed_ini,pression_ini,rho_ini): #vitesse des fluides 
    for i in range(0,len(points)):
        if points[i][1] < 0 and len(points[i])==2: #immobile
            points[i].append(0)
            points[i].append(0)
        if points[i][1] >= 0 and len(points[i])==2: #fluide fast 
            points[i].append(speed_ini[0])
            points[i].append(speed_ini[1])
        points[i].append(pression_ini)
        points[i].append(rho_ini)
     
    return points
# Affichage graphique (densité)

#Variables pour l'affichage

def affichage(t,dimFig,dpi,pasTemps,tempsMax,rho):
    frame = 1
    if t<tempsMax:
        fig = plt.figure(figsize=(dimFig,dimFig),dpi = dpi)
        plt.clear()
        plt.imshow(rho)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.pause(pasTemps)
    return 


def calculdt(vx,vy,C_FL,dx):
    vmax_x = np.maximum(vx)
    vmax_y = np.maximum(vy)
    if vmax_x > vmax_y:
        dt = C_FL*dx/vmax_x
    if vmax_y > vmax_y:
        dt = C_FL*dx/vmax_y
    if vmax_x == vmax_y:
        dt = C_FL*dx/vmax_x
    
    return dt

def getGradient(var,dx):
    """
    Calcul d'un gradient 
    var est la matrice de la variable sur laquelle on applique le grad
    dx est la taille de cellule
    var_dx est la matrice de la derivé de la variable dans la direction x
    var_dy est la matrice de la derivé de la variable dans la direction y
    variable indexe de la variable à dériver dans points
    """
    var_dx = ( np.roll(var,-1,axis=0) - np.roll(var,1,axis=0) ) / (2*dx)
    var_dy = ( np.roll(var,-1,axis=1) - np.roll(var,1,axis=1) )/ (2*dx)
    return var_dx, var_dy
  
def extrapo(var,var_dx,var_dy,dx):
    
    var_xL = var - var_dx * dx/2
    var_xL = np.roll(var_xL,-1,axis=0)
    var_xR = var + var_dx * dx/2
  
    var_yT = var - var_dy * dx/2
    var_yT = np.roll(var_yT,-1,axis=1)
    var_yB = var + var_dy * dx/2  
    

    return var_xL, var_xR, var_yT, var_yB

def calculFlux(rho_L,rho_R,vx_L,vx_R,vy_L,vy_R,pression_L,pression_R,gamma):
    flux_masse = 0.5*(rho_L*vx_L+rho_R*vx_R)
    NRG_L = pression_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
    NRG_R = pression_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)
    flux_momx = (0.5*(rho_L*vx_L+rho_R*vx_R))**2/(0.5*(rho_L +rho_R))+(gamma-1)*(0.5*(NRG_L+NRG_R)-0.5*( (0.5*(rho_L*vx_L+rho_R*vx_R))**2+(0.5*(rho_L*vy_L+rho_R*vy_R))**2)/(0.5*(rho_L + rho_R)))
    flux_momy = 0.5*(rho_L*vx_L+rho_R*vx_R)*0.5*(rho_L*vy_L+rho_R*vy_R)/(0.5*(rho_L +rho_R))
    flux_NRG = 0.5*(NRG_L+NRG_R + ((gamma-1)*(0.5*(NRG_L+NRG_R)-0.5*( (0.5*(rho_L*vx_L+rho_R*vx_R))**2+(0.5*(rho_L*vy_L+rho_R*vy_R))**2)/(0.5*(rho_L + rho_R)))))* 0.5*(rho_L*vx_L+rho_R*vx_R)/(0.5*(rho_L +rho_R))
    return flux_masse, flux_NRG, flux_momx, flux_momy

def appFlux(var,dt,dx,flux_var_X,flux_var_Y):
    var = var - dt*dx*flux_var_X
    var = var + dt*dx*np.roll(flux_var_X,-1,axis=0)
    var = var - dt*dx*flux_var_Y
    var = var + dt*dx*np.roll(flux_var_Y,1,axis=1)
    
    return var

def mesher(): #maillage time
    points, cells = mz.rectangle_quad(
        np.linspace(-1.0, 1.0, 242),
        np.linspace(-0.5, 0.5, 121),
        cell_type="quad9"
    )
    #Pour afficher maillage (prends du temps)
    #x=[]
    #y=[]
    #fig, ax = plt.subplots()
    #for i in range(0,len(points)):
    #    x.append(points[i][0])
    #    y.append(points[i][1])
    #ax.scatter(x,y,s=0.2)
    #plt.xlim([-1.1,1.1])
    #plt.ylim([-0.6,0.6])
    #plt.show()
    points= points.tolist() #mieux sans array
    return points

def main():
    gamma = 5/3 
    t=0
    dt = 0.1 #a verifier
    dx = 0.0082644
    tmax = 5
    speed_ini = [1,-0.2]
    pression_ini = 1
    rho_ini = 1
    points = mesher()
    points = front_cond(points)
    points = ini_fluide(points,speed_ini,pression_ini,rho_ini)
    masse = rho_ini*dx**2
    momx=[]
    momy=[]
    NRG=[]
    P = []
    vy =[]
    vx = []
    rho = []
    for i in range(0,len(points)):
        momx.append(rho_ini*points[i][2]*dx**2)
        momy.append(rho_ini*points[i][3]*dx**2) 
        NRG.append((pression_ini/(gamma-1)+0.5*rho_ini*(points[i][2]+points[i][3]**2))*dx**2)
        vx.append(points[i][2])
        vy.append(points[i][3])
        P.append(points[i][4])
        rho.append(points[i][5])
    momx = np.resize(momx , (242,121))
    #momx =momx.tolist()
    momy = np.resize(momy , (242,121))
    #momy =momy.tolist()
    NRG = np.resize(NRG , (242,121))
    #NRG =NRG.tolist()
    P = np.resize(P , (242,121))
    #P =P.tolist()
    vy = np.resize(vy , (242,121))
    for i in range(0,len(vy)):
        for j in range(0, len(vy[0])):
            x = (i - len(vy)/2)/50 # 50 est valeur test pour la pente de gaussienn
            vy[i][j] = vy[i][j] * math.exp(-(x**2)/0.1) #impusle gauss
            
    #vy =vy.tolist()
    vx = np.resize(vx, (242,121))
    #vx =vx.tolist()
    rho = np.resize(rho , (242,121))
    #rho =rho.tolist()



    while t < tmax:
        rho_dx, rho_dy = getGradient(rho,dx)
        vx_dx, vx_dy = getGradient(vx,dx)
        vy_dx, vy_dy = getGradient(vy,dx)
        pression_dx, pression_dy = getGradient(P,dx)
        
        rhoprime = rho - 0.5*dt*(rho_dx*vx+rho*vx_dx+rho_dy*vy+rho*vy_dy)
        vxprime = vx - 0.5*dt*(vx_dx*vx+vy*vx_dy+(1/rho)*pression_dx)
        vyprime = vy - 0.5*dt*(vy_dx*vx+vy*vy_dy+(1/rho)*pression_dy)
        pressionprime = P - 0.5*dt*(gamma*P*(vx_dx+vy_dy)+vx*pression_dy+vy*pression_dy)
        
        rho_L, rho_R, rho_T, rho_B = extrapo(rhoprime, rho_dx, rho_dy, dx)
        vx_L, vx_R, vx_T, vx_B = extrapo(vxprime, vx_dx, vx_dy, dx)
        vy_L, vy_R, vy_T, vy_B = extrapo(vyprime, vy_dx, vy_dy, dx)
        pression_L, pression_R, pression_T, pression_B = extrapo(pressionprime, pression_dx, pression_dy, dx)

        flux_masse_X, flux_momx_X, flux_momy_X, flux_NRG_X = calculFlux(rho_L,rho_R,vx_L,vx_R,vy_L,vy_R,pression_L, pression_R,gamma)
        flux_masse_Y, flux_momx_Y, flux_momy_Y, flux_NRG_Y = calculFlux(rho_T,rho_B,vx_T,vx_B,vy_T,vy_B,pression_T, pression_B,gamma)

        masse = appFlux(masse,dt,dx, flux_masse_X, flux_masse_Y)
        momx = appFlux(momx,dt,dx, flux_momx_X, flux_momx_Y)
        momy = appFlux(momy,dt,dx, flux_momy_X, flux_momy_Y)
        NRG = appFlux(NRG,dt,dx, flux_NRG_X, flux_NRG_Y)

        t=t+1


        plt.cla()
        plt.imshow(rho.T)
        plt.clim(0.8, 2.2)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)	
        ax.set_aspect('equal')	
        plt.pause(0.001)

    plt.savefig('finitevolume.png',dpi=240)
    plt.show()


main()
