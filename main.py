def main():
    gamma = 5/3 
    t=0
    dt = 0.1 #a verifier
    dx = 0.0082644
    tmax = 5
    speed_ini = [1,0]
    pression_ini = 1
    rho_ini = 1
    points = mesher()
    points = front_cond(points)
    points = ini_fluide(points,speed_ini,pression_ini,rho_ini)
    Mass = rho_ini*dx**2
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
    momx =momx.tolist()
    momy = np.resize(momy , (242,121))
    momy =momy.tolist()
    NRG = np.resize(NRG , (242,121))
    NRG =NRG.tolist()
    P = np.resize(P , (242,121))
    P =P.tolist()
    vy = np.resize(vy , (242,121))
    vy =vy.tolist()
    vx = np.resize(vx, (242,121))
    vx =vx.tolist()
    rho = np.resize(rho , (242,121))
    rho =rho.tolist()

    while t < tmax:
        rho_dx, rho_dy = getGradient(rho,dx)
        vx_dx, vx_dy = getGradient(vx,dx)
        vy_dx, vy_dy = getGradient(vy,dx)
        pression_dx, pression_dy = getGradient(P,dx)
        rhoprime = rho - 0.5*dt*(rho_dx*vx+rho*vx_dx+rho_dy*vy+rho*vy_dy)
        vxprime = vx - 0.5*dt*(vx_dx*vx+vy*vx_dy+rho**(-1)*pression_dx)
        vxprime = vy - 0.5*dt*(vy_dx*vx+vy*vy_dy+rho**(-1)*pression_dy)
        pressionprime = P - 0.5*dt*(gammma*P*(vx_dx+vy_dy)+vx*pression_dy+vy*pression_dy)
        rho_L, rho_R, rho_T, rho_B = extrapo(rhoprime, rho_dx, rho_dy, dx)
        vx_L, vx_R, vx_T, vx_B = extrapo(vxprime, vx_dx, vx_dy, dx)
        vy_L, vy_R, vy_T, vy_B = extrapo(vyprime, vy_dx, vy_dy, dx)
        pression_L, pression_R, pression_T, pression_B = extrapo(pressionprime, pression_dx, pression_dy, dx)
        
        
        
