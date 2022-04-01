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
        points[i].append(pressure_ini)
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
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.pause(0.0001)

    return 

def calculdt(vx,vy,C_FL)
    vmax_x = np.maximum(vx)
    vmax_y = np.maximum(vy)
    if vmax_x > vmax_y:
        dt = C_FL*dx/vmax_x
    if vmax_y > vmax_y:
        dt = C_FL*dx/vmax_y
    if vmax_x = vmax_y:
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

# Création du maillage pour la méthode de volumes finis (Vince)
#needs meshzoo as mz
#      numpy as np
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
