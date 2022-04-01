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
    
