# Création du maillage pour la méthode de volumes finis

#Discrétization du domaine et géométrie
def maillage(dimension, resolution,largeurFluide): 
    largeurCellule = dimension/resolution
    volumeCellule = largeurCellule**2
    linsp = np.linspace(-0.5*dimension,0.5*dimension,resolution)
    cellX, cellY = np.meshgrid(linsp,linsp)
    fluideCentralY = np.largeurFluide * cellY  
    return cellX, cellY, volumeCellule


# Définition des variables conservées dans les volumes
def variablesCell(cellX,cellY,volumeCellule): 
    massCell = rho * volumeCellule
    qmxCell = rho * vx * volumeCellule
    qmyCell= rho * vy * volumeCellule
    energieCell = densité énergie * rho *volumeCellule
    return(massCell,qmxCell,qmyCell,energieCell)
