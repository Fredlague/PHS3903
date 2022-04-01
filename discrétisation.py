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
