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
    
  var_xl = var - var_dx * dx/2
  var_xl = np.roll(var_xl,-1,axis=0)
  var_xr = var + var_dx * dx/2
  
  var_yl = var - var_dy * dx/2
  var_yl = np.roll(var_yl,-1,axis=1)
  var_yr = var + var_dy * dx/2  
    

    return var_xl, var_xr, var_yl, var_yr
