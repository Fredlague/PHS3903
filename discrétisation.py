def getGradient(var,dx):
  """
  Calcul d'un gradient de flux
  f est la matrice de champs
  dx est la taille de cellule
  f_dx est la matrice de la derivé de f dans la direction x
  f_dy est la matrice de la derivé de f dans la direction y
  variable indexe de la variable à dériver dans points
  position est l'indexe de la position
  """

  var_dx = ( np.roll(var,-1,axis=0) - np.roll(var,1,axis=0) ) / (2*dx)
  var_dy = ( np.roll(var,-1,axis=1) - np.roll(var,1,axis=1) )/ (2*dx)
  
  return var_dx, var_dy
  
  def extrapo(var,var_dx,var_dy,dx):
    
  var_xl = var - var_dx * dx/2
  var_xl = np.roll(var_XL,-1,axis=0)
  var_xr = var + var_dx * dx/2
  
  var_yl = var - var_dy * dx/2
  var_yl = np.roll(var_YL,-1,axis=1)
  var_yr = var + var_dy * dx/2  
    

    return var_xl, var_xr, var_yl, var_yr
