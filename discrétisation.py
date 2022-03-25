def getGradient(points,var,posx,posy,dx):
  """
  Calcul d'un gradient de flux
  f est la matrice de champs
  dx est la taille de cellule
  f_dx est la matrice de la derivé de f dans la direction x
  f_dy est la matrice de la derivé de f dans la direction y
  variable indexe de la variable à dériver dans points
  position est l'indexe de la position
  """

  f_dx = ( np.roll(points,-1,axis=0) - np.roll(f,1,axis=0) ) / (2*dx)
  f_dy = ( np.roll(points,-1,axis=1) - np.roll(f,1,axis=1) )/ (2*dx)
  
  return f_dx, f_dy
  
  def extrapo(points,pos,var,f_dx,f_dy,dx):
    
  f_xl = f - f_dx * dx/2
  f_xl = np.roll(f_XL,-1,axis=0)
  f_xr = f + f_dx * dx/2
  
  f_yl = f - f_dy * dx/2
  f_yl = np.roll(f_YL,-1,axis=1)
  f_yr = f + f_dy * dx/2  
    

    return f_xl, f_xr, f_yl, f_yr
