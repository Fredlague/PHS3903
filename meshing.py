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
