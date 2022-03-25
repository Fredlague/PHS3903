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
