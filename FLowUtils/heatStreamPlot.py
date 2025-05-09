def heatStreamPlot(vecfield,   timeStepSkip: int = 2, saveFolder: str = "./", 
                  saveName: str = "vector_field_heatstream", colorBarmin=None,colorBarmax=None,redudant:bool=False ):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(saveFolder, exist_ok=True)

    x, y = np.meshgrid(np.arange(vecfield.Xdim), np.arange(vecfield.Ydim))

    for t in range(0, vecfield.time_steps, timeStepSkip):

        #slice_data shape [y,x,2]
        u = vecfield.field[t,:,:, 0]
        v = vecfield.field[t,:,:, 1]

        magnitude = np.sqrt(u**2 + v**2)

        fig, ax = plt.subplots(figsize=(10, 10))

        if colorBarmin is None:
            colorBarmin=magnitude.min()
        if colorBarmax is None:
            colorBarmax=magnitude.max()
 
        
        heatmap = ax.imshow(magnitude, 
                          origin='lower',
                          extent=[0, vecfield.Xdim - 1, 0, vecfield.Ydim - 1],
                          # Available colormaps in matplotlib:
                          # Sequential: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                          # Sequential2: 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'
                          # Diverging: 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                            #   'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                            #   'berlin', 'managua', 'vanimo'
                          # Qualitative: 'Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3'
                          # Misc: 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern'
                          #https://matplotlib.org/stable/users/explain/colors/colormaps.html
                          cmap='coolwarm',  # Change this to any colormap name from above
                          alpha=0.6,
                          vmin=colorBarmin,
                          vmax=colorBarmax)
        
        ax.streamplot(x, y, u, v,
                     color='black', 
                     linewidth=0.25,
                     density=3.0,
                     arrowsize=0.25,  # Reduce arrow size
                     maxlength=1.0)  # Reduce integration length
        current_time = vecfield.getTime(t)
        if redudant ==True:
            plt.colorbar(heatmap, ax=ax, label='Velocity Magnitude', 
                    boundaries=np.linspace(colorBarmin,colorBarmax, 100))
            # ax.set_title(f"Velocity Field at t = {current_time:.3f}")
            # ax.set_xlabel("X-axis")
            # ax.set_ylabel("Y-axis")
            # ax.set_xlim(0, vecfield.Xdim - 1)
            # ax.set_ylim(0, vecfield.Ydim - 1)
        
        ax.set_axis_off()
        save_path = os.path.join(saveFolder, f"{saveName}_t{t:04d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
                