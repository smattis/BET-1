import numpy as np
import bet.sample as samp
import bet.sampling.goalOrientedSampling as gos
import bet.postProcess.plotVoronoi as pv
import bet.sampling.basicSampling as bsam
import matplotlib.pyplot as plt
import matplotlib

def plot_hist(disc, x, show=True, save=False, savename = 'hist'):
    N = len(x)
    H, xedges, yedges = np.histogram2d(x[:,0], x[:,1], bins=40, range=[[0.0, 3.0], [-1.0, 3.0]])
    H=H.T
    H = H/float(N)
    fig = plt.figure()
    colormap_type='BuGn'
    cmap = matplotlib.cm.get_cmap(colormap_type)
    plt.imshow(H, interpolation='bicubic', origin='low', cmap=cmap,
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    sample_obj = disc._input_sample_set
    plt.axis([sample_obj._domain[0][0], sample_obj._domain[0][1], sample_obj._domain[1][0], sample_obj._domain[1][1]])
    ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=0.9)
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=np.max(H)), label=r'$P_{\Lambda}(\mathcal{V}_i)$')
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=20)
    text.set_font_properties(font)
    if show:
        plt.show()
    if save:
        plt.savefig(savename + '.png')
    plt.close()

def plot_error_id_voronoi(disc, show=True, save=False, savename = 'error_id', label="error identifier"):
    disc._input_sample_set._probabilities_local = disc._input_sample_set._error_id
    disc._input_sample_set._probabilities = disc._input_sample_set._error_id
    disc._input_sample_set.exact_volume_2D()
    pv.plot_2D_voronoi(disc, density=True, interactive=show, label=label)
    if save:
        plt.savefig(savename + '.png')
    plt.close()

def plot_levels_voronoi(disc, show=True, save=False, savename = 'levels', data_range=None, label='level'):
    disc._input_sample_set._probabilities_local = disc._input_sample_set._levels
    disc._input_sample_set._probabilities = disc._input_sample_set._levels
    #disc._input_sample_set.exact_volume_2D()
    pv.plot_2D_voronoi(disc, density=False, interactive=show, data_range=data_range, label=label)
    if save:
        plt.savefig(savename + '.png')
    plt.close()
  
    
def plot_prob_voronoi(disc, pmeas, show=True, save=False, savename = 'prob'):
    disc._input_sample_set._probabilities_local = pmeas
    disc._input_sample_set._probabilities = pmeas
    disc._input_sample_set.exact_volume_2D()
    pv.plot_2D_voronoi(disc, density=True, interactive=show)
    if save:
        plt.savefig(savename + '.png')
    plt.close()
        
