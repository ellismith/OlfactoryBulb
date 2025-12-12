import matplotlib.cm as cm

def get_colors_list():
    n_colors = 10
    colors_set1 = [cm.get_cmap('Set1')(i / (n_colors - 1)) for i in range(n_colors)]
    colors_set2 = [cm.get_cmap('Set2')(i / (n_colors - 1)) for i in range(n_colors)]
    return colors_set1 + colors_set2