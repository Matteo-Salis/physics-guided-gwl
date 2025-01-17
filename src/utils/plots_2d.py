import numpy as np
import matplotlib.pyplot as plt
import random
import wandb

def plot_random_station_time_series(y, y_hat, i, save_dir = None, model_name = None, title = None, mode = "training"):

    pz_h_mask = y[0,1,0,:,:]
    avail_idxs = np.argwhere(pz_h_mask)
    idx_stat = random.randint(0, avail_idxs.shape[1]-1)
    coords_station = avail_idxs[:,idx_stat]

    y_plot = y[0,0,:,coords_station[0],coords_station[1]]
    x_y_plot = np.argwhere(y[0,1,:,coords_station[0],coords_station[1]])[0]
    y_hat_plot = y_hat[0,0,:,coords_station[0],coords_station[1]]

    fig, ax = plt.subplots()
    fig.suptitle("Loss vs iterations")
    ax.plot(y_hat_plot, label = "predicted")
    ax.plot(x_y_plot, y_plot, label = "true", marker = "o")
    ax.legend()

    if title is not None:
        ax.set_title(title)

    if save_dir and model_name:
        plt.savefig(f"{save_dir}/timeseries_{model_name}_ep{i}_{mode}.png", bbox_inches = 'tight') # dpi = 400, transparent = True
        wandb.log({
                f"{mode}_timeseries_prediction" :  wandb.Image(f"{save_dir}/timeseries_{model_name}_ep{i}_{mode}.png", caption=f"Prediction series ep{i} {mode}")
            })
        

def plot_2d_prediction(Y_hat, i, plots_dir, timestep, model_name, mode = "training"):
    plt.figure(figsize = (10,10))
    plt.imshow(Y_hat[0,0,timestep,:,:])
    plt.colorbar()
    plt.savefig(f"{plots_dir}/image_t{timestep}_{i}_{model_name}_{mode}.png", bbox_inches = 'tight') #d pi = 400, transparent = True
    wandb.log({
        f"{mode}_image_prediction" :  wandb.Image(f"{plots_dir}/image_t{timestep}_{i}_{model_name}_{mode}.png", caption=f"Prediction image ep{i} t{timestep} {mode}")
    })