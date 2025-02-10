from tqdm import tqdm
from loss.load_losses import *
from utils.plots_2d import *
import wandb
import matplotlib.pyplot as plt

def test_model(i, model, test_loader, dataset, dtm, config, model_name, device = "cuda"):
    
    dtm = torch.from_numpy(dataset.dtm_roi_downsampled.values).to(device)
    wtd_mean = dataset.wtd_numpy_mean
    wtd_std = dataset.wtd_numpy_std
    
    c0_superres_loss = config["c0_superres_loss"]
    c1_masked_loss = config["c1_masked_loss"]
    c2_pde_darcy_loss = config["c2_pde_darcy_loss"]
    c3_positive_loss = config["c3_positive_loss"]

    h_timesteps = int(config["timesteps"]/2)
    timesteps = int(config["timesteps"])

    plots_dir = config["wandb_dir_plots"]
    model_name_short = model_name.split(".")[0]

    X = None # input
    Y = None # ground truth
    Y_hat = None # prediction
    
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")

                X = (init_wtd.to(device), dtm.to(device), weather.to(device))
                # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                Y = pred_wtds
                Y_hat = model(X)
                weather_hd = None

                loss = 0

                if type(Y_hat) is tuple:
                    weather_hd = Y_hat[1]
                    Y_hat = Y_hat[0]

                    if c0_superres_loss:
                        loss_super_res = super_res_loss(weather_hd, weather.to(device), device)
                        wandb.log({"test_loss_super_res" : loss_super_res})
                        loss = loss + c0_superres_loss * loss_super_res

                if c1_masked_loss:
                    loss_mask = loss_masked(Y_hat, pred_wtds, device)
                    wandb.log({"test_loss_mask" : loss_mask})
                    loss = loss + c1_masked_loss * loss_mask

                if c2_pde_darcy_loss:
                    loss_pde = pde_grad_loss_darcy(Y_hat, device)
                    wandb.log({"test_loss_pde" : loss_pde})
                    loss = loss + c2_pde_darcy_loss * loss_pde

                if c3_positive_loss:
                    loss_pos = loss_positive_height(Y_hat, wtd_mean, wtd_std, device)
                    wandb.log({"test_loss_pos" : loss_pos})
                    loss = loss + c3_positive_loss * loss_pos

                wandb.log({"test_loss" : loss})
                print(f"Test loss: {loss}")

        
                
        # plots on wandb
        with torch.no_grad():
            Y[:,0,:,:,:] = (Y[:,0,:,:,:] * wtd_std) + wtd_mean
            Y_hat = (Y_hat.cpu() * wtd_std) + wtd_mean

            plot_random_station_time_series(Y, Y_hat, i, plots_dir, model_name_short, f"Testing random time series ep:{i}", mode = "test")

            plot_2d_prediction(Y_hat, i, plots_dir, 0, model_name_short, mode = "test")

            plot_2d_prediction(Y_hat, i, plots_dir, h_timesteps, model_name_short, mode = "test")

            plot_2d_prediction(Y_hat, i, plots_dir, timesteps-1, model_name_short, mode = "test")


            