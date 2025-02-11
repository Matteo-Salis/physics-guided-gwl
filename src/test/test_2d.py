from tqdm import tqdm
from loss.losses_2d import *
from utils.plot import *
import wandb
import matplotlib.pyplot as plt

def test_model_2d(epoch, dataset, model, test_loader,
                   c0_superres_loss,
                   c1_masked_loss,
                   c2_pde_darcy_loss,
                   c3_positive_loss,
                   h_timesteps,
                   timesteps,
                   device = "cuda"):
    
    dtm = torch.from_numpy(dataset.dtm_roi_downsampled.values).to(device)

    X = None # input
    Y = None # ground truth
    Y_hat = None # prediction
    
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

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
                    loss_pos = loss_positive_height(Y_hat, dataset.wtd_numpy_mean, dataset.wtd_numpy_std, device)
                    wandb.log({"test_loss_pos" : loss_pos})
                    loss = loss + c3_positive_loss * loss_pos

                wandb.log({"test_loss" : loss})
                print(f"Test loss: {loss}")

        
                
        # plots on wandb
        with torch.no_grad():
            Y[:,0,:,:,:] = (Y[:,0,:,:,:] * dataset.wtd_numpy_std) + dataset.wtd_numpy_mean
            Y_hat = (Y_hat.cpu() * dataset.wtd_numpy_std) + dataset.wtd_numpy_mean

            plot_random_station_time_series(Y, Y_hat, epoch, f"Testing random time series ep:{epoch}", mode = "test")

            plot_2d_prediction(Y_hat, epoch, 0, mode = "test")

            plot_2d_prediction(Y_hat, epoch, h_timesteps, mode = "test")

            plot_2d_prediction(Y_hat, epoch, timesteps-1, mode = "test")


if __name__ == "__main__":
    pass

            