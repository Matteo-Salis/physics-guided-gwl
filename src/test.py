from tqdm import tqdm
from loss.load_losses import *
import wandb
import matplotlib.pyplot as plt

def test_model(i, model, test_loader, wtd_mean, wtd_std, dtm, config, device = "cuda"):
    c0_superres_loss = config["c0_superres_loss"]
    c1_masked_loss = config["c1_masked_loss"]
    c2_pde_darcy_loss = config["c2_pde_darcy_loss"]
    c3_positive_loss = config["c3_positive_loss"]
    Y = None
    
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")

                X = (init_wtd.to(device), dtm.to(device), weather.to(device))
                # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                Y = model(X)
                weather_hd = None

                loss = 0

                if type(Y) is tuple:
                    weather_hd = Y[1]
                    Y = Y[0]

                    if c0_superres_loss:
                        loss_super_res = super_res_loss(weather_hd, weather.to(device))
                        wandb.log({"test_loss_super_res" : loss_super_res})
                        loss = loss + c0_superres_loss * loss_super_res

                if c1_masked_loss:
                    loss_mask = loss_masked(Y,pred_wtds)
                    wandb.log({"test_loss_mask" : loss_mask})
                    loss = loss + c1_masked_loss * loss_mask

                if c2_pde_darcy_loss:
                    loss_pde = pde_grad_loss_darcy(Y)
                    wandb.log({"test_loss_pde" : loss_pde})
                    loss = loss + c2_pde_darcy_loss * loss_pde

                if c3_positive_loss:
                    loss_pos = loss_positive_height(Y, wtd_mean, wtd_std)
                    wandb.log({"test_loss_pos" : loss_pos})
                    loss = loss + c3_positive_loss * loss_pos

                wandb.log({"test_loss" : loss})
                print(f"Test loss: {loss}")
                
        with torch.no_grad():
            predict = (Y.cpu() * wtd_std) + wtd_mean
            plt.figure(figsize = (10,10))
            plt.imshow(predict[0,0,0,:,:])
            plt.colorbar()
            plt.savefig(f"predict_test_a{i}.png", bbox_inches = 'tight')
            wandb.log({
                "test_prediction" :  wandb.Image(f"predict_test_a{i}.png", caption="prediction A on test")
            })

        with torch.no_grad():
            predict = (Y.cpu() * wtd_std) + wtd_mean
            plt.figure(figsize = (10,10))
            plt.imshow(predict[0,0,100,:,:])
            plt.colorbar()
            plt.savefig(f"predict_test_b{i}.png", bbox_inches = 'tight')
            wandb.log({
                "test_prediction" :  wandb.Image(f"predict_test_b{i}.png", caption="prediction B on test")
            })

            