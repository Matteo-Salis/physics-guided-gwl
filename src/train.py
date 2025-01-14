from tqdm import tqdm
from loss.load_losses import *
import wandb
import matplotlib.pyplot as plt
from torchview import draw_graph


def train_model(i, model, train_loader, optimizer, wtd_mean, wtd_std, dtm, config, model_name, device = "cuda"):
    c0_superres_loss = config["c0_superres_loss"]
    c1_masked_loss = config["c1_masked_loss"]
    c2_pde_darcy_loss = config["c2_pde_darcy_loss"]
    c3_positive_loss = config["c3_positive_loss"]
    X = None
    Y = None
    
    with tqdm(train_loader, unit="batch") as tepoch:
            
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
                    wandb.log({"train_loss_super_res" : loss_super_res})
                    loss = loss + c0_superres_loss * loss_super_res

            if c1_masked_loss:
                loss_mask = loss_masked(Y,pred_wtds)
                wandb.log({"train_loss_mask" : loss_mask})
                loss = loss + c1_masked_loss * loss_mask

            if c2_pde_darcy_loss:
                loss_pde = pde_grad_loss_darcy(Y)
                wandb.log({"train_loss_pde" : loss_pde})
                loss = loss + c2_pde_darcy_loss * loss_pde

            if c3_positive_loss:
                loss_pos = loss_positive_height(Y, wtd_mean, wtd_std)
                wandb.log({"train_loss_pos" : loss_pos})
                loss = loss + c3_positive_loss * loss_pos

            wandb.log({"train_loss" : loss})
            print(f"Train loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # plots on wandb
        with torch.no_grad():
            predict = (Y.cpu() * wtd_std) + wtd_mean
            plt.figure(figsize = (10,10))
            plt.imshow(predict[0,0,0,:,:])
            plt.colorbar()
            plt.savefig(f"predict_a{i}.png", bbox_inches = 'tight')
            wandb.log({
                "train_prediction" :  wandb.Image(f"predict_a{i}.png", caption="prediction A on training")
            })

        with torch.no_grad():
            predict = (Y.cpu() * wtd_std) + wtd_mean
            plt.figure(figsize = (10,10))
            plt.imshow(predict[0,0,100,:,:])
            plt.colorbar()
            plt.savefig(f"predict_b{i}.png", bbox_inches = 'tight')
            wandb.log({
                "train_prediction" :  wandb.Image(f"predict_b{i}.png", caption="prediction B on training")
            })

        if i == 0:
            print("Saving plot of the model...")
            model_file_path = config['save_model_dir']
            model_graph = draw_graph(model, input_data=([X]), device=device)
            model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_file_path}/")
            model_arch = wandb.Image(f"{model_file_path}/{model_name}.png", caption="model's architecture")
            wandb.log({"model_arch": model_arch})

if __name__ == "__main__":
    pass