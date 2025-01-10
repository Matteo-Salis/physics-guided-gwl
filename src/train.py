from tqdm import tqdm
from loss.load_losses import *
import wandb
import matplotlib.pyplot as plt
from torchview import draw_graph


def train_model(i, model, train_loader, optimizer, wtd_mean, wtd_std, config, model_name, device = "cuda"):
    c1_loss = config["c1_loss"]
    c2_loss = config["c2_loss"]
    c3_loss = config["c3_loss"]
    X = None
    Y = None
    
    with tqdm(train_loader, unit="batch") as tepoch:
            
        for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):

            tepoch.set_description(f"Epoch {i}")

            X = (init_wtd.to(device), weather.to(device))
            # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

            Y = model(X)
            # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)clea

            loss_mask = loss_masked(Y,pred_wtds)
            loss_pde = pde_grad_loss_darcy(Y)
            loss_pos = loss_positive_height(Y, wtd_mean, wtd_std)
            loss = c1_loss * loss_mask + c2_loss * loss_pde + c3_loss * loss_pos
            print(f"Train loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {
                "train_loss_mask" : loss_mask,
                "train_loss_pde" : loss_pde,
                "train_loss_pos" : loss_pos,
                "train_loss" : loss
            }

            wandb.log(metrics)
        
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