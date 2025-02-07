import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time
import tqdm
import wandb

from torchview import draw_graph

from dataloaders.dataset_1d import *
from utils.plot_1d import *
from loss.losses_1d import *
    
    
def test_dl_model_1d(epoch, ds, model, test_loader, device = "cuda"):
    
    with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {epoch}")

                                x = x.to(device)
                                x_mask = x_mask.to(device)
                                z = z.to(device)
                                weather_coords_batch = ds.get_weather_coords(dtm = True).unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                                w = [w_values.to(device), weather_coords_batch.to(device)]
                                y = y.to(device)
                                y_mask = y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                y_hat = model(x, z, w, x_mask)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss = masked_mse(y_hat,
                                              y,
                                              y_mask)
                                
                                print("Test_loss: ", loss.item())
                                wandb.log({"Test_loss":loss.item()})
                                
                            plot_series_maps(ds, model, device, 
                                    dates_list = dict_files["test_plot_dates"],
                                    tsteps_list= dict_files["test_plot_tstep_map"])  
                                
                                
                    



            


def train_dl_model(ds, model, train_loader, test_loader,
                   optimizer, model_name,
                   epochs, device):
    
    
        weather_coords = ds.get_weather_coords(dtm = True)

        print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
        for i in range(epochs):
            
            model.train(True)
            start_time = time.time()
            print(f"############### Training epoch {i} ###############")
            
            with tqdm(train_loader, unit="batch") as tepoch:
                    for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {i}")
                        
                        x = x.to(device)
                        x_mask = x_mask.to(device)
                        z = z.to(device)
                        weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                        w = [w_values.to(device), weather_coords_batch.to(device)]
                        y = y.to(device)
                        y_mask = y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        y_hat = model(x, z, w, x_mask)
                        
                        #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        loss = masked_mse(y_hat,
                                          y,
                                          y_mask)
                        
                        print("Training_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        # metrics = {
                        #     "train_loss" : loss
                        # }
                        wandb.log({"Training_loss":loss.item()})              
                        
            end_time = time.time()
            exec_time = end_time-start_time

            wandb.log({"tr_epoch_exec_t" : exec_time})
            
            model_dir = dict_files["save_model_dir"]
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

            print(f"############### Test epoch {i} ###############")
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()
            start_time = time.time()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {i}")

                                x = x.to(device)
                                x_mask = x_mask.to(device)
                                z = z.to(device)
                                weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                                w = [w_values.to(device), weather_coords_batch.to(device)]
                                y = y.to(device)
                                y_mask = y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                y_hat = model(x, z, w, x_mask)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss = masked_mse(y_hat,
                                              y,
                                              y_mask)
                                
                                print("Test_loss: ", loss.item())
                                wandb.log({"Test_loss":loss.item()}) 
                    
            end_time = time.time()
            exec_time = end_time-start_time
            wandb.log({"test_epoch_exec_t" : exec_time})

            # Plots
            train_test_plots(ds, model, device, 
                             dates_list = dict_files["plot_dates"],
                             tsteps_list= dict_files["plot_tstep_map"])

        model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
        model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
        model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
        wandb.log({"model_arch": model_arch})
                        
        wandb.finish()

        print(f"Execution time: {end_time-start_time}s")
        
        
#######################
## ONLY PDE TRAINING ##
#######################

# cpoint generation come dataset e dataloader? 

def train_pde_model(ds, model, train_loader, test_loader,
                   optimizer, model_name,
                   epochs, device,
                   num_cpoints,
                   num_ctsteps,
                   S_y,):
    
    
        weather_coords = ds.get_weather_coords(dtm = True)

        print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
        for i in range(epochs):
            
            model.train(True)
            start_time = time.time()
            print(f"############### Training epoch {i} ###############")
            
            with tqdm(train_loader, unit="batch") as tepoch:
                    for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {i}")
                        
                        x = x.to(device)
                        x_mask = x_mask.to(device)
                        z = z.to(device)
                        weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                        w = [w_values.to(device), weather_coords_batch.to(device)]
                        y = y.to(device)
                        y_mask = y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        y_hat = model(x, z, w, x_mask)
                        
                        #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        loss = masked_mse(y_hat,
                                          y,
                                          y_mask)
                        
                        print("Training_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        # metrics = {
                        #     "train_loss" : loss
                        # }
                        wandb.log({"Training_loss":loss.item()})              
                        
            end_time = time.time()
            exec_time = end_time-start_time

            wandb.log({"tr_epoch_exec_t" : exec_time})
            
            model_dir = dict_files["save_model_dir"]
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

            print(f"############### Test epoch {i} ###############")
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()
            start_time = time.time()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {i}")

                                x = x.to(device)
                                x_mask = x_mask.to(device)
                                z = z.to(device)
                                weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                                w = [w_values.to(device), weather_coords_batch.to(device)]
                                y = y.to(device)
                                y_mask = y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                y_hat = model(x, z, w, x_mask)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss = masked_mse(y_hat,
                                              y,
                                              y_mask)
                                
                                print("Test_loss: ", loss.item())
                                wandb.log({"Test_loss":loss.item()}) 
                    
            end_time = time.time()
            exec_time = end_time-start_time
            wandb.log({"test_epoch_exec_t" : exec_time})

            # Plots
            train_test_plots(ds, model, device, 
                             dates_list = dict_files["plot_dates"],
                             tsteps_list= dict_files["plot_tstep_map"])

        model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
        model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
        model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
        wandb.log({"model_arch": model_arch})
                        
        wandb.finish()

        print(f"Execution time: {end_time-start_time}s")
        
        

#######################
## PINNS TRAINING ##
#######################

def train_pde_model(model, train_loader, test_loader,
                   optimizer, model_name,
                   epochs, device):
    
    
        weather_coords = ds.get_weather_coords()
        weather_dtm = ds.get_weather_dtm()
        weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)

        print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
        for i in range(epochs):
            
            model.train(True)
            start_time = time.time()
            print(f"############### Training epoch {i} ###############")
            
            with tqdm(train_loader, unit="batch") as tepoch:
                    for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {i}")
                        
                        x = x.to(device)
                        x_mask = x_mask.to(device)
                        z = z.to(device)
                        weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                        w = [w_values.to(device), weather_coords_batch.to(device)]
                        y = y.to(device)
                        y_mask = y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        y_hat = model(x, z, w, x_mask)
                        
                        #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        loss = masked_mse(y_hat,
                                          y,
                                          y_mask)
                        
                        print("Training_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        # metrics = {
                        #     "train_loss" : loss
                        # }
                        wandb.log({"Training_loss":loss.item()})              
                        
            end_time = time.time()
            exec_time = end_time-start_time

            wandb.log({"tr_epoch_exec_t" : exec_time})
            
            model_dir = dict_files["save_model_dir"]
            torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

            print(f"############### Test epoch {i} ###############")
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()
            start_time = time.time()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {i}")

                                x = x.to(device)
                                x_mask = x_mask.to(device)
                                z = z.to(device)
                                weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                                w = [w_values.to(device), weather_coords_batch.to(device)]
                                y = y.to(device)
                                y_mask = y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                y_hat = model(x, z, w, x_mask)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss = masked_mse(y_hat,
                                              y,
                                              y_mask)
                                
                                print("Test_loss: ", loss.item())
                                wandb.log({"Test_loss":loss.item()}) 
                    
            end_time = time.time()
            exec_time = end_time-start_time
            wandb.log({"test_epoch_exec_t" : exec_time})

            # Plots
            train_test_plots(ds, model, device, 
                             dates_list = dict_files["plot_dates"],
                             tsteps_list= dict_files["plot_tstep_map"])

        model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
        model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
        model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
        wandb.log({"model_arch": model_arch})
                        
        wandb.finish()

        print(f"Execution time: {end_time-start_time}s")