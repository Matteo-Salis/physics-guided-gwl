import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time
from tqdm import tqdm
import wandb

from torchview import draw_graph

from dataloaders.dataset_1d import *
from utils.plot import *
from loss.losses_1d import *


def train_dl_model_1d(epoch, dataset, model, train_loader, optimizer, model_dir, model_name,
                      dates_list, tsteps_list, device = "cuda"):
    
    with tqdm(train_loader, unit="batch") as tepoch:
                    
                    for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {epoch}")
                        
                        x = x.to(device)
                        x_mask = x_mask.to(device)
                        z = z.to(device)
                        weather_coords_batch = dataset.weather_coords_dtm.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
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
                        
                        wandb.log({"Training_loss":loss.item()})   
                        
                    # Plots
                    with torch.no_grad():
                        plot_series_maps(dataset, model, device, 
                        dates_list = dates_list,
                        tsteps_list= tsteps_list)
                        
                        if epoch == 0:
                            print("Saving plot of the model...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (dataset[0][0], dataset[0][1],
                                                                       [dataset[0][2], dataset.get_weather_coords(dtm = True)],
                                                                       dataset[0][4]),
                                                                      device = device)})
                        
                        
def train_dl_pde_model_1d(epoch, dataset, model, train_loader, optimizer,
                          model_dir, model_name,
                          num_cpoint_batch,
                          num_cpoint_instance,
                          dates_list,
                          tsteps_list,
                          g =  torch.tensor([0]), S_y =  torch.tensor([1]),
                          fdif_step = 0.0009,
                          device = "cuda",
                          coeff_data_loss = 1,
                          coeff_pde_loss = 1):
    
    g = g.to(device)
    S_y = S_y.to(device)
    
    with tqdm(train_loader, unit="batch") as tepoch:
                    
                    for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {epoch}")
                        
                        x = x.to(device)
                        x_mask = x_mask.to(device)
                        z = z.to(device)
                        weather_coords_batch = dataset.weather_coords_dtm.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                        w = [w_values.to(device), weather_coords_batch.to(device)]
                        y = y.to(device)
                        y_mask = y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        y_hat = model(x, z, w, x_mask)
                        
                        #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        loss_data = masked_mse(y_hat,
                                          y,
                                          y_mask)
                        
                        print("Training_data_loss: ", loss_data.item(), end = " --- ")
                        
                        ## Control Points Generation
                        sample_idx = torch.randint(0,x.shape[0],
                                                 (num_cpoint_batch,))
                        
                        x_cpoints = x[sample_idx,:,:]#.unsqueeze(0)
                        x_mask_cpoints = x_mask[sample_idx,:]#.unsqueeze(0)
                        
                        w_cpoints = [w[0][sample_idx,:,:,:,:],#.unsqueeze(0).to(device),
                                     w[1][sample_idx,:,:,:]]#.unsqueeze(0).to(device)]
                        
                        x_cpoints = torch.repeat_interleave(x_cpoints, num_cpoint_instance*9, dim=0).to(device) # repeat instances 
                        x_mask_cpoints = torch.repeat_interleave(x_mask_cpoints, num_cpoint_instance*9, dim=0).to(device)
                        w_cpoints = [torch.repeat_interleave(w_cpoints[0], num_cpoint_instance*9, dim=0).to(device),
                                     torch.repeat_interleave(w_cpoints[1], num_cpoint_instance*9, dim=0).to(device)]
                        
                        
                        z_cpoints = np.stack([dataset.control_points_generator(
                                 mode = "urandom+nb",
                                 num_lon_point = num_cpoint_instance,
                                 num_lat_point = num_cpoint_instance,
                                 step = fdif_step) for i in range(num_cpoint_batch)], axis = 0)
                        
                        z_cpoints = torch.tensor(z_cpoints, requires_grad=True).to(torch.float32).to(device) # (num_cpoint_batch, num_cpoint_instance, 3, 9)
                        
                        print("before mva", z_cpoints.shape)
                        z_cpoints = z_cpoints.moveaxis(-1, 2).flatten(start_dim=0, end_dim=2)  # flatten cpoints as instances in the batch
                        print("after mva", z_cpoints.shape)
                        
                        y_hat_pde, hydro_cond_hat = model(x_cpoints, z_cpoints, w_cpoints, x_mask_cpoints, hc_out = True)
                        
                        y_hat_pde = y_hat_pde.reshape(num_cpoint_batch, num_cpoint_instance, 9, y_hat.shape[-1]) ##(num_cpoint_batch, num_cpoint_instance, 9, 180)
                        hydro_cond_hat = hydro_cond_hat.reshape(num_cpoint_batch, num_cpoint_instance, 9, 2) #(num_cpoint_batch, num_cpoint_instance, 9, 2)
                        
                        loss_pde = disc_physics_loss(y_hat = y_hat_pde[:,:,0,:],
                                                    y_hat_two_right = y_hat_pde[:,:,5,:], y_hat_two_left = y_hat_pde[:,:,6,:],
                                                    y_hat_two_up = y_hat_pde[:,:,7,:], y_hat_two_down = y_hat_pde[:,:,8,:],
                                                    k_lon_right = hydro_cond_hat[:,:,1,1].unsqueeze(-1).expand(-1,-1,y_hat.shape[-1]),
                                                    k_lon_left = hydro_cond_hat[:,:,2,1].unsqueeze(-1).expand(-1,-1,y_hat.shape[-1]),
                                                    k_lat_up = hydro_cond_hat[:,:,3,0].unsqueeze(-1).expand(-1,-1,y_hat.shape[-1]),
                                                    k_lat_down = hydro_cond_hat[:,:,4,0].unsqueeze(-1).expand(-1,-1,y_hat.shape[-1]),
                                                    step = fdif_step,
                                                    g = g,
                                                    S_y = S_y)
                        
                        # coords, coords_right, coords_left, coords_up, coords_down,
                        #coords_two_right, coords_two_left, coords_two_up, coords_two_down
                        
                        ### Print and Backward
                        print("Training_pde_loss: ", loss_pde.item(), end = " --- ")
                        
                        tot_loss = coeff_data_loss * loss_data + coeff_pde_loss * loss_pde
                        print("Training_Total_loss: ", tot_loss.item())
                        
                        tot_loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Training_data_loss": loss_data.item(),
                                   "Training_pde_loss": loss_pde.item(),
                                   "Training_Total_loss": tot_loss.item()})   

                        
                    # Plots
                    with torch.no_grad():
                        plot_series_maps(dataset, model, device, 
                        dates_list = dates_list,
                        tsteps_list = tsteps_list)
                        
                        if epoch == 0:
                            print("Saving plot of the model...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (dataset[0][0], dataset[0][1],
                                                                       [dataset[0][2], dataset.get_weather_coords(dtm = True)],
                                                                       dataset[0][4]),
                                                                      device = device)})

if __name__ == "__main__":
    pass

# def train_dl_model(ds, model, train_loader, test_loader,
#                    optimizer, model_name,
#                    epochs, device):
    
    
#         weather_coords = ds.get_weather_coords(dtm = True)

#         print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#         print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
#         for i in range(epochs):
            
#             model.train(True)
#             start_time = time.time()
#             print(f"############### Training epoch {i} ###############")
            
#             with tqdm(train_loader, unit="batch") as tepoch:
#                     for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                         tepoch.set_description(f"Epoch {i}")
                        
#                         x = x.to(device)
#                         x_mask = x_mask.to(device)
#                         z = z.to(device)
#                         weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                         w = [w_values.to(device), weather_coords_batch.to(device)]
#                         y = y.to(device)
#                         y_mask = y_mask.to(device)
#                         #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
#                         optimizer.zero_grad()
                        
#                         y_hat = model(x, z, w, x_mask)
                        
#                         #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#                         loss = masked_mse(y_hat,
#                                           y,
#                                           y_mask)
                        
#                         print("Training_loss: ", loss.item())
                        
#                         loss.backward()
#                         optimizer.step()
                        
#                         # metrics = {
#                         #     "train_loss" : loss
#                         # }
#                         wandb.log({"Training_loss":loss.item()})              
                        
#             end_time = time.time()
#             exec_time = end_time-start_time

#             wandb.log({"tr_epoch_exec_t" : exec_time})
            
#             model_dir = dict_files["save_model_dir"]
#             torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

#             print(f"############### Test epoch {i} ###############")
#             # Set the model to evaluation mode, disabling dropout and using population
#             # statistics for batch normalization.
#             model.eval()
#             start_time = time.time()
#             # Disable gradient computation and reduce memory consumption.
#             with torch.no_grad():
#                 with tqdm(test_loader, unit="batch") as tepoch:
#                             for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                                 tepoch.set_description(f"Epoch {i}")

#                                 x = x.to(device)
#                                 x_mask = x_mask.to(device)
#                                 z = z.to(device)
#                                 weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                                 w = [w_values.to(device), weather_coords_batch.to(device)]
#                                 y = y.to(device)
#                                 y_mask = y_mask.to(device)
#                                 #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 y_hat = model(x, z, w, x_mask)
                                
#                                 #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 loss = masked_mse(y_hat,
#                                               y,
#                                               y_mask)
                                
#                                 print("Test_loss: ", loss.item())
#                                 wandb.log({"Test_loss":loss.item()}) 
                    
#             end_time = time.time()
#             exec_time = end_time-start_time
#             wandb.log({"test_epoch_exec_t" : exec_time})

#             # Plots
#             train_test_plots(ds, model, device, 
#                              dates_list = dict_files["plot_dates"],
#                              tsteps_list= dict_files["plot_tstep_map"])

#         model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
#         model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
#         model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
#         wandb.log({"model_arch": model_arch})
                        
#         wandb.finish()

#         print(f"Execution time: {end_time-start_time}s")
        
        
# #######################
# ## ONLY PDE TRAINING ##
# #######################

# # cpoint generation come dataset e dataloader? 

# def train_pde_model(ds, model, train_loader, test_loader,
#                    optimizer, model_name,
#                    epochs, device,
#                    num_cpoints,
#                    num_ctsteps,
#                    S_y,):
    
    
#         weather_coords = ds.get_weather_coords(dtm = True)

#         print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#         print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
#         for i in range(epochs):
            
#             model.train(True)
#             start_time = time.time()
#             print(f"############### Training epoch {i} ###############")
            
#             with tqdm(train_loader, unit="batch") as tepoch:
#                     for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                         tepoch.set_description(f"Epoch {i}")
                        
#                         x = x.to(device)
#                         x_mask = x_mask.to(device)
#                         z = z.to(device)
#                         weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                         w = [w_values.to(device), weather_coords_batch.to(device)]
#                         y = y.to(device)
#                         y_mask = y_mask.to(device)
#                         #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
#                         optimizer.zero_grad()
                        
#                         y_hat = model(x, z, w, x_mask)
                        
#                         #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#                         loss = masked_mse(y_hat,
#                                           y,
#                                           y_mask)
                        
#                         print("Training_loss: ", loss.item())
                        
#                         loss.backward()
#                         optimizer.step()
                        
#                         # metrics = {
#                         #     "train_loss" : loss
#                         # }
#                         wandb.log({"Training_loss":loss.item()})              
                        
#             end_time = time.time()
#             exec_time = end_time-start_time

#             wandb.log({"tr_epoch_exec_t" : exec_time})
            
#             model_dir = dict_files["save_model_dir"]
#             torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

#             print(f"############### Test epoch {i} ###############")
#             # Set the model to evaluation mode, disabling dropout and using population
#             # statistics for batch normalization.
#             model.eval()
#             start_time = time.time()
#             # Disable gradient computation and reduce memory consumption.
#             with torch.no_grad():
#                 with tqdm(test_loader, unit="batch") as tepoch:
#                             for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                                 tepoch.set_description(f"Epoch {i}")

#                                 x = x.to(device)
#                                 x_mask = x_mask.to(device)
#                                 z = z.to(device)
#                                 weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                                 w = [w_values.to(device), weather_coords_batch.to(device)]
#                                 y = y.to(device)
#                                 y_mask = y_mask.to(device)
#                                 #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 y_hat = model(x, z, w, x_mask)
                                
#                                 #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 loss = masked_mse(y_hat,
#                                               y,
#                                               y_mask)
                                
#                                 print("Test_loss: ", loss.item())
#                                 wandb.log({"Test_loss":loss.item()}) 
                    
#             end_time = time.time()
#             exec_time = end_time-start_time
#             wandb.log({"test_epoch_exec_t" : exec_time})

#             # Plots
#             train_test_plots(ds, model, device, 
#                              dates_list = dict_files["plot_dates"],
#                              tsteps_list= dict_files["plot_tstep_map"])

#         model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
#         model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
#         model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
#         wandb.log({"model_arch": model_arch})
                        
#         wandb.finish()

#         print(f"Execution time: {end_time-start_time}s")
        
        

# #######################
# ## PINNS TRAINING ##
# #######################

# def train_pde_model(model, train_loader, test_loader,
#                    optimizer, model_name,
#                    epochs, device):
    
    
#         weather_coords = ds.get_weather_coords()
#         weather_dtm = ds.get_weather_dtm()
#         weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)

#         print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#         print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
#         for i in range(epochs):
            
#             model.train(True)
#             start_time = time.time()
#             print(f"############### Training epoch {i} ###############")
            
#             with tqdm(train_loader, unit="batch") as tepoch:
#                     for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                         tepoch.set_description(f"Epoch {i}")
                        
#                         x = x.to(device)
#                         x_mask = x_mask.to(device)
#                         z = z.to(device)
#                         weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                         w = [w_values.to(device), weather_coords_batch.to(device)]
#                         y = y.to(device)
#                         y_mask = y_mask.to(device)
#                         #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
#                         optimizer.zero_grad()
                        
#                         y_hat = model(x, z, w, x_mask)
                        
#                         #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#                         loss = masked_mse(y_hat,
#                                           y,
#                                           y_mask)
                        
#                         print("Training_loss: ", loss.item())
                        
#                         loss.backward()
#                         optimizer.step()
                        
#                         # metrics = {
#                         #     "train_loss" : loss
#                         # }
#                         wandb.log({"Training_loss":loss.item()})              
                        
#             end_time = time.time()
#             exec_time = end_time-start_time

#             wandb.log({"tr_epoch_exec_t" : exec_time})
            
#             model_dir = dict_files["save_model_dir"]
#             torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

#             print(f"############### Test epoch {i} ###############")
#             # Set the model to evaluation mode, disabling dropout and using population
#             # statistics for batch normalization.
#             model.eval()
#             start_time = time.time()
#             # Disable gradient computation and reduce memory consumption.
#             with torch.no_grad():
#                 with tqdm(test_loader, unit="batch") as tepoch:
#                             for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
#                                 tepoch.set_description(f"Epoch {i}")

#                                 x = x.to(device)
#                                 x_mask = x_mask.to(device)
#                                 z = z.to(device)
#                                 weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
#                                 w = [w_values.to(device), weather_coords_batch.to(device)]
#                                 y = y.to(device)
#                                 y_mask = y_mask.to(device)
#                                 #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 y_hat = model(x, z, w, x_mask)
                                
#                                 #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 loss = masked_mse(y_hat,
#                                               y,
#                                               y_mask)
                                
#                                 print("Test_loss: ", loss.item())
#                                 wandb.log({"Test_loss":loss.item()}) 
                    
#             end_time = time.time()
#             exec_time = end_time-start_time
#             wandb.log({"test_epoch_exec_t" : exec_time})

#             # Plots
#             train_test_plots(ds, model, device, 
#                              dates_list = dict_files["plot_dates"],
#                              tsteps_list= dict_files["plot_tstep_map"])

#         model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
#         model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
#         model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
#         wandb.log({"model_arch": model_arch})
                        
#         wandb.finish()

#         print(f"Execution time: {end_time-start_time}s")