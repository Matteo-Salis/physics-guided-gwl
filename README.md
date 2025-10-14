We developed models capable of learning from spatially sparse data and predicting at an arbitrary and variable number of locations, leveraging available spatially sparse data measured from in situ sensors (piezometers) and the spatio-temporal meteorological information structured as a video.
Furthermore, explored physics-guided deep learning approaches. Specifically, we tested two strategies, inductive and learning bias, respectively, to embed prior knowledge derived from the groundwater flow equation into the models, focusing on Piedmont.
The proposed models: STNet (pure deep learning), STDisNet (inductive bias), STDisNetPI (inductive+learning bias), STDisNetPI-RCH (inductive+learning bias+recharge zone constraint).
Refer to the manuscritp for additional details.

Architecture of STDisNet, STDisNetPI, and STDisNetPI-RCH:

![MultiPoint_Net-STDisNet](https://github.com/user-attachments/assets/8c545100-cd38-4711-bb82-2def27696cd6)

Here is the predicted evolution of groundwater level for the best-performing model(STDisNetPI):

![gif_from_2021_06_20_iter_pred_GWL_STDisNetPI](https://github.com/user-attachments/assets/ea1ed8c2-9845-4fee-b389-8f8796206843)

Here is the predicted evolution of the diffusion component $\hat{\Delta_{GW}}}$ and the source/sink term $\mathcal{\hat{R}}$:

