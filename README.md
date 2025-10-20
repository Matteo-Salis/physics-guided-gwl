We developed models capable of learning from spatially sparse data and predicting at an arbitrary and variable number of locations, leveraging available spatially sparse in situ measurements (from piezometers) and spatio-temporal meteorological information structured as a video.
Furthermore, we explored physics-guided deep learning approaches. Specifically, we tested two strategies, inductive and learning bias, respectively, to embed prior knowledge derived from the groundwater flow equation into the models, focusing on Piedmont. <br />
The proposed models: STAINet (pure deep learning), STAIDiNet (inductive bias), STAIDiNet-PL (inductive+learning bias), and STAIDiNet-PRL (inductive+learning bias+recharge zones constraint). 
Refer to the manuscript for additional details.

The groundwater flow equation:

$$
\frac{\partial h}{\partial t} = \frac{K}{S_s} (\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}) + \mathcal{R}
$$

We defined $\Delta_{GW} = \mathcal{D}(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2})$, and $\mathcal{D} = \frac{K}{S_s}$.

Architecture of STAIDiNet, STAIDiNet-PL, and STAIDiNet-PRL:

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1D92U0-lUl_ESRayI73k0by_NY-seBWz3=s700" >
</p>

Here is the predicted evolution of groundwater level using the best-performing model (STAIDiNet-PL):

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1f1dNbodNo2VZqZmi2sEtKQbj4LpNXKsS=s500" >
</p>

Here is the predicted evolution of the diffusion component $\hat{\Delta}_{GW}$ and the source/sink term $\mathcal{\hat{R}}$:

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1auS_ij_uGEyeiBbKKOPvpcSPLGwsTu1y=s400" >
  <img src="https://lh3.googleusercontent.com/d/18b30t3wg9dJzH1pGVBLR0Ek_-W8xgZax=s400" >
</p>
