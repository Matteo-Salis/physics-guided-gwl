We developed models capable of predicting Ground Water Level (GWL) at an arbitrary number of points inside the ROI, leveraging available spatially sparse data measured from in situ sensors (piezometers) and spatially distributed meteorological information (meteorological video). Furthermore, we explored physics-guided deep learning approaches. Specifically, we tested two strategies, inductive and learning bias, respectively, to embed prior knowledge derived from the groundwater flow equation into the models, focusing on Piedmont. <br />
The proposed models: STAINet (pure deep learning), PSTAINet-IL (inductive bias), PSTAINet-ILB (inductive+learning bias), and PSTAINet-ILRC (inductive+learning+recharge zones biases). 
Please refer to the manuscript for additional details.

The groundwater flow equation:

$$
\frac{\partial h}{\partial t} = \frac{K}{S_s} (\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2})
$$

We applied the Euler integration and defined:

- $\mathcal{D} = \frac{K}{S_s}$

- $\Delta_{GW_{t}} = \Delta t\left[\mathcal{D}\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right)\right]$

We further add a residual (sink/source) term $\mathcal{R}_{t}$ related to all exogenous factors occurring in the interval $\Delta t$, which can be either anthropogenic (e.g., water abstraction for irrigation) or natural (e.g., rainfall or snowmelt recharge).

We thus obtained:

$h_{t} = h_{t-1} + \Delta_{GW_{t}} + \mathcal{R}_{t}$

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
