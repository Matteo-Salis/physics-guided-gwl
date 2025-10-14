<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

We developed models capable of learning from spatially sparse data and predicting at an arbitrary and variable number of locations, leveraging available spatially sparse in situ measurements (from piezometers) and spatio-temporal meteorological information structured as a video.
Furthermore, we explored physics-guided deep learning approaches. Specifically, we tested two strategies, inductive and learning bias, respectively, to embed prior knowledge derived from the groundwater flow equation into the models, focusing on Piedmont. <br />
The proposed models: STNet (pure deep learning), STDisNet (inductive bias), STDisNetPI (inductive+learning bias), STDisNetPI-RCH (inductive+learning bias+recharge zones constraint). 
Refer to the manuscript for additional details.

The groundwater flow equation:
$$
\frac{\partial h}{\partial t} = \frac{K}{S_s} (\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}) + \mathcal{R}
$$
We defined $\Delta_{GW} = \mathcal{D}(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2})$, and $\mathcal{D} = \frac{K}{S_s}$.

Architecture of STDisNet, STDisNetPI, and STDisNetPI-RCH:

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1D92U0-lUl_ESRayI73k0by_NY-seBWz3=s700?authuser=0" >
</p>

Here is the predicted evolution of groundwater level for the best-performing model(STDisNetPI):

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1ruQXoQAd8ZhjnFQAN5VTeoutB2mq4kmz=s500?authuser=0" >
</p>

Here is the predicted evolution of the diffusion component $\hat{\Delta_{GW}}$ and the source/sink term $\mathcal{\hat{R}}$:

<p align="center">
  <img src="https://lh3.googleusercontent.com/d/1b_dWmuk9Julp5_WU1119OqugvwLPUUiI=s400?authuser=0" >
  <img src="https://lh3.googleusercontent.com/d/1TzpAK-yKWTklDmDvHlEF3sOWIxiUq5TB=s400?authuser=0" >
</p>

