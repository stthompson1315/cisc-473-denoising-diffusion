# cisc-473-denoising-diffusion

All model training and image generation is recommended to be run on Google Colab: upload a given notebook and press "run all" to train and generate images.

Due to the stochastic nature of DDPM, the results will vary from run to run if not using a seed. Original results did not record seeds, but should reproduce approximately.

Run all_results.ipynb to get every FID and IS value all at once, but be warned: can take multiple hours to fully run

Reproducability is also a potential issue with weights and training, so recommended reproduction strategy is running each notebook in "runs for results" individually

Starting with DDPM linear, then DDPM cosine, DDIM, CFG and finally Consistency 





