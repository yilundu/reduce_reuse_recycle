# Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC
## [<a href="https://energy-based-model.github.io/reduce-reuse-recycle/" target="_blank">Project Page</a>][<a href="https://colab.research.google.com/drive/1jvlzWMc6oo-TH1fYMl6hsOYfrcQj2rEs?usp=sharing" target="_blank">Colab</a>]

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/5572232/220243743-7826c5b0-a8f0-452b-b62e-5b4f6aed4a26.gif)

We provide a framework for probabilistically composing and repurposing diffusion models across ifferent domains as described <a href="https://energy-based-model.github.io/reduce-reuse-recycle/" target="_blank">here</a>.

[//]: # (### Abstract)
> Since their introduction, diffusion models have quickly become the prevailing approach to generative modeling in many domains. They can be interpreted as learning the gradients of a time-varying sequence of log-probability density functions. This interpretation has motivated classifier-based and classifier-free guidance as methods for post-hoc control of diffusion models. In this work, we build upon these ideas using the score-based interpretation of diffusion models, and explore alternative ways to condition, modify, and reuse diffusion models for tasks involving compositional generation and guidance. In particular, we investigate why certain types of composition fail using current techniques and present a number of solutions. We conclude that the sampler (not the model) is responsible for this failure and propose new samplers, inspired by MCMC, which enable successful compositional generation. Further, we propose an energy-based parameterization of diffusion models which enables the use of new compositional operators and more sophisticated, Metropolis-corrected samplers. Intriguingly we find these samplers lead to notable improvements in compositional generation across a wide variety of problems such as classifier-guided ImageNet modeling and compositional text-to-image generation.

For more info see the [project webpage](https://energy-based-model.github.io/reduce-reuse-recycle/).

## Notebooks

We provide two separate notebooks to aid in implementing the results illustrated in the paper. 

* **notebooks/simple_distributions.ipynb** This notebook contains code for reproducing 2D distribution results in the paper. The notebook contains a stand-alone diffusion trainer for a EBM-parameterized model as well as code for different MCMC samplers (HMC, ULA, MALA, UHA) across different distribution combinations
* **notebooks/image_tapestry.ipynb** This notebook contains code illustrating how we may use MCMC sampling on existing text-to-image to construct image tapestries. Our image tapestry results are done using the Imagen 64x64 model where we can define diffusion models across different image sizes in the pixel space. Here, tapestires are defined in latent space (where linear interpolations are not well defined) and thus results are substantially poorer (but illustrate how MCMC sampling may be implemented in existing text-to-image models)

## Training Code

Most of the larger-scale experiments done in the paper were done using the computational infrastructure at DeepMind and cannot be released. If there is sufficient interest (feel free to start a github issue), I'll add a external PyTorch reimplementation of the experiments in the paper. Changing a diffusion model to a energy-based parameterization should only involve 3 lines of code -- just replace output prediction with `torch.autograd.grad([energy], [input])[0]` (see the 2D colab for an example).
