# AlgRobSSAT
Make SSAT more scalable.


## Overview
Idea: During sample generation, you steer the unconditional diffusion model with the non-robust classifier’s uncertainty gradient so that the samples land near the decision boundary. Then you pseudo-label those boundary samples and use them (plus the real data) to train the robust classifier.

Improve the denoising process such that adversarial examples are near decision boundaries, i.e. do not filter after "blind" generation of synthetic data, instead on the fly generate already adversarial examples that are near the decision boundary (condition the denoising process on the decision boundary of the non-robust classifier (pseudo labeler)).

We can do this cleanly by treating the classifier as an energy (or log-density) term that rewards being near the boundary and use it as guidance during the reverse diffusion. Pick a scalar potential ϕ(x) that is large near the boundary and small away from it. Then we can use the classifier to steer the diffusion process by adding a term to the loss that encourages samples to be near the boundary.

## Learning Objectives
- Understand the SSAT algorithm and its limitations.
- Include the motivation:
  -  such that the curse of dimensionality is addressed (i.e. the number of samples needed to cover the input space grows exponentially with the number of dimensions & the distance between samples becomes the same).
  -  Not all samples are equally informative, i.e. some samples are more informative than others (use active learning).
  -  Use the labeler as a conditioning mechanism.
- [x] Understand the DDPM model and how it can be used to generate conditioned synthetic data.