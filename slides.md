# A Survey on Uncertainty Quantification Methods for Deep Learning

Chrysoula, July 2025

---


# Introduction
<div style="max-width: 1000px; margin: 0 auto; text-align: left;">

Uncertainty quantification (UQ) aims to estimate the confidence of DNN predictions in addition to
prediction accuracy. DNNs can make unexpected, incorrect and overconfident predictions, particularly in complex real-world scenarios.
UQ assigns uncertainty scores to DNN predictions, addressing data and model uncertainty.
This survey looks at different ways to measure uncertainty in Deep Neural Networks (DNNs). 
 
There are two main types of uncertainty:
- Data Uncertainty (Aleatoric): Comes from inherent noise or ambiguous labels.
- Model Uncertainty (Epistemic): Comes from the lack of evidence or knowledge during model training
or inference, e.g., limited training samples.

</div>


---

# Taxonomy of UQ methods for DNNS  
<div style="max-width: 1000px; margin: 0 auto; text-align: left;">

**Model uncertainty:**

</div>


- Bayesian Neural Network (BNN): Learn the posterior distribution of model parameters to reflect parameter uncertainty.
- Monte Carlo (MC) dropout: Uncertainty estimation can be obtained by computing the variance of multiple forward passes, each using a different pattern of deactivated neurons via dropout. 
- Ensemble models: Combine multiple neural networks to form an output distribution, quantifying model uncertainty through the variability of the distribution. 
- Sample distribution-related methods: These include two cases. The first one includes test samples that follow a different distribution than training samples (OOD scenario), while in the second a test sample is far from other training samples or is surrounded by sparse training samples. Existing approaches for the second case include Gaussian process hybrid neural networks and distance-aware neural networks. 

---

# Taxonomy of UQ methods for DNNS  

<div style="max-width: 1000px; margin: 0 auto; text-align: left;">

**Data uncertainty:**

</div>

- Deep discriminative models: Outputs a predictive distribution directly using a neural network, modeled as a parametric or non-parametric model.
- Deep generative models learn the complex, high-dimensional data distribution, employing variational autoencoders (VAEs), generative adversarial networks (GANs) and diffusion models.


<div style="max-width: 1000px; margin: 0 auto; text-align: left;">

**Combination of model and data uncertainty:**

</div>


- Combine BNN model or Ensemble model with prediction distribution.
- Combine ensemble model with prediction interval.
- Evidential Deep Learning.
- Conformal Prediction.

---

# Uncertainty Estimation in Various Machine Learning Problems
- Out-of-distribution detection: A DNN model should be able to recognize out-of-distribution (OOD) samples that differ from the training data distribution.
- Active learning: Active learning aims to solve the data labeling issue by prioritizing instances where predictions are most uncertain.
- Deep Reinforcement Learning: It aims to train an agent with the environment to maximize its total rewards.

---
# Future Direction
- UQ for Large Language Models: LLMs sometimes generate over-confident outputs that are incorrect. Designing UQ methods for LLMs is essential for improving trustworthiness.
- UQ for Deep Learning in Scientific Simulations: UQ for deep learning in scientific simulation is crucial in high-stake decision-making applications (e.g., disaster response).
- Combine UQ with DNN Explainability: Combining uncertainty quantification and explanation are important for a robust, trustworthy AI model.

---
# Conclusion
- This paper categorizes Uncertainty Quantification (UQ) methods for deep neural networks (DNNs) into three groups: model uncertainty, data uncertainty and their combination.
- It analyzes and evaluates strengths and weaknesses of each approach based on the type of uncertainty addressed. (This part is not mentioned in the presentation!)
- It summarizes the sources of uncertainty across various machine learning problems and proposes future research directions.

---
# References 
- [1] W. He, Z. Jiang, T. Xiao, Z. Xu, and Y. Li, “A survey on uncertainty quantification methods for deep learning,” *arXiv preprint arXiv:2302.13425*, 2023. DOI: https://doi.org/10.48550/arXiv.2302.13425


---
# Thank you!



