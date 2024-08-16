# Attacks-and-Security-of-Federated-Learning

> **Abstract:**
> Efficient prediction of short-term energy consumption in homes is essential for the development of smart grids and requires the exploration of advanced techniques such as deep neural networks. However, the traditional approach of centralising data for model training raises privacy concerns. Decentralised solutions, in particular federated learning (FL), have emerged to address this, but they are not immune to security threats. This research addresses the vulnerabilities and security risks associated with FL in short-term load forecasting, exploring existing security measures and proposing a focus on GAN-based attacks. The study reviews the relevant literature, conducts an in-depth analysis of GAN attacks, evaluates security measures such as noise and differential privacy, and implements and evaluates GAN attacks on real datasets to improve the robustness of FL models. The contribution lies in identifying defensive measures to strengthen the security and privacy of FL models, with a particular focus on mitigating GAN attacks in the context of short-term load forecasting.

### Project structure
This repsoitory consists of the following parts: 
- **data** folder: Here are all datasets and scripts to collect the datasets, preprocess them, performe feature engineering and create the final dataset used for the forecasting task.
- **evaluations** folder: Here are all the evaluation results stored
- **images** folder: Here are all figures and plots stored and the script to create them
- **models** folder: Here the model weights are stored
- **src** folder: Here the main scripts are stored for the forecasting baseline, local learning, federated learning and evaluation
  - **utils** folder: Here helper classes for data handling, model generation, and model handling are stored

### Install and Run the project 
To run the project you can fork this repository and following the instructions: 
First create your own virtual environment: 
```
python -m venv .venv
```
Activate your environment with
```
 .\.venv\Scripts\activate.ps1
```
Install all dependencies with
```
pip install -r requirements.txt
```
