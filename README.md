# Natural Language Processing in PyTorch (ft. PyTorch Lightning)

## Built with
<code>
<img height="50" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png">
<img height="50" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg">
<img height="50" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg">
<br>
<img height="50" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png">
<a href="https://pytorch.org/"><img height="50" src="https://miro.medium.com/max/691/0*xXUYOs5MWWenxoNz"></a>
<a href="https://github.com/PyTorchLightning/pytorch-lightning"><img height="50" src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png?raw=true"></a>
<a href="https://github.com/huggingface/transformers"><img height="50" src="https://image4.owler.com/logo/hugging-face_owler_20191218_073707_original.png"></a>
</code>

## Summary
- This repo was developed to show how to label text data and train deep learning models for sentiment analysis and question answering tasks.
- **DistilBert** transformer models were trained with the `HuggingFace` `transformers` library for the two tasks mentioned.

## Notebooks

**1. Labeling Text.ipynb**: [Open in GitHub](https://github.com/ansonnn07/Natural-Language-Processing-PyTorch/blob/master/1.%20Labeling%20Text.ipynb) <br>
Showing how to label text data using Label Studio, for text classification and question answering tasks.

**2. Training - Sentiment Analysis (Huggingface & PyTorch Lightning).ipynb**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ecwNoxd2jMAJyzrcMpvIjJkgOgPpONrr?usp=sharing) <br>
Showing how to use the Huggingface `transformers` library to easily obtain pretrained models, and use them for training, fine-tuning, and inference. Used [Huggingface Trainer API](https://huggingface.co/transformers/main_classes/trainer.html) to easily fine-tune models. Also used [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for more flexibility if needed. This notebook trains on the [Coronavirus Tweets Dataset](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification) obtained from Kaggle.

**3. Training - Question Answering.ipynb**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wKX1GUjEhjuc0V0UnnR7z1g5ePITD3lk?usp=sharing) <br>
This notebook fine-tuned another DistilBert model specifically for question answering task. The dataset used in the notebook is the The Stanford Question Answering Dataset 2.0 ([SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)) dataset.

**4. Training - Chatbot with Rick & Morty Dataset**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GTdl5mR3h1enh3pFMH2J7E4TJpEWGTU?usp=sharing)
This notebook fine-tuned the DialoGPT model on the [Rick & Morty dataset](https://www.kaggle.com/andradaolteanu/rickmorty-scripts) to create a Rick Chatbot. This notebook is simply adapted from the Huggingface example [script](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) and this [tutorial](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html).

NOTE: Please refer the training notebooks in Google Colab for the latest updates there.

## PyTorch Installation
To install PyTorch with GPU support, refer to the official page from PyTorch [here](https://pytorch.org/get-started/locally/). Or just run the code below in your virtual environment.

```
conda install pytorch torchvision torchaudio cudatoolkit=<YOUR_CUDA_VERSION> -c pytorch -c conda-forge
```

Keep in mind that you will need a compatible CUDA version (preferably v10.2) to work with PyTorch GPU. You can check whether you have installed CUDA or not by checking the existence of any folder inside `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"` (for Windows, I am not sure about Linux), you should see a folder named with CUDA version, e.g. `v10.2` for CUDA version 10.2. You may download and install CUDA from [here](https://developer.nvidia.com/cuda-downloads) if you haven't done so yet. 

NOTE: If you have used TensorFlow with GPU support before, then you have already installed CUDA and cuDNN, if not, you may refer to how to install them in this YouTube [video](https://youtu.be/hHWkvEcDBO0) (without installing the TensorFlow in the same environment of course, since we are using PyTorch here).

If you get an error as shown below:
```
ImportError: DLL load failed while importing win32api: The specified module could not be found.
```
Then you need to run the following commands below to uninstall and reinstall `pywin32`, the uninstall command might tell you that `pywin32` is not found but it's ok, just proceed to the next command.
```
pip uninstall pywin32
conda uninstall pywin32
conda install pywin32
```

## PyTorch Lightning Installation
```
pip install pytorch-lightning
```

## Package Installation
Just run the following command to install the rest of the dependencies:

```
pip install -r requirements.txt
```