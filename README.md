# Installing Miniconda and Creating Environments

Miniconda is a popular package and environment management tool used for data science and scientific computing. It allows users to easily install, manage, and work with different environments containing various Python packages.

**If you want to use jupyter notebooks locally follow the instructions below. Otherwise jump to the final part of this document.**

## Downloading & Installing Miniconda
   - Visit Miniconda installation [site](https://docs.anaconda.com/free/miniconda/miniconda-install/)
   - Choose the appropriate installer for your operating system
   - Follow the instructions for installing it on the website

## Create Environments:
**1. Creating a New Environment:**
- Open an Anaconda (miniconda) terminal (search in the task bar for "miniconda")
- Create a new environment with a python version 3.9:
    ```
    conda create --name myenv python=3.9
    ```
    Replace myenv with the desired name for your environment. **Make sure you use Python version 3.9**

**2. Activate the Environment:**

```
conda activate myenv
```

**3. Install Packages:**

Either install each package manually from the requirement.txt file

```
pip install package_name
```

Or use the following command

```
pip install -r requirement.txt
```

**4. Deactivate Environment**
When you're done working in the environment, you can deactivate it using:
```
conda deactivate
```

## Install Tensorflow

(We will not use this in the first week)

We will be using Tensorflow version 2.10, you can use the following [link](https://www.tensorflow.org/install/pip) for step-by-step installation process (scroll down to **Step-by-step instructions** section ).

(TF Version 2.10 will not work unless you have Python version 3.9)

If you do not have a GPU and still want to use the notebooks locally, make sure that you don't install the GPU version.

## Jupyter Notebooks
**1. Notebooks through Jupyter server**

- Open a new miniconda terminal and activate the environment that you created
- Start a jupyter server using the following command:
  
```
jupyter notebook
```

- you can access the server using the following URL"
  
```
localhost:8888
```

**2. Jupyter notebooks using VScode**
- Open Extensions tab in vscode
- Install "Jupyter" extension
- Reload vscode
- Create a file with the extension "ipynb", when opened you will be able to see the cells and other jupyter functionalities

**3. Using Online Notebooks**
- Kaggle notebooks: [https://www.kaggle.com/code](https://www.kaggle.com/code)
- Google Colab: [https://colab.google/](https://colab.google/)