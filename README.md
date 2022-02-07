# Masked face people detection
This repository consists of the implementation of a VGG16 convolutional model for the detection of people with masks

For use the trained model, you must download the model in the following link to google drive [here](https://drive.google.com/file/d/185ye-JuwDpGiUXkuD-6ZrPCWu_7O3ASm/view?usp=sharing). Once downloaded, you must unzip the file and put it in the root directory of the project (Masked Face Detection folder).

### Deploying the app in streamlit (Linux)

You must activate the virtual environment of python, to later be able to deploy the application. To start the environment, copy and paste the following code into your terminal at the root of the project.



```bash
root_path=$(pwd)

. venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$root_path
```

Once activated, you must go to **app directory** and run **deploy.sh**. Below I show how the project file hierarchy should be.

```bash
├── README.md
├── app
│   ├── __init__.py
│   ├── deploy_app.sh # ❗Execute this script.
│   └── main.py
├── base
│   ├── __init__.py
│   ├── __pycache__
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
├── code_tests
│   ├── load_callbacks.py
│   ├── model_creation.py
│   └── training_test.py
├── configs
│   └── maskedfacepeople_vgg16_exp_004.json
├── data_loader
│   ├── __init__.py
│   ├── __pycache__
│   └── data_loader_01.py
├── execute_train.sh
├── experiments # ❗Experiments directory must be unzipped.
│   ├── 2022-01-27
├── models
│   ├── __init__.py
│   ├── __pycache__
│   └── model_01.py
├── optimizers
│   ├── __init__.py
│   ├── __pycache__
│   └── learning_rate_schedules.py
├── real_time_detection
│   ├── haarcascade_frontalface_default.xml
│   ├── image_detection.py
│   ├── real_time_detection.py
├── requirements.txt
├── start_venv.sh
├── train.py
├── trainers
│   ├── __init__.py
│   ├── __pycache__
│   └── vgg_trainer.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   ├── config.py
│   ├── factory.py
│   ├── get_learning_rate.py
│   ├── get_model_size.py
│   └── process_args.py
└── venv
```