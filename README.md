# Predict Customer Churn

- Project **Predict Customer Churn** for ML DevOps Engineer Nanodegree Udacity

## Project Description

This project identifies credit card customers that are most likely to churn.

## Files and data description

Overview of the files and data present in the root directory.
.
├── Notebooks
| ├── Guide.ipynb
│ ├── churn_notebook.ipynb
├── data
│ └── bank_data.csv
├── images
│ ├── eda # Store EDA results
│ ├── results # feature importance, ROC curves
| └── reports # Metrics Reports
├── logs # Store logs
├── models # Store models
├── churn_library.py # Core Library
├── churn_script_logging_and_tests.py # Tests and logs
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md

## Running Files

To run the files you can use depending on your OS system. Running these
commands will take some time.

```cmd
./run.sh windows
```

or

```cmd
./run.sh linux
```

### churn_library.py flow

![My Remote Image](https://video.udacity-data.com/topher/2022/March/6240a53a_sequencediagram/sequencediagram.jpeg)

### TODO later

- Move constants to their own constants.py file
- Re-organize each script to work as a class.
- Add dockerfile
