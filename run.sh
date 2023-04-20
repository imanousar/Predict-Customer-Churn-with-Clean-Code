#!/bin/bash

if [[ "$1" == "windows" ]]; then
    python -m venv venv
    source venv/Scripts/activate
elif [[ "$1" == "linux" ]]; then
    python -m venv venv
    source venv/bin/activate
else
    echo "Invalid argument. Usage: $0 [windows|linux]"
    exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt

echo 'running churn_library.py'
python churn_library.py

echo 'churn_script_logging_and_tests.py'
python churn_script_logging_and_tests.py

echo 'end of execution'

deactivate