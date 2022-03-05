# Flask api implement universial-sentence-encoder

-   This project has rank first in the FPT Code Arena 2022.

## How to use

### 0.0 dowload universal-sentence-encoder

https://tfhub.dev/google/universal-sentence-encoder/4
un-tar the file

### 0. install virtual env

pip install --user virtualenv

### 1. create virtual env

> Only run this command when first install project
> Next times, just run (2) and (4)

python -m venv ./myenv

### 2. activate virtual env

-   window ver

> myenv\Scripts\activate

### 3. Install libraries

pip install -r .\requirements.txt

### 4. run web app

-   window ver

> set FLASK_APP=app.py
> set FLASK_ENV=development
> flask run

### 5. deactivate virtual env

deactivate
