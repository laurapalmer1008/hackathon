# more.tech-4.0

hackathon, data

## Launch instructions

1. install required libraries
2. Set up following properties:
- ```FLASK_APP = backend.py```
- ```FLASK_ENV = development```
- ```FLASK_DEBUG = 0```
3. launch application with ```python -m flask run```

## Use application
By default the app will be using address ```127.0.0.1:5000```
The following calls can be made:
- ```/director_news``` parses fresh news and returns most relevant for director news from the data set
- ```/accountant_news``` parses fresh news and returns most relevant for accountant news from the data set
