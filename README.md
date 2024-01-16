# Introduction
This repository contains code for stock technical and fundamental analysis. Additionally, it has notebooks on **fourier transforms**, **technical indicators**, **fundamental indicators** and the **Kalman Filter**. This also has sample code for downloading market data (from stock price data to money market interest rate data, treasuries curve data, etc.).

In the notebooks you see how to build different financial features using technical indicators, fundamental indicators, money market data, fourier transforms and Kalman Filter (for the alpha $\alpha$ and beta $\beta$ dynamic computations) with their corresponding documentations. Moreover, there is code to generate CNN + RNN time series models, using `PyTorch` and optimization of the hyper-parameters with `Optuna`.

# Documentation
The only language used in this repository is Python. The main modeling framework is PyTorch. 

+ `data_processing.py`: Python module containing objects to pre-process data for modeling.
+ `model_utils.py`: Python module containing objects for time-series modeling.
+ `fourier_transforms.ipynb`: Jupyter notebook with theory and sample code for understanding the intuition and mechanics behind a Fourier Transform.
+ `fundamental_data.ipynb`: Jupyter notebook with theory and sample code for downloading Microsoft's quarterly fundamental data from their webpage.
+ `market_data.ipynb`: Jupyter notebook with graphs and sample code for downloading market data. This includes the SPY ETF, FED interest rate data, Treasury bond curve data, Vix (volatility), etc.
+ `technical_indicators.ipynb`: Jupyter notebook containing theory and sample code for various technical indicators (*e.g.* MACD, momentum, RSI). It also contains a deep dive into the Kalman Filter and its applications in Finance (and sources).
+ `modeling.ipynb`: Jupyter notebook that puts all together to build a time-series deep learning model, using Optuna as the hyper-parameter optimization framework, with result analysis and optimization outputs.

