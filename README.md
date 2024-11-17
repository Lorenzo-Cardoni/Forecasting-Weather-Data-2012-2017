# Forecasting Weather Data Los Angeles 2012-2017

![Python](https://img.shields.io/badge/Python-3.x-blue) ![pmdarima](https://img.shields.io/badge/pmdarima-latest-orange) ![Pandas](https://img.shields.io/badge/Pandas-1.x-yellowgreen) ![Forecasting](https://img.shields.io/badge/Forecasting-ARIMA/SARIMA/SARIMAX-brightgreen)

## Overview

This project is focused on predicting temperature trends in various US cities by applying time series analysis and forecasting techniques. The model primarily leverages `SARIMAX` (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) to account for seasonality and patterns in temperature data across different time intervals. The predictions are used to analyze future temperature trends.

## Features

- **Time Series Forecasting**: Uses SARIMAX for reliable seasonal forecasting.
- **Customizable Forecasts**: Allows users to adjust the forecast interval (e.g., weekly, bi-weekly, monthly).
- **Visualization**: Plots historical data, forecasts, and confidence intervals for easy interpretation.

## Dataset

The dataset consists of hourly temperature readings from several US cities over multiple years (2012 - 2017). The data includes:
- **Datetime**: Timestamp for each observation.
- **City Temperatures**: Temperatures in cities including Los Angeles, Vancouver, Portland, San Francisco, and others.

## PDF
In the second part of `Progetto_Classification_Clustering_Forecasting.pdf` file is reported the result of this work.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `pmdarima`
  - `sklearn`
  - `statsmodels`

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib pmdarima scikit-learn statsmodels
