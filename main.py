from __future__ import annotations
from enum import EnumType

class Period(EnumType):
  END_PERIOD = "2015_12_31"
  MONTHLY = 30
  QUARTERLY = 90
  YEARLY = 365
  WEEKLY = 7
  SEASON_QUARTERLY = 4
  SEASON_WEEKLY = 7
  SEASON_MONTHLY = 12
  START_PERIOD = "2015_01_01"
  START_PERIOD_INT = "1/1/2015"
  END_PERIOD_INT = "12/31/2015"
  FIVE_YEARS = 1825
class ColumnData(EnumType):
  DATE = "date"
  SALES = "sales"
  ID_SALES = "id_sales"
  PLAN = "plan"
  ID_PLAN = "id_plan"
  PLAN_SALES = "sales / plan, %"
  GROWTH_RATE = "growth_rate"
  GROWTH_RATE_PCT = "growth_rate_pct"
  SALES_TREND = "sales_trend"
  FORECAST_TREND = "forecast_trend"
  FORECAST_TREND_PCT = "forecast_trend_pct"
  FORECAST = "forecast"
  HISTORICAL_DATA = "historical_data"
  FORECAST_COMPARISON = "Forecast Comparison Models"
  STATISTICS = "statistics"
  GOODS = "goods"
  REGION = "region"
  ID_REGION = "id_region"
  ID_GOODS = "id_goods"
  ID_REGION_GOODS = "id_region_goods"
  QUARTER: str = "quarter"
  YEAR: str = "year"
  MONTH: str = "month"
  ANALYSIS_RESULTS = "Analysis Results (mln)"
class QueryDB(EnumType):
  SELECT_DATE_SALES_PLAN = f"""select 
            {ColumnData.DATE},
            {ColumnData.ID_SALES},
            {ColumnData.ID_PLAN}
          from
            sales
          order by
            date desc"""

  CREATE_TABLE_SALES = f"""
    CREATE TABLE IF NOT EXISTS sales (
    id int not null auto_increment primary key,
    {ColumnData.DATE} date not null,
    {ColumnData.ID_SALES} int not null,
    {ColumnData.ID_PLAN} int not null,

    FOREIGN KEY({ColumnData.ID_SALES}) REFERENCES sales_data(id) 
    ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY({ColumnData.ID_PLAN}) REFERENCES plan_data(id) 
    ON UPDATE CASCADE ON DELETE CASCADE 

    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_PLAN_DATA = f"""
    CREATE TABLE IF NOT EXISTS plan_data (
    id int not null auto_increment primary key,
    {ColumnData.PLAN} int not null,
    {ColumnData.DATE} date not null
    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_SALES_DATA = f"""
    CREATE TABLE IF NOT EXISTS sales_data (
    id int not null auto_increment primary key,
    {ColumnData.SALES} int not null,
    {ColumnData.DATE} date not null
    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_GOODS = f"""
    CREATE TABLE IF NOT EXISTS goods (
    id int not null auto_increment primary key,
    {ColumnData.GOODS} varchar(100) not null
    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_REGION = f"""
    CREATE TABLE IF NOT EXISTS region (
    id int not null auto_increment primary key,
    {ColumnData.REGION} varchar(100) not null
    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_REGION_GOODS = f"""
    CREATE TABLE IF NOT EXISTS region_goods (
    id int not null auto_increment primary key,
    {ColumnData.ID_REGION} int not null,
    {ColumnData.ID_GOODS} int not null,

    FOREIGN KEY({ColumnData.ID_REGION}) REFERENCES region(id)
    ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY({ColumnData.ID_GOODS}) REFERENCES goods(id)
    ON UPDATE CASCADE ON DELETE CASCADE

    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_GOODS_SALES = f"""
    CREATE TABLE IF NOT EXISTS goods_sales (
    id int not null auto_increment primary key,
    {ColumnData.ID_GOODS} int not null,
    {ColumnData.ID_SALES} int not null,

    FOREIGN KEY({ColumnData.ID_GOODS}) REFERENCES goods(id)
    ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY({ColumnData.ID_SALES}) REFERENCES sales_data(id) 
    ON UPDATE CASCADE ON DELETE CASCADE

    ) ENGINE=InnoDB;
  """
  CREATE_TABLE_GOODS_PLAN = f"""
    CREATE TABLE IF NOT EXISTS goods_plan (
    id int not null auto_increment primary key,
    {ColumnData.ID_GOODS} int not null,
    {ColumnData.ID_PLAN} int not null,

    FOREIGN KEY({ColumnData.ID_GOODS}) REFERENCES goods(id)
    ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY({ColumnData.ID_PLAN}) REFERENCES plan_data(id) 
    ON UPDATE CASCADE ON DELETE CASCADE

    ) ENGINE=InnoDB;
  """

  INSERT_REGION = f"""
    INSERT INTO region (region)
    VALUES (:region)
  """

  CREATE_ALL_TABLES = f"""
    {CREATE_TABLE_GOODS} 
    {CREATE_TABLE_REGION} 
    {CREATE_TABLE_PLAN_DATA} 
    {CREATE_TABLE_SALES_DATA} 
    {CREATE_TABLE_GOODS_PLAN} 
    {CREATE_TABLE_GOODS_SALES} 
    {CREATE_TABLE_REGION_GOODS}  
    {CREATE_TABLE_SALES}  
  """
  INSERT_PlAN_DATE = f"""
    INSERT INTO plan_data (plan, date)
    VALUES (:plan, :date) 
    ON DUPLICATE KEY UPDATE 
    date = :date
  """

  UPDATE_DATE = f"""
    UPDATE plan_data 
    SET date (:date) 
    WHERE id (:id)
  """
class SheetName(EnumType):
  ZERO = 0
  FIRST = 1
  SECOND = 2
class SourceType(EnumType):
  CSV = "csv"
  EXCEL = "excel"
  SAMPLE = "sample"
  MYSQL = "mysql"
  ABSOLUTE_PATH = "/mdl_warehouse/avocado_plan.csv"
  DATA_FOLDER = "mdl_warehouse/avocado_plan.csv"
class FiguresEnum(EnumType):
  TREND = "trend"
  FORECAST = "forecast"
  ANALYSIS = "analysis"
  STATISTICS = "statistics"
  EVALUATION = "evaluation"
class Metrics(EnumType):

  MAPE = "MAPE"
  MEAN = "mean"
  MEDIAN = "median"
  MODE = "mode"
  RMSE = "RMSE"
  R2 = "R2"
  MAE = "MAE"
  MIN = "min"
  MAX = "max"
  TOTAL = "total"
  STD = "std"
  SUM = "sum"
  COUNT = "count"
  PLAN_RATE = 1.1
  ABS = "abs"
  METRICS = "Metrics"
  DESCRIPTIVE_STATISTICS = "Descriptive Statistics"
  MAPE_NORM = "Normalised Mean"
  MAE_NORM = "Normalised Mean"
  RMSE_NORM = "Normalised Root Mean Squared Error"
  R2_NORM = "Normalised Mean Squared Error"
  SCORE = "score"

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from model.model_test import ModelExample, show_figure_blocking

import logging

import time

import MySQLdb
import mysql.connector
import sqlalchemy
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
from pandas import DataFrame
from sqlalchemy import Connection, text
from sqlalchemy.dialects.mysql import insert

from typing import Any

from pandas.core.arrays import ExtensionArray
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents, UnobservedComponentsResults
from sklearn.metrics import (
  mean_absolute_error,
  r2_score, root_mean_squared_error, mean_absolute_percentage_error
)
from numpy import dtype, signedinteger
import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt

class Formulas:
  def __init__(self, data: pd.DataFrame | None = None, period: int = Period.YEARLY) -> None:
    self.data = data
    self.period = period

  def generate_data(self,
                    periods: int = Period.YEARLY,
                    plan_rate: float = Metrics.PLAN_RATE,
                    loc: float = 0.0,
                    scale: float = 1.0,
                    seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
      start='2015-01-01', end='2017-12-30',
      freq="D")
    trend = self.trend_calculate
    season = self.season_calculate
    base = 100 + trend + season
    noise = rng.normal(
      loc=loc, scale=scale, size=periods * 3)
    sales = base + noise
    plan = sales * plan_rate
    try:
      df = pd.DataFrame(
        {ColumnData.DATE: dates,
         ColumnData.SALES: sales,
         ColumnData.PLAN: plan})
    except:
      raise Exception("No data: %s, %s, %s" % (len(dates), len(sales), len(plan)))
    self.data = df
    logging.info(f"Generated sample data with %d periods", periods)
    return df

  @property
  def trend_calculate(self) -> np.ndarray[tuple[Any, ...], dtype[signedinteger[Any]]]:
    trend = 0.05 * np.arange(Period.YEARLY * 3)
    return trend

  @property
  def season_calculate(self) -> np.ndarray[tuple[Any, ...], dtype[signedinteger[Any]]]:
    season = 10 * np.sin(2 * np.pi * np.arange(Period.YEARLY * 3) / 90)
    return season

class TestData:
  def __init__(self, data: pd.DataFrame | None | pd.Series[Any] = None, period: int | None = None) -> None:
    self.data = data
    self.period = period

  @property
  def test_data(self) -> pd.DataFrame | pd.Series[Any] | None :

    if self.data is None:
      raise TypeError("Test data not provided")

    if len(self.data) == 0:
      raise Exception("No test data")

    if ColumnData.SALES not in self.data.columns:
      raise Exception(f"Column {ColumnData.SALES} not in test data")
    if ColumnData.DATE not in self.data.columns:
      raise Exception(f"Column {ColumnData.DATE} not in test data")
    if ColumnData.PLAN not in self.data.columns:
      raise Exception(f"Column {ColumnData.PLAN} not in test data")

    return self.data

class DatabaseConnection:
  def __init__(
    self,
    host: str = "localhost",
    user: str = "root",
    password: str = "",
    database: str = "sales") -> None:

    self.config = {
      "host": host,
      "user": user,
      "password": password,
      "database": database,
      "raise_on_warnings": True}
    self.connection = None
    self.logger = logging.getLogger(f"DatabaseConnection.{__name__}")
    self.logger.setLevel(logging.INFO)
    self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    self.handler = logging.StreamHandler()
    self.handler.setFormatter(self.formatter)
    self.logger.addHandler(self.handler)
    self.file_handler = logging.FileHandler("DatabaseConnection.log")
    self.file_handler.setFormatter(self.formatter)
    self.logger.addHandler(self.file_handler)

  def load_sales(
    self,
    query: str = QueryDB.SELECT_DATE_SALES_PLAN
  ) -> DataFrame | None:
    connect = self.connect_by_sqlalchemy()
    # connect = self.connect_by_mysql()
    try:
      df_temp = pd.read_sql(query, connect)
      if ColumnData.DATE in df_temp.columns:
        df_temp[ColumnData.DATE] = (
          pd.to_datetime(df_temp[ColumnData.DATE]))
      df = df_temp
      connect.close()
      self.close()
      return df
    except MySQLdb.ProgrammingError as error:
      self.logger.info(f"Table '{ColumnData.SALES}.{ColumnData.SALES}' does not exist: {error}")
    return None

  def close(self) -> None:
    if self.connection is not None:
      self.connection.close()

  def connect_by_mysql(
    self,
    attempts: int = 3,
    delay: int = 2) -> None | PooledMySQLConnection | MySQLConnectionAbstract:
    attempt = 1
    while attempt < attempts + 1:
      try:
        self.connection = mysql.connector.connect(**self.config)
        return self.connection
      except (mysql.connector.Error, IOError) as error:
        if attempt == attempts:
          self.logger.info(f"Failed to connect, exiting without a connection: {error}")
          return None
        self.logger.info(
          "Connection failed: %s. Retrying in (%d/%d) seconds.",
          error,
          attempt,
          attempts - 1)
        time.sleep(delay ** attempt)
        attempt += 1
    return None

  def create_table(self) -> None:
    # connect = self.connect_by_mysql()
    # query = QueryDB.CREATE_ALL_TABLES
    # connect.execute(query)
    # self.logger.info("Tables created")
    # self.connection.close()
    connect = self.connect_by_sqlalchemy()
    query = QueryDB.CREATE_ALL_TABLES
    with connect as cursor:
      cursor.execute(text(query))
      connect.commit()
    connect.close()
    self.close()

  def insert_data(self, df: pd.DataFrame) -> None:
    # connect = self.connect_by_mysql()
    # cursor = connect.cursor()
    # query = QueryDB.INSERT_REGION
    # insert_data = [{}]
    # cursor.executemany(query, region)
    # connect.commit()
    # cursor.close()

    connect = self.connect_by_sqlalchemy()
    query = QueryDB.INSERT_PlAN_DATE
    # df_clean = df.drop(columns=[ColumnData.SALES, ColumnData.PLAN])
    # df_drop_plan = df_drop_sales.drop(columns=[ColumnData.PLAN])
    df_clean = df.drop(columns=[ColumnData.SALES])
    df_array = df_clean.to_sql(
      "plan_data",
      connect,
      if_exists="append",
      method=self.insert_on_conflict_update,
      index=False)
    #
    # with connect as cursor:
    #   cursor.execute(text(query), df_array)
    #   connect.commit()

    self.logger.info("All data inserted")
    connect.close()
    if self.connection is not None:
      self.connection.close()

  def connect_by_sqlalchemy(self) -> Connection:
    user = self.config.get("user")
    password = self.config.get("password")
    host = self.config.get("host")
    database = self.config.get("database")
    # client_mysql_official = "mysql-connector-python"
    client_mysql_alternative = "mysqldb"
    dialect = f"mysql+{client_mysql_alternative}"
    db = sqlalchemy.create_engine(
      f"{dialect}://{user}:{password}@{host}/{database}")
    connect = db.connect()
    return connect

  def insert_on_conflict_update(self, table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (insert(table.table).values(data))
    stmt = stmt.on_duplicate_key_update(
      date=stmt.inserted.date)
    result = conn.execute(stmt)
    return result.rowcount

  def update_data_duplicate_key(self, df: pd.DataFrame) -> None:
    connect = self.connect_by_sqlalchemy()
    for d in df[ColumnData.DATE]:
      connect.execute(
        text("""
          INSERT INTO plan_data (date) 
          VALUES (:date) 
          ON DUPLICATE KEY UPDATE date=:date
        """
        ),
        {"date": d}
      )
    connect.commit()
    connect.close()

class DataPreprocessor:

  @staticmethod
  def prepare_time_series(df: pd.DataFrame,
                          date_col: str = 'date',
                          value_col: str = 'sales',
                          plan_col: str = 'plan',
                          freq: str = 'D') -> pd.DataFrame:
    """
    Полная подготовка временного ряда
    """
    # 1. Копируем и сортируем
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 2. Агрегация по дате (если есть дубликаты)
    # Проверяем, есть ли план в данных
    cols_to_aggregate = {value_col: 'sum'}
    if plan_col in df.columns:
      cols_to_aggregate[plan_col] = 'sum'

    # Добавляем остальные колонки
    other_cols = {col: 'first' for col in df.columns
                  if col not in [date_col, value_col, plan_col]}

    df = df.groupby(date_col, as_index=False).agg({
      **cols_to_aggregate,
      **other_cols
    })

    # 3. Создаем полный временной ряд
    full_range = pd.date_range(
      start=df[date_col].min(),
      end=df[date_col].max(),
      freq=freq
    )

    full_df = pd.DataFrame({date_col: full_range})
    full_df = pd.merge(full_df, df, on=date_col, how='left')

    # 4. Заполняем пропуски комбинированным методом
    # Сначала forward fill для небольших пропусков
    full_df[value_col] = full_df[value_col].ffill(limit=7)
    full_df[plan_col] = full_df[plan_col].ffill(limit=7)

    # Затем backward fill
    full_df[value_col] = full_df[value_col].bfill(limit=7)
    full_df[plan_col] = full_df[plan_col].bfill(limit=7)


    # Оставшиеся пропуски - линейная интерполяция
    full_df[value_col] = full_df[value_col].interpolate(
      method='linear',
      limit_direction='both'
    )
    full_df[plan_col] = full_df[plan_col].interpolate(
      method='linear',
      limit_direction='both'
    )

    # 5. Добавляем временные признаки
    full_df['day_of_week'] = full_df[date_col].dt.dayofweek
    full_df['month'] = full_df[date_col].dt.month
    full_df['quarter'] = full_df[date_col].dt.quarter
    full_df['year'] = full_df[date_col].dt.year
    full_df['day_of_year'] = full_df[date_col].dt.dayofyear
    full_df['is_weekend'] = full_df['day_of_week'].isin([5, 6]).astype(int)

    return full_df

  @staticmethod
  def detect_anomalies(df: pd.DataFrame, value_col: str = 'sales', plan_col: str = "plan") -> pd.DataFrame:
    """
    Обнаружение и обработка аномалий
    """
    df = df.copy()

    cols_to_check = [value_col]
    if plan_col in df.columns:
      cols_to_check.append(plan_col)

    for col in cols_to_check:
      if col not in df.columns:
        continue

      # Простой метод: удаляем выбросы за пределами 3 стандартных отклонений
      mean = df[col].mean()
      std = df[col].std()

      lower_bound = mean - 3 * std
      upper_bound = mean + 3 * std

      # Заменяем выбросы медианой
      median_val = df[col].median()
      df[col] = np.where(
        (df[col] < lower_bound) | (df[col] > upper_bound),
        median_val,
        df[col]
      )

    return df

class DataLoader:
  def __init__(
    self,
    generate: Formulas = None,
    data_source: str = None,
    db: DatabaseConnection = None,
    sheet_name: int | str = 0,
    periods: int = Period.YEARLY,
    start: str = Period.START_PERIOD,
    end: str = Period.END_PERIOD,
    plan_rate: float = Metrics.PLAN_RATE,
    seed: int | None = None,
    data: pd.DataFrame | None = None,
  ) -> None:
    self.generate = generate
    self.data_source = data_source
    self.db = db
    self.sheet_name = sheet_name
    self.periods = periods
    self.start = start
    self.end = end
    self.plan_rate = plan_rate
    self.seed = seed
    self.data = data
    self.prepare = DataPreprocessor

  @property
  def load_csv(self) -> pd.DataFrame:
    logging.info(f"Loading {self.data_source}")
    df = pd.read_csv(self.data_source)
    # df = TestData(data=df).test_data
    print(df)
    df_prepare = self.prepare.DataPreprocessor.prepare_time_series(
      df=df,
      date_col=ColumnData.DATE,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
      freq="D"
    )
    df_prepare = self.prepare.DataPreprocessor.detect_anomalies(
      df=df_prepare,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
    )
    self.data = df_prepare
    print(df_prepare)
    logging.info(f"Loaded %d rows", len(df))
    return df_prepare

  @property
  def load_excel(self) -> pd.DataFrame:
    logging.info(f"Loading {self.data_source}: {self.sheet_name}")
    df = pd.read_excel(self.data_source, sheet_name=self.sheet_name)
    # df = TestData(data=df).test_data
    df_prepare = self.prepare.prepare_time_series(
      df=df,
      date_col=ColumnData.DATE,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
      freq="D"
    )
    df_prepare = self.prepare.detect_anomalies(
      df=df_prepare,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
    )

    self.data = df_prepare
    # df = self.create_new_column
    logging.info(f"Loaded %d rows", len(df))
    return df_prepare

  @property
  def generate_sample_data(self) -> pd.DataFrame:
    formulas = Formulas()
    df = formulas.generate_data(
      periods=self.periods,
      plan_rate=self.plan_rate,
      seed=self.seed
    )
    df_prepare = self.prepare.prepare_time_series(
      df=df,
      date_col=ColumnData.DATE,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
      freq="D"
    )
    df_prepare = self.prepare.detect_anomalies(
      df=df_prepare,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
    )
    self.data = df_prepare

    # df = self.create_new_column
    return df_prepare

  @property
  def load_from_db(self) -> pd.DataFrame:
    self.db = DatabaseConnection()
    # self.db.create_table()
    df = self.db.load_sales()
    # df = TestData(data=df).test_data
    df_prepare = self.prepare.prepare_time_series(
      df=df,
      date_col=ColumnData.DATE,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
      freq="D"
    )
    df_prepare = self.prepare.detect_anomalies(
      df=df_prepare,
      value_col=ColumnData.SALES,
      plan_col=ColumnData.PLAN,
    )
    self.data = df_prepare
    # df = self.create_new_column
    return df_prepare

  @property
  def create_new_column(self) -> pd.DataFrame:
    df = self.data.copy()
    df[ColumnData.DATE] = pd.to_datetime(df[ColumnData.DATE])
    df[ColumnData.YEAR] = df[ColumnData.DATE].dt.year
    df[ColumnData.QUARTER] = df[ColumnData.DATE].dt.quarter
    df[ColumnData.MONTH] = df[ColumnData.DATE].dt.month
    df = (df
          .groupby([ColumnData.YEAR,
                    ColumnData.QUARTER,
                    ColumnData.MONTH,
                    ColumnData.DATE
                    ])
          .agg({ColumnData.SALES: Metrics.SUM,
                ColumnData.PLAN: Metrics.SUM})
          .reset_index())
    return df

class SalesAnalyzer:
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    self.analysis_results = {}

  @property
  def calculate_growth_rates(self) -> pd.DataFrame:
    df = self.calculate_rate
    logging.info("Calculated growth rate: %s", ColumnData.GROWTH_RATE)
    logging.info("Calculated growth rate diff: %s", ColumnData.GROWTH_RATE_PCT)
    logging.info("Calculated plan sales fulfillment")
    return df

  @property
  def descriptive_statistics(self) -> dict[str, float]:
    if ColumnData.SALES not in self.data.columns:
      raise KeyError(f"{ColumnData.SALES} not found in data")
    stats = {
      Metrics.MEAN: round(self.data[ColumnData.SALES].mean() / 1_000_000, 2),
      Metrics.MIN: round(self.data[ColumnData.SALES].min() / 1_000_000, 2),
      Metrics.MAX: round(self.data[ColumnData.SALES].max() / 1_000_000, 2),
      Metrics.SUM: round(self.data[ColumnData.SALES].sum() / 1_000_000, 2),
      Metrics.MEDIAN: round(self.data[ColumnData.SALES].median() / 1_000_000, 2),
    }
    logging.info("Descriptive statistics computed")
    self.analysis_results[ColumnData.STATISTICS] = stats
    return stats

  @property
  def calculate_rate(self) -> pd.DataFrame:
    df = self.data.copy()

    df[ColumnData.GROWTH_RATE_PCT] = df[ColumnData.SALES].pct_change()
    df[ColumnData.GROWTH_RATE] = df[ColumnData.GROWTH_RATE_PCT] + 1
    df[ColumnData.PLAN_SALES] = df[ColumnData.SALES] / df[ColumnData.PLAN]

    df[ColumnData.GROWTH_RATE_PCT] = df[ColumnData.GROWTH_RATE_PCT] * 100
    df[ColumnData.GROWTH_RATE] = df[ColumnData.GROWTH_RATE] * 100
    df[ColumnData.PLAN_SALES] = df[ColumnData.PLAN_SALES] * 100

    return df

class SalesForecaster:
  def __init__(
    self,
    data: pd.DataFrame | ExtensionArray | np.ndarray[tuple[Any, ...], Any],
    periods: int,
    models: dict[str, Any] = None,
    forecasts: dict[str, np.ndarray] = None,
    train_data: int = 0,
    test_data: int = 0,
    train_or_forecast: bool = True

  ) -> None:
    if forecasts is None:
      forecasts = {}
    if models is None:
      models = {}
    self.models = models
    self.forecasts = forecasts
    self.data = data
    self.periods = periods
    self.train_data = train_data
    self.test_data = test_data
    self.train_or_forecast = train_or_forecast

  # 1. LINEAR REGRESSION
  @property
  def linear_regression_forecast(self) -> np.ndarray:
    df = self.data.copy()

    if self.train_or_forecast:
      data_test = df[ColumnData.SALES].values[self.train_data:self.train_data + self.test_data]
      data_train = df[ColumnData.SALES].values[:self.train_data]
      forecast_steps = self.test_data
      # data_train_or_forecast = future_test
    else:
      # data_train_or_forecast = future_forecast
      data_test = None
      # data_train = df[ColumnData.SALES].values[:self.train_data]
      data_train = df[ColumnData.SALES].values[:]  # !!!

      forecast_steps = self.periods

    if self.train_or_forecast:
      if len(df) != (len(data_train) + len(data_test)):
        raise Exception("Data and test data are not equal")

    X_train = np.arange(len(data_train)).reshape(-1, 1)
    X_forecast = np.arange(len(data_train), len(data_train) + forecast_steps).reshape(-1, 1)

    model = LinearRegression()
    result_model = model.fit(X_train, data_train)

    result: LinearRegression | object | None = result_model

    forecast_data = result.predict(X_forecast)
    forecast_data = np.asarray(forecast_data)

    self.models[Model.LINEAR] = result
    self.forecasts[Model.LINEAR] = forecast_data
    logging.info("Linear regression forecast done (%d steps)", forecast_steps)
    return forecast_data

  # 2. EXPONENTIAL SMOOTHING
  @property
  def exponential_smoothing_forecast(self) -> np.ndarray:
    df = self.data.copy()
    if self.train_or_forecast:
      data_test = df[ColumnData.SALES].values[self.train_data:self.train_data + self.test_data]
      data_train = df[ColumnData.SALES].values[:self.train_data]
      forecast_steps = self.test_data
    else:
      data_test = None
      # data_train = df[ColumnData.SALES].values[:self.train_data]
      data_train = df[ColumnData.SALES].values[:]  # !!!
      forecast_steps = self.periods
    model = ExponentialSmoothing(
      endog=data_train,
      # trend="add",
      seasonal_periods=365,
      seasonal="add",
    )
    result = model.fit()
    forecast_data_test = result.forecast(steps=forecast_steps)
    forecast_data = np.asarray(forecast_data_test)

    self.models[Model.EXPONENTIAL_SMOOTHING] = result
    self.forecasts[Model.EXPONENTIAL_SMOOTHING] = forecast_data
    logging.info("Holt winters forecast done (%d steps)", forecast_steps)
    return forecast_data

  # 3. STATE SPACE
  @property
  def state_space_forecast(self) -> np.ndarray:
    df = self.data.copy()

    if self.train_or_forecast:
      data_test = df[ColumnData.SALES].values[self.train_data:self.train_data + self.test_data]
      data_train = df[ColumnData.SALES].values[:self.train_data]
      forecast_steps = self.test_data
      # data_train_or_forecast = future_test
    else:
      # data_train_or_forecast = future_forecast
      data_test = None
      # data_train = df[ColumnData.SALES].values[:self.train_data]
      data_train = df[ColumnData.SALES].values[:]  # !!!
      forecast_steps = self.periods
    model = UnobservedComponents(
      endog=data_train,
      # cycle=True,
      stochastic_cycle=True,
      stochastic_level=True,
      stochastic_seasonal=True,
      level="smooth trend",
      irregular=True,
      trend=True,
      stochastic_trend=True

    )
    result = model.fit()
    # forecast_data = UnobservedComponentsResults.forecast(result, steps=(self.periods + self.test_data))
    forecast_data = UnobservedComponentsResults.forecast(result, steps=forecast_steps)
    forecast_data = np.asarray(forecast_data)

    self.models[Model.STATE_SPACE] = result
    self.forecasts[Model.STATE_SPACE] = forecast_data
    logging.info("State-space forecast done (%d steps)", forecast_steps)
    return forecast_data

  # 4. ARIMA
  @property
  def arima_forecast(self) -> np.ndarray:
    df = self.data.copy()

    if self.train_or_forecast:
      data_test = df[ColumnData.SALES].values[self.train_data:self.train_data + self.test_data]
      data_train = df[ColumnData.SALES].values[:self.train_data]
      forecast_steps = self.test_data
      # data_train_or_forecast = future_test
    else:
      # data_train_or_forecast = future_forecast
      data_test = None
      # data_train = df[ColumnData.SALES].values[:self.train_data]
      data_train = df[ColumnData.SALES].values[:]  # !!!
      forecast_steps = self.periods
    model = ARIMA(
      endog=data_train,
      order=(1, 1, 0),
    )
    result = model.fit()
    # forecast_data = result.forecast(steps = (self.periods + self.test_data))
    forecast_data = result.forecast(steps=forecast_steps)
    forecast_data = np.asarray(forecast_data)

    self.models[Model.ARIMA] = result
    self.forecasts[Model.ARIMA] = forecast_data
    logging.info("ARIMA forecast done (%d steps)", forecast_steps)
    return forecast_data

class ModelEvaluator:
  def __init__(
    self,
    metrics: dict[str, dict[str, float]] | None = None,
    actual: np.ndarray | None = None,
    predicted: np.ndarray | None = None,
    model_name: str | None = None,
  ) -> None:
    if metrics is None:
      metrics = {}
    self.metrics = metrics
    self.actual = actual
    self.predicted = predicted
    self.model_name = model_name

  @property
  def calculate_metrics(self) -> dict[str, float]:
    actual = np.asarray(self.actual).ravel()
    predicted = np.asarray(self.predicted).ravel()

    try:
      mape = mean_absolute_percentage_error(actual, predicted)
      mae = mean_absolute_error(actual, predicted) / 1_000_000
      r2 = r2_score(actual, predicted)
      rmse = root_mean_squared_error(actual, predicted) / 1_000_000
    except ZeroDivisionError:
      raise ZeroDivisionError('Cannot calculate metrics')

    results = {
      Metrics.MAPE: float(round(
        number=mape,
        ndigits=2)),
      Metrics.RMSE: float(round(
        number=rmse,
        ndigits=2)),
      Metrics.R2: float(round(
        number=r2,
        ndigits=2)),
      Metrics.MAE: float(round(
        number=mae,
        ndigits=2)),
    }

    self.metrics[self.model_name] = results
    logging.info("Metrics for %s: %s", self.model_name, results)
    return results

  @property
  def compare_models(self) -> pd.DataFrame:
    comparison_df = pd.DataFrame(self.metrics).T
    if Metrics.RMSE in comparison_df.columns:
      comparison_df = comparison_df.sort_values(
        Metrics.RMSE,
        ascending=True)
    return comparison_df

class SalesVisualizer:
  def __init__(
    self,
    data: pd.DataFrame,
    periods: int,
    forecasts: dict[str, np.ndarray] = None,
    evaluations: dict[str, float] = None,
    statistics: dict[str, float] = None,
    growth_rates: pd.DataFrame | None = None,
    train_or_forecast: bool = True
  ) -> None:
    # self.figure_size = (12, 6)
    self.data = data
    self.periods = periods
    self.forecasts = forecasts
    self.evaluations = evaluations
    self.statistics = statistics
    self.growth_rates = growth_rates
    self.train_data = int(len(data) * 2 / 3)
    self.test_data = len(data) - self.train_data
    self.train_or_forecast = train_or_forecast

  @property
  def plot_analysis_results(self) -> plt.Figure:
    growth_rates = pd.DataFrame(self.growth_rates)

    growth_rates = (growth_rates
                    .groupby([ColumnData.YEAR,
                              ColumnData.QUARTER,
                              ])
                    .agg({ColumnData.SALES: Metrics.SUM,
                          ColumnData.PLAN: Metrics.SUM})
                    .reset_index())

    growth_rates[ColumnData.SALES] = growth_rates[ColumnData.SALES].values / 1_000_000
    growth_rates[ColumnData.PLAN] = growth_rates[ColumnData.PLAN].values / 1_000_000

    growth_rates[ColumnData.SALES] = growth_rates[ColumnData.SALES].round(2)
    growth_rates[ColumnData.PLAN] = growth_rates[ColumnData.PLAN].round(2)

    growth_rates[ColumnData.PLAN_SALES] = (growth_rates[ColumnData.SALES] / growth_rates[ColumnData.PLAN] * 100).round(
      2)
    growth_rates[ColumnData.GROWTH_RATE_PCT] = (growth_rates[ColumnData.SALES].pct_change() * 100).round(2)
    growth_rates[ColumnData.GROWTH_RATE] = ((growth_rates[ColumnData.GROWTH_RATE_PCT] + 1) * 100).round(2)

    growth_rates[ColumnData.YEAR] = growth_rates[ColumnData.YEAR].round(0)
    growth_rates[ColumnData.QUARTER] = growth_rates[ColumnData.QUARTER].round(0)

    plt.style.use('seaborn-v0_8')

    fig, ax = plt.subplots()
    ax.axis("off")
    table = plt.Axes.table(
      ax=ax,
      cellText=growth_rates.values,
      colLabels=growth_rates.columns,
      loc="center",
    )

    # table.auto_set_font_size(True)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    ax.set_title(ColumnData.ANALYSIS_RESULTS)
    plt.tight_layout()

    save_path = self.get_save_path("analysis_table.png")
    plt.savefig(save_path)

    return fig

  @property
  def plot_statistics_table(self) -> plt.Figure:
    statistics = self.statistics.copy()
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots()
    ax.axis("off")
    items = list(statistics.items())
    table = plt.Axes.table(
      ax=ax,
      cellText=[
        [key, value] for key, value in items
      ],
      colLabels=[ColumnData.STATISTICS, "Value,mln"],
      cellLoc="center",
      loc="center",
      colLoc="center"
    )
    table.auto_set_font_size(True)
    # table.scale(1.2, 1.2)
    # table.set_fontsize(14)
    ax.set_title(Metrics.DESCRIPTIVE_STATISTICS)
    save_path = self.get_save_path("statistics_table.png")
    plt.savefig(save_path)
    return fig

  @property
  def plot_forecast_comparison(self) -> plt.Figure:
    df = self.data.copy()
    forecasts = self.forecasts.copy()
    self.plot_forecast_table(df=df, forecasts=forecasts)
    # ==========
    # 1. FIGURE AND AXES:
    fig, ax = plt.subplots()

# ==============================
    # step 1: SET POINTS (begin and len)

    # ========== 1.1. TRAINING ==========
    if self.train_or_forecast:
      begin_forecast_date = df[ColumnData.DATE].iloc[self.train_data]
      data_test = df[ColumnData.SALES].values[self.train_data:(self.test_data + self.train_data)]
      data_train_or_forecast = len(data_test)
    # ========== (df = train + test (not empty))
      train_df = df.iloc[:self.train_data]
      test_df = df.iloc[self.train_data:(self.test_data + self.train_data)]

    # ========== 1.2. FORECAST ==========
    else:
      begin_forecast_date = df[ColumnData.DATE].iloc[-1]
      data_train_or_forecast = self.periods
      # ========== (df = train + test (empty))
      train_df = df
      test_df = pd.DataFrame() # empty

    # step 2: SET PLOT (x = date; y = sales)
    ax.plot(
      train_df[ColumnData.DATE],
      train_df[ColumnData.SALES],
      linewidth=2, markersize=3, marker="o",
      label="Training Data", alpha=0.7
    )

    # step 3: SET PLOT FOR FORECAST
    if self.train_or_forecast and not test_df.empty:
      ax.plot(
        test_df[ColumnData.DATE],
        test_df[ColumnData.SALES],
        linewidth=2, markersize=1, marker="o",
        label="Actual Test Data", color="orange", alpha=0.5
      )

    # step 4: PAINT TIME SERIES
    start_forecast_date = begin_forecast_date + pd.Timedelta(days=1)
    x_axe = pd.date_range(
      start=start_forecast_date,
      periods=data_train_or_forecast,
      freq="D")

    for model, forecast in forecasts.items():

        if model == Model.STATE_SPACE:
            continue

        last_actual_value_sales = train_df[ColumnData.SALES].iloc[-1] if not train_df.empty else 0
        first_forecast = forecast[0]
        offset = last_actual_value_sales - first_forecast
        offset_forecast = forecast + offset
        # forecast_absolute = last_actual_value_sales + forecast[:data_train_or_forecast]
        ax.plot(
            x_axe[:len(offset_forecast[:data_train_or_forecast])],
            offset_forecast[:data_train_or_forecast],
            # forecast_absolute,
            linewidth=1, markersize=1, marker="o", alpha=0.5,
            label=model)

    ax.axvline(
      x=begin_forecast_date,
      linestyle="--",
      linewidth=1,
      alpha=0.5,
    )
    label_text = "Train/Test Split" if self.train_or_forecast else "Start Forecast"
    ax.text(
      begin_forecast_date,
      ax.get_ylim()[1] * 0.95,
      label_text,
      rotation=90,
      verticalalignment="top",
      fontsize=10,
      alpha=0.5,
    )

    if self.train_or_forecast and not test_df.empty:
      test_start = test_df[ColumnData.DATE].iloc[0]
      ax.axvspan(test_start, test_df[ColumnData.DATE].iloc[-1], alpha=0.1, color="green", label="Test Period")
    else:
      ax.axvspan(begin_forecast_date, x_axe[-1], alpha=0.1, color="red", label="Forecast Period")

    ax.legend()
    test_count = self.test_data if self.train_or_forecast else len(self.data)
    ax.set_title(ColumnData.FORECAST_COMPARISON + f" (test: {test_count}, forecast: {data_train_or_forecast} days)")
    plt.xlabel(ColumnData.DATE)
    plt.ylabel(ColumnData.SALES)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    fig.autofmt_xdate()
    plt.tight_layout()

    save_path = self.get_save_path("forecast-comparison_table.png")
    plt.savefig(save_path)

    return fig

  @staticmethod
  def plot_forecast_table(df: pd.DataFrame, forecasts: dict[str, np.ndarray]) -> plt.Figure:

    if not forecasts:
      raise ValueError("No forecasts provided")

    df = df.copy()
    forecasts = forecasts.copy()
    begin_forecast_date = df[ColumnData.DATE].iloc[-1]
    start_forecast_date = begin_forecast_date + pd.Timedelta(days=1)
    x_axe = pd.date_range(
      start=start_forecast_date,
      periods=len(next(iter(forecasts.values()))),
      freq="D")

    forecast_df_round = pd.DataFrame(forecasts, index=x_axe)
    forecast_df_temp = forecast_df_round / 1_000_000
    df_rounded = forecast_df_temp.round(2)
    forecast_df = df_rounded.copy()
    forecast_df.index.name = ColumnData.DATE

    forecast_df[ColumnData.YEAR] = forecast_df.index.year
    sum_forecast_all = (forecast_df.groupby(ColumnData.YEAR)
                        .sum()
                        .reset_index())
    mean_forecast_all = (forecast_df.groupby(ColumnData.YEAR)
                         .mean()
                         .reset_index())

    forecast_df[ColumnData.QUARTER] = forecast_df.index.quarter
    mean_model = ((forecast_df.groupby([ColumnData.YEAR, ColumnData.QUARTER])
                   .agg({Model.EXPONENTIAL_SMOOTHING: Metrics.MEAN}))
                  .reset_index())

    sum_forecast_all = sum_forecast_all.round(2)
    mean_forecast_all = mean_forecast_all.round(2)
    mean_model = mean_model.round(2)

    fig, axes = plt.subplots(3, 1)
    for ax in axes:
      ax.axis("off")

    axes[0].set_title(f"Forecast for {ColumnData.YEAR} ({Metrics.SUM})")
    tbl_sum_all = plt.Axes.table(
      ax=axes[0],
      cellText=sum_forecast_all.values,
      colLabels=sum_forecast_all.columns,
      cellLoc="center",
      loc="center")

    axes[1].set_title(f"Forecast for {ColumnData.YEAR} ({Metrics.MEAN})")
    tbl_mean_all = plt.Axes.table(
      ax=axes[1],
      cellText=mean_forecast_all.values,
      colLabels=mean_forecast_all.columns,
      cellLoc="center",
      loc="center")

    axes[2].set_title(f"Model: {Model.EXPONENTIAL_SMOOTHING} for {ColumnData.YEAR}/{ColumnData.QUARTER} ({Metrics.MEAN})\n")
    tbl_model = plt.Axes.table(
      ax=axes[2],
      cellText=mean_model.values,
      colLabels=mean_model.columns,
      cellLoc="center",
      loc="center")

    tbl_sum_all.auto_set_column_width(
      list(range(len(sum_forecast_all.columns))),
    )
    tbl_mean_all.auto_set_column_width(
      list(range(len(mean_forecast_all.columns))),
    )
    tbl_model.auto_set_column_width(
      list(range(len(mean_model.columns))),
    )

    tbl_sum_all.auto_set_font_size(True)
    tbl_mean_all.auto_set_font_size(True)
    tbl_model.auto_set_font_size(True)
    # fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.95, bottom = 0.15, top = 0.95,
                        wspace = 0.4, hspace = 0.7)

    save_path = SalesVisualizer.get_save_path("forecast_table.png")
    fig.savefig(save_path)
    return fig

  @property
  def plot_evaluation_table(self) -> plt.Figure:
    evaluation = self.evaluations.copy()
    df = pd.DataFrame(evaluation).T
    normal_mean_metrics = self.normal_mean_metrics(df)

    fig, ax = plt.subplots()
    ax.axis("off")
    table = plt.Axes.table(
      ax=ax,
      cellText=df.values,
      colLabels=df.columns,
      rowLabels=df.index,
      cellLoc="center",
      loc="center",
      colLoc="center",
    )
    table.auto_set_font_size(True)
    row_metric = list(df.index).index(normal_mean_metrics)
    table.auto_set_column_width([row_metric])

    for (row, column), cell in table.get_celld().items():
      if row == row_metric + 1:
        cell.set_color("lightgreen")
        cell.set_text_props(weight="bold")

    ax.set_title(Model.COMPARISON_METRICS)
    save_path = SalesVisualizer.get_save_path("evaluation_table.png")
    fig.savefig(save_path)
    return fig

  @staticmethod
  def normal_mean_metrics(df: pd.DataFrame) -> str:

    df = df.copy()
    df[Metrics.MAPE_NORM] = -df[Metrics.MAPE]
    df[Metrics.MAE_NORM] = -df[Metrics.MAE]
    df[Metrics.RMSE_NORM] = -df[Metrics.RMSE]
    df[Metrics.R2_NORM] = df[Metrics.R2]

    metrics_norm = [[Metrics.MAPE_NORM],
                    [Metrics.MAE_NORM],
                    [Metrics.RMSE_NORM],
                    [Metrics.R2_NORM]]

    for metric_norm in metrics_norm:
      df[metric_norm] = (
        (df[metric_norm] - df[metric_norm].min()) /
        (df[metric_norm].max() - df[metric_norm].min())
      )

    df[Metrics.SCORE] = df[
      [Metrics.MAPE_NORM,
       Metrics.MAE_NORM,
       Metrics.RMSE_NORM,
       Metrics.R2_NORM]].mean(axis=1)
    return df[Metrics.SCORE].idxmax()

  @staticmethod
  def path_results(
    folder: str = "results",
    subfolder: str = "visualizations",
  ) -> Path:

    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    results_dir = project_dir / folder / subfolder
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir

  @staticmethod
  def get_save_path(
    filename: str = "",
    folder: str = "results",
    subfolder: str = "visualizations",
  ) -> Path:

    dir_path = SalesVisualizer.path_results(folder, subfolder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{timestamp}.{ext}"
    return dir_path / unique_filename

class SalesAnalysisForecastUI:
  def __init__(self) -> None:
    self.data = None
    self.win = tk.Tk()
    self.win.title("Sales analysis and forecasting")
    self.win.geometry("700x500")
    self.arima_params = (0, 0, 0)
    self.button_style = {
      "width": 15,
      "padding": 10,
    }

    ttk.Label(self.win, text="Data source:").grid(row=0, column=0)
    ttk.Button(self.win, text="Load CSV", command=self.load_csv, **self.button_style).grid(row=0, column=1)
    ttk.Button(self.win, text="Load Excel", command=self.load_excel, **self.button_style).grid(row=0, column=2)
    ttk.Button(self.win, text="Load from MySQL", command=self.load_mysql, **self.button_style).grid(row=0, column=3)
    ttk.Button(self.win, text="Generate sample", command=self.load_sample, **self.button_style).grid(row=0, column=4)

    ttk.Label(self.win, text="Period:").grid(row=15, rowspan=2, column=0)
    self.period = tk.IntVar()
    rb_period_year = tk.Radiobutton(
      self.win,
      text=Period.YEARLY,
      variable=self.period,
      value=Period.YEARLY)
    rb_period_quartal = tk.Radiobutton(
      self.win,
      text=Period.QUARTERLY,
      variable=self.period,
      value=Period.QUARTERLY
    )
    rb_period_month = tk.Radiobutton(
      self.win,
      text=Period.MONTHLY,
      variable=self.period,
      value=Period.MONTHLY
    )
    rb_period_week = tk.Radiobutton(
      self.win,
      text=Period.SEASON_MONTHLY,
      variable=self.period,
      value=Period.SEASON_MONTHLY
    )
    # rb_period_five_years.grid(row=15, rowspan=2, column=1)
    rb_period_year.grid(row=15,
                        # rowspan=2,
                        column=1)
    rb_period_quartal.grid(row=15,
                           # rowspan=2,
                           column=2)
    rb_period_month.grid(row=15,
                         # rowspan=2,
                         column=3)
    rb_period_week.grid(row=15,
                        # rowspan=2,
                        column=4)
    ttk.Button(
      self.win,
      text="Set period",
      command=self.set_period,
      **self.button_style
    ).grid(row=15,
           # rowspan=2,
           column=5)

    ttk.Button(self.win,
               text="Analysis",
               command=self.run_analysis,
               **self.button_style
               ).grid(row=30, rowspan=2, column=1)
    ttk.Button(self.win,
               text="Training",
               command=self.run_train,
               **self.button_style
               ).grid(row=30, rowspan=2, column=2)
    ttk.Button(self.win,
               text="Forecast",
               command=self.run_forecast,
               **self.button_style
               ).grid(row=30, rowspan=2, column=3)

  # ==============================

  def load_csv(self) -> None:
    path = filedialog.askopenfilename()
    self.data = DataLoader(data_source=path).load_csv
    messagebox.showinfo("OK", "CSV loaded")

  def load_excel(self) -> None:
    path = filedialog.askopenfilename()
    self.data = DataLoader(data_source=path).load_excel
    messagebox.showinfo("OK", "Excel loaded")

  def load_mysql(self) -> None:
    self.data = DataLoader().load_from_db
    messagebox.showinfo("OK", "Data from MySQL loaded")

  def load_sample(self) -> None:
    self.data = DataLoader().generate_sample_data
    messagebox.showinfo("OK", "Sample data generated")

  def run_analysis(self) -> None:
    messagebox.showinfo("OK", "Analysis started")
    model = ModelExample(
      data=self.data,
      period=self.period.get(),
      # model=self.method.get(),
      type_run=1
    )
    results = model.run_analysis
    for fig in results["figures"].values():
      show_figure_blocking(fig)

  def run_train(self) -> None:
    messagebox.showinfo("OK", "Training started")
    model = ModelExample(
      data=self.data,
      period=self.period.get(),
      # model=self.method.get(),
      type_run=2,
      train_or_forecast=True
    )
    results = model.run_analysis
    for fig in results["figures"].values():
      show_figure_blocking(fig)

  def run_forecast(self) -> None:
    messagebox.showinfo("OK", "Forecast started")
    model = ModelExample(
      data=self.data,
      period=self.period.get(),
      # model=self.method.get(),
      type_run=3,
      train_or_forecast=False
    )
    results = model.run_analysis
    for fig in results["figures"].values():
      show_figure_blocking(fig)

  def set_period(self) -> None:
    print("Setting period")
    messagebox.showinfo("OK", "Period set to: " + str(self.period.get()))

  def run(self) -> None:
    self.win.mainloop()

if __name__ == "__main__":
  SalesAnalysisForecastUI().run()
