"""This module contains the objects used to pre-process and transform the data
for modeling.
"""
# --------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------

# Basic libraries
import re
from datetime import datetime

# Data libraries
import numpy as np
import pandas as pd
from pandas import DataFrame

# --------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------


# ------------------------------ Data Cleaning Functions -------------------------------
def camel_to_snake(name: str) -> str:
    """
    Convert a string from CamelCase to snake_case.

    Args:
        name (str): The CamelCase string to be converted.

    Returns:
        str: The converted snake_case string.
    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def tuple_to_datetime(date_tuple: tuple) -> datetime:
    """
    Converts a tuple representing a date into a datetime object.

    Args:
        date_tuple (tuple): A tuple representing a date in the format
            ("Three Months Ended <Month> <Day>", "<Year>").

    Returns:
        datetime: A datetime object representing the converted date.
    """
    # Assuming the format is always "Three Months Ended <Month> <Day>,"
    date_string = (
        date_tuple[0].split("Ended ")[1].strip(",").strip() + " " + date_tuple[1]
    )

    # Parse the date
    date_object = datetime.strptime(date_string, "%B %d %Y")

    return date_object


def clean_values_series(s: pd.Series) -> pd.Series:
    output = s.str.replace("(", "-").str.replace(pat="[^0-9.-]", repl="", regex=True)
    return pd.to_numeric(output, errors="coerce")


def camel_to_snake(name):
    """
    Convert a given string from CamelCase to snake_case.

    Parameters:
    - name (str): The input string in CamelCase.

    Returns:
    - str: The converted string in snake_case.
    """
    # Replace CamelCase with Camel_Case
    name_with_space = re.sub(r"(?<!^)([A-Z][a-z]+)", r"_\1", name)

    # Replace Acronyms like CIA with _CIA
    name_with_acronyms = re.sub(
        pattern=r"([A-Z]+)([A-Z][a-z]|$)", repl=r"_\1\2", string=name_with_space
    )

    # Lowercase all characters
    return name_with_acronyms.lower()


def clean_account(account: str) -> str:
    """
    Clean the account string by dropping prefixes, converting to snake
    case, and stripping empty spaces.

    Parameters:
        account (str): The account string to be cleaned.

    Returns:
        str: The cleaned account string.
    """
    # Drop prefixes
    account = re.sub(r"^.*?:", "", account)

    # To snake case
    account = camel_to_snake(account)

    # Strip empty spaces and line breaks
    account = account.strip().replace("\n", " ").replace(" ", "_")

    return account


# --------------------------- Technical Analysis Functions -----------------------------
def garman_klass_volatility(df: DataFrame) -> float:
    """
    Calculate the Garman-Klass volatility of a given DataFrame.

    Parameters:
    - df: DataFrame containing the necessary OHLC (Open, High, Low,
        and Close) data.

    Returns:
    - volatility: The calculated Garman-Klass volatility.
    """
    # Calculate the logarithm of the ratio between High and Low prices,
    # squared and assigned to the log_hl variable.
    log_hl = np.log(df["high"] / df["low"])  # ** 2

    # Calculate the logarithm of the ratio between Adjusted Close and Open
    # prices, squared and assigned to the log_co variable.
    log_co = np.log(df["close"] / df["open"])  # ** 2

    # Calculate the Garman-Klass volatility using the log_hl and log_co
    # variables, and assign the result to the volatility variable.
    volatility = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    # Return the calculated volatility.
    return volatility


def rolling_garman_klass(df, window_size):
    """
    Calculate the rolling Garman-Klass volatility for a given DataFrame.

    Parameters:
        - df (DataFrame): The input DataFrame containing the data.
        - window_size (int): The size of the rolling window used for
            the rolling calculation.

    Returns:
        - Series: The rolling Garman-Klass volatility.

    This function applies the Garman-Klass formula to each row of the
    DataFrame to calculate the Garman-Klass volatility. It then
    calculates the rolling Garman-Klass volatility by taking the mean of
    the Garman-Klass volatility values over a rolling window of size
    window_size. The calculated rolling Garman-Klass volatility is then
    returned as a Series.

    Note: The Garman-Klass formula is applied using the
    `garman_klass_volatility` function.

    Example usage:
        df = pd.DataFrame(...)
        rolling_gk_vol = rolling_garman_klass(df, window_size=10)
    """
    # Apply the Garman-Klass formula to each row
    df["GK"] = df.apply(garman_klass_volatility, axis=1)

    # Calculate rolling Garman-Klass Volatility
    df["Rolling_GK_Vol"] = df["GK"].rolling(window=window_size).mean().apply(np.sqrt)

    return df["Rolling_GK_Vol"]


def identify_rsi_signals(df, overbought_threshold=70, oversold_threshold=30):
    """
    Generates a signal column in the given DataFrame based on the
    Relative Strength Index (RSI).

    Parameters:
        - df: The DataFrame containing the data.
        - overbought_threshold (optional): The RSI threshold for the
            overbought signal. Default is 70.
        - oversold_threshold (optional): The RSI threshold for the
            oversold signal. Default is 30.

    Returns:
        - df: The DataFrame with the signal column added.
    """
    # Overbought signal
    df.loc[df["rsi"] > overbought_threshold, "rsi_signal"] = "sell"

    # Oversold signal
    df.loc[df["rsi"] < oversold_threshold, "rsi_signal"] = "buy"

    # Potential reversal signal (RSI crosses back from extreme levels)
    df["signal_shifted"] = df["rsi_signal"].shift(1)
    bearish_mask = (df["rsi_signal"] == "") & (df["signal_shifted"] == "sell")
    bullish_mask = (df["rsi_signal"] == "") & (df["signal_shifted"] == "buy")
    df.loc[bearish_mask, "rsi_signal"] = "bearish_reversal"
    df.loc[bullish_mask, "rsi_signal"] = "bullish_reversal"

    # Replace empty strings with NaN
    df["rsi_signal"] = df["rsi_signal"].replace("", np.nan)

    # Drop the 'signal_shifted' column
    df.drop("signal_shifted", axis=1, inplace=True)

    return df


def identify_macd_signals(df):
    """
    Identify MACD signals in a DataFrame.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the MACD data.

    Returns:
    - pandas DataFrame
        The original DataFrame with an additional column
            'macd_signal_cross' indicating the MACD signal cross type.
    """
    df["macd_signal_cross"] = ""

    # Identifying the crossover points
    df["prev_macd"] = df["macd"].shift(1)
    df["prev_macd_signal"] = df["macd_signal"].shift(1)

    # Bullish crossover (MACD crosses above the signal line)
    bullish_mask = (df["macd"] > df["macd_signal"]) & (
        df["prev_macd"] < df["prev_macd_signal"]
    )
    bearish_mask = (df["macd"] < df["macd_signal"]) & (
        df["prev_macd"] > df["prev_macd_signal"]
    )
    df.loc[bullish_mask, "macd_signal_cross"] = "bullish"

    # Bearish crossover (MACD crosses below the signal line)
    df.loc[bearish_mask, "macd_signal_cross"] = "bearish"

    # Replace empty strings with NaN
    df["macd_signal_cross"] = df["macd_signal_cross"].replace("", np.nan)

    return df.drop(["prev_macd", "prev_macd_signal"], axis=1)


def identify_bollinger_band_signals(df):
    """
    Identifies Bollinger Band signals based on the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the Bollinger
            Band data.

    Returns:
        pandas.DataFrame: The DataFrame with the 'signal' column updated
            with the identified signals.
    """
    df["bb_signal"] = ""

    # Overbought signal
    df.loc[df["close"] > df["bb_upperband"], "bb_signal"] = "overbought"

    # Oversold signal
    df.loc[df["close"] < df["bb_lowerband"], "bb_signal"] = "oversold"

    # Replace empty strings with NaN
    df["bb_signal"] = df["bb_signal"].replace("", np.nan)

    return df


# ------------------------------ Mathematical Functions --------------------------------
def optimize_function(
    func,
    df: pd.DataFrame,
    price_name: str,
    returns_name: str,
    prefix: str,
    bounds: tuple,
    tests: int,
    best_guess: float,
    relevant_diff: float = 0.001,
):
    # Define the search space
    input_values = np.linspace(bounds[0], bounds[1], num=tests)

    # Compute loss in search space: find value that minimizes output
    min_output = 1e100000
    for i, n in enumerate(input_values):
        # Compute loss
        output, output_df = func(
            raw_df=df,
            price_name=price_name,
            returns_name=returns_name,
            prefix=prefix,
            std_value=n,
        )

        # Move towards global minimum
        if output < min_output:
            min_index = i
            min_value = n
            diff = min_output - output
            min_output = output
            best_df = output_df

            # Return output if there is very small or no progress
            if diff <= relevant_diff:
                return min_value, min_output, best_df

    # Test for very small change in initial guess and minimum found
    if max((best_guess, min_value)) / min((best_guess, min_value)) - 1 < 0.01:
        return min_value, min_output, best_df

    # Define bounds for next zoomed-in iteration
    if min_index == 0:
        lower_bound = bounds[0]
    elif min_index == len(input_values) - 1:
        print(f"Minimum value didn't converge, got to {min_value}")
        return min_value, min_output, best_df
    else:
        lower_bound = input_values[min_index - 1]
    upper_bound = input_values[min_index + 1]

    # Recurse in zoomed-in search space
    min_value, min_output, best_df = optimize_function(
        func=func,
        df=df,
        price_name=price_name,
        returns_name=returns_name,
        prefix=prefix,
        tests=tests,
        bounds=(lower_bound, upper_bound),
        best_guess=min_value,
        relevant_diff=relevant_diff,
    )

    return min_value, min_output, best_df


def compute_rmse_from_std(
    raw_df: pd.DataFrame,
    price_name: str,
    returns_name: str,
    prefix: str,
    std_value: str,
) -> tuple:
    # Compute the Fast Fourier Transform
    df = generate_fourier_transform(
        raw_df=raw_df,
        price_name=price_name,
        returns_name=returns_name,
        prefix=prefix,
        std_value=std_value,
    )
    # Compute RMSE
    rmse = np.sqrt(np.mean((df[price_name] - df[prefix + "_fft_regression"]) ** 2))

    return np.array(rmse), df


def generate_fourier_transform(
    raw_df: pd.DataFrame,
    price_name: str,
    returns_name: str,
    prefix: str,
    std_value: str,
) -> pd.DataFrame:
    # Generate data frame
    df = raw_df.copy()
    first_date = df.index[1]

    # Compute the discrete Fourier Transform with the fast Fourier transform
    dft = np.fft.fft(df.loc[first_date:, returns_name])

    # Compute theta, frequency and amplitude
    df.loc[first_date:, prefix + "_theta"] = np.arctan(dft.imag / dft.real)
    df.loc[first_date:, prefix + "_amplitude"] = np.sqrt(
        dft.real**2 + dft.imag**2
    ) / (len(df) / 2)
    df.loc[first_date:, prefix + "_frequency"] = np.fft.fftfreq(dft.size, d=1)

    # -------------- Get dominant values based on input standard deviation -------------
    mean_amp = df[prefix + "_amplitude"].mean()
    std_amp = df[prefix + "_amplitude"].std()

    # Generate masks for dominant and positive frequency values
    dominant_amp_mask = df[prefix + "_amplitude"] > (std_value * std_amp + mean_amp)
    positive_freq_mask = df[prefix + "_frequency"] > 0

    # Select only dominant values
    cols = ["amplitude", "frequency", "theta"]
    cols = [prefix + "_" + col for col in cols]
    filtered_df = df.loc[dominant_amp_mask & positive_freq_mask, cols]

    # Calculate regression delta
    reg_delta = 0
    for n in range(len(filtered_df)):
        shift = filtered_df.iloc[n][prefix + "_theta"]
        reg_delta += filtered_df.iloc[n][prefix + "_amplitude"] * np.cos(
            n * np.array(range(len(df) - 1)) + shift
        )

    # Convert time delta to time from starting point
    df.loc[first_date:, prefix + "_fft_regression"] = df.loc[
        df.index.min(), price_name
    ] * np.cumprod(1 + reg_delta)
    df.loc[df.index.min(), prefix + "_fft_regression"] = df.loc[
        df.index.min(), price_name
    ]

    return df


# --------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------
