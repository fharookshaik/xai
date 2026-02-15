"""
Feature engineering utilities for NYC Taxi dataset
"""
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points
    
    Parameters:
    -----------
    lat1, lon1, lat2, lon2 : float or array-like
        Coordinates in degrees
    
    Returns:
    --------
    distance : float or array
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    r = 6371
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Manhattan distance
    
    Parameters:
    -----------
    lat1, lon1, lat2, lon2 : float or array-like
        Coordinates in degrees
    
    Returns:
    --------
    distance : float or array
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    lat_diff = np.abs(lat2 - lat1) * 111  # 1 degree latitude ≈ 111 km
    lon_diff = np.abs(lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))  # Adjust for longitude
    
    return lat_diff + lon_diff


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing (direction) from point 1 to point 2
    
    Parameters:
    -----------
    lat1, lon1, lat2, lon2 : float or array-like
        Coordinates in degrees
    
    Returns:
    --------
    bearing : float or array
        Bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    
    return bearing


def is_rush_hour(hour):
    """
    Determine if hour is during rush hour
    
    Parameters:
    -----------
    hour : int or array-like
        Hour of day (0-23)
    
    Returns:
    --------
    bool or array of bool
    """
    morning_rush = (hour >= 7) & (hour <= 9)
    evening_rush = (hour >= 17) & (hour <= 19)
    return morning_rush | evening_rush


def get_time_of_day(hour):
    """
    Categorize hour into time of day
    
    Parameters:
    -----------
    hour : int or array-like
        Hour of day (0-23)
    
    Returns:
    --------
    category : str or array of str
        'early_morning', 'morning', 'afternoon', 'evening', 'night'
    """
    conditions = [
        (hour >= 0) & (hour < 6),
        (hour >= 6) & (hour < 12),
        (hour >= 12) & (hour < 17),
        (hour >= 17) & (hour < 21),
        (hour >= 21) & (hour <= 23)
    ]
    choices = ['early_morning', 'morning', 'afternoon', 'evening', 'night']
    
    if isinstance(hour, (int, float)):
        for i, cond in enumerate(conditions):
            if cond:
                return choices[i]
    else:
        return np.select(conditions, choices, default='unknown')


def is_weekend(day_of_week):
    """
    Determine if day is weekend
    
    Parameters:
    -----------
    day_of_week : int or array-like
        Day of week (0=Monday, 6=Sunday)
    
    Returns:
    --------
    bool or array of bool
    """
    return (day_of_week == 5) | (day_of_week == 6)


def is_airport_trip(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, threshold_km=5.0):
    """
    Determine if trip is to/from airport
    
    Parameters:
    -----------
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon : float or array-like
        Coordinates in degrees
    threshold_km : float
        Distance threshold in km
    
    Returns:
    --------
    bool or array of bool
    """
    airports = {
        'JFK': (40.6413, -73.7781),
        'LaGuardia': (40.7769, -73.8740),
        'Newark': (40.6895, -74.1745)
    }
    
    is_airport = np.zeros_like(pickup_lat, dtype=bool)
    
    for airport_name, (airport_lat, airport_lon) in airports.items():
        # Distance from pickup to airport
        pickup_dist = haversine_distance(pickup_lat, pickup_lon, airport_lat, airport_lon)
        # Distance from dropoff to airport
        dropoff_dist = haversine_distance(dropoff_lat, dropoff_lon, airport_lat, airport_lon)
        
        # Mark as airport trip if either pickup or dropoff is within threshold
        is_airport = is_airport | (pickup_dist <= threshold_km) | (dropoff_dist <= threshold_km)
    
    return is_airport


def get_distance_from_center(lat, lon, center_lat=40.758, center_lon=-73.9855):
    """
    Calculate distance from city center (Times Square)
    
    Parameters:
    -----------
    lat, lon : float or array-like
        Coordinates in degrees
    center_lat, center_lon : float
        Center coordinates
    
    Returns:
    --------
    distance : float or array
        Distance in kilometers
    """
    return haversine_distance(lat, lon, center_lat, center_lon)


def create_distance_features(df):
    """
    Create all distance-related features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added distance features
    """
    # Haversine distance
    df['haversine_distance'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Manhattan distance
    df['manhattan_distance'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Bearing
    df['bearing'] = calculate_bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Distance from center
    df['pickup_distance_from_center'] = get_distance_from_center(
        df['pickup_latitude'], df['pickup_longitude']
    )
    
    df['dropoff_distance_from_center'] = get_distance_from_center(
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    return df


def create_temporal_features(df, pickup_datetime_column='pickup_datetime'):
    """
    Create temporal features from datetime
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain pickup_datetime column
    pickup_datetime_column : str
        Name of datetime column
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added temporal features
    """
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[pickup_datetime_column]):
        df[pickup_datetime_column] = pd.to_datetime(df[pickup_datetime_column])
    
    # Extract temporal features
    df['hour'] = df[pickup_datetime_column].dt.hour
    df['day_of_week'] = df[pickup_datetime_column].dt.dayofweek
    df['month'] = df[pickup_datetime_column].dt.month
    df['year'] = df[pickup_datetime_column].dt.year
    
    # Derived features
    df['is_rush_hour'] = is_rush_hour(df['hour'])
    df['is_weekend'] = is_weekend(df['day_of_week'])
    df['time_of_day'] = get_time_of_day(df['hour'])
    
    return df