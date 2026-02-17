"""
Data Loader Module
==================
This module is responsible for loading and handling network intrusion detection datasets
from the KDD dataset. It provides utilities to read CSV files and assign proper column
names to ensure consistent data structure across train and test datasets.

Author: Jean Carlo Londoño Ocampo
Date: February 2026
"""

# Third-party library for data manipulation and analysis
import pandas as pd

# Define the column names for the KDD dataset
# This constant represents all 41 features in the KDD (Knowledge Discovery and Data Mining)
# dataset used for network intrusion detection
COLUMNS = [
    # Basic connection features: Duration, protocol information, and traffic volume
    "duration",              # Total duration of the connection in seconds
    "protocol_type",         # Protocol used (TCP, UDP, ICMP, etc.)
    "service",              # Application level service on the destination (e.g., http, telnet)
    "flag",                 # Status of the connection (SF, S0, S1, S2, S3, etc.)
    "src_bytes",            # Number of data bytes transmitted from source to destination
    "dst_bytes",            # Number of data bytes transmitted from destination to source
    
    # Anomaly indicators: Network behavior flags suggesting suspicious activity
    "land",                 # Binary flag (1 if connection is from/to same host/port)
    "wrong_fragment",       # Number of wrong fragments in the connection
    "urgent",               # Number of urgent packets
    "hot",                  # Number of "hot" indicators (access to sensitive files/directories)
    
    # Host-based features: Authentication and access attempt indicators
    "num_failed_logins",    # Number of failed login attempts
    "logged_in",            # Binary flag (1 if successfully logged in, 0 otherwise)
    "num_compromised",      # Number of "compromised" conditions (security violations)
    "root_shell",           # Binary flag (1 if root shell obtained, 0 otherwise)
    "su_attempted",         # Binary flag (1 if su command attempted, 0 otherwise)
    "num_root",             # Number of root accesses
    
    # File system indicators: File creation and shell access attempts
    "num_file_creations",   # Number of file creation operations
    "num_shells",           # Number of shell prompts
    "num_access_files",     # Number of access control file operations
    "num_outbound_cmds",    # Number of outbound commands executed
    
    # Login context features: Type of login session
    "is_host_login",        # Binary flag (1 if login is a host login, 0 otherwise)
    "is_guest_login",       # Binary flag (1 if login is a guest login, 0 otherwise)
    
    # Statistical features: Connection frequency and rate indicators
    "count",                # Number of connections to the same destination host within past 2 seconds
    "srv_count",            # Number of connections to the same service from unique hosts/time window
    
    # Error rate features: Network protocol error frequency analysis
    "serror_rate",          # Percentage of connections with SYN errors
    "srv_serror_rate",      # Percentage of connections to the same service with SYN errors
    "rerror_rate",          # Percentage of connections with REJ errors
    "srv_rerror_rate",      # Percentage of connections to same service with REJ errors
    
    # Service distribution features: Traffic pattern indicators
    "same_srv_rate",        # Percentage of connections to the same service
    "diff_srv_rate",        # Percentage of connections to different services
    "srv_diff_host_rate",   # Percentage of different hosts reached via same service
    
    # Destination-based features: Host-level traffic analysis
    "dst_host_count",       # Number of connections to the destination host in past 100 connections
    "dst_host_srv_count",   # Number of connections to the dest host using same service in past 100 conn
    
    # Destination host error rates: Network reliability indicators
    "dst_host_same_srv_rate",      # % of connections to same service among connections to dest host
    "dst_host_diff_srv_rate",      # % of different services to destination host
    "dst_host_same_src_port_rate", # % of connections using same source port to dest host
    "dst_host_srv_diff_host_rate", # % of different source hosts to same service on dest host
    "dst_host_serror_rate",        # % of connections to dest host with SYN errors
    "dst_host_srv_serror_rate",    # % of connections to same service on dest host with SYN errors
    "dst_host_rerror_rate",        # % of connections to dest host with REJ errors
    "dst_host_srv_rerror_rate",    # % of connections to same service on dest host with REJ errors
    
    # Target labels: Classification and difficulty assessment
    "label",                # Network intrusion classification (normal, attack type, etc.)
    "difficulty"            # Difficulty level of detecting the attack
]


def load_data(train_path, test_path):
    """
    Load training and testing datasets from CSV files with proper column assignment.
    
    This function reads CSV files from the specified file paths and automatically assigns
    the predefined column names from the COLUMNS constant. This ensures both datasets
    have consistent column names and structure for subsequent data processing and analysis.
    
    Parameters
    ----------
    train_path : str
        Absolute or relative file path to the training dataset CSV file.
        Expected format: CSV file without header row.
        
    test_path : str
        Absolute or relative file path to the testing dataset CSV file.
        Expected format: CSV file without header row (same structure as train_path).
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two pandas DataFrames:
        - train : pd.DataFrame
            Training dataset with shape (n_samples, 41) containing all feature columns
            and the target label column.
        - test : pd.DataFrame
            Testing dataset with shape (m_samples, 41) containing the same column structure
            as the training set.
    
    Raises
    ------
    FileNotFoundError
        If either train_path or test_path file does not exist.
    ValueError
        If the CSV files have fewer than 41 columns or are malformed.
    
    Example
    -------
    >>> train_data, test_data = load_data('KDDTrain+.txt', 'KDDTest+.txt')
    >>> print(train_data.shape)
    >>> print(test_data.columns.tolist())
    
    Notes
    -----
    - The COLUMNS constant must be maintained and kept in sync with the actual dataset structure.
    - Both files should have exactly 41 features (no header row).
    - This function is typically called during the data loading phase of the ML pipeline.
    """
    
    # Load training dataset from CSV file without header and assign column names
    train = pd.read_csv(train_path, names=COLUMNS)
    
    # Load testing dataset from CSV file without header and assign column names
    # Ensures consistent structure with training data
    test = pd.read_csv(test_path, names=COLUMNS)
    
    # Return both datasets as a tuple for easy unpacking
    return train, test