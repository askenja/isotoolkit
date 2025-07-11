
import sys
import logging


def setup_logging(log_file='execution.log', logger_level=logging.INFO, stream_level=logging.INFO, stream=sys.stdout):
    """
    Sets up logging to output both to a file and to the notebook (or console).
    
    Parameters
    ----------
    - log_file: (str)
        Path to the log file.
    - level: (int)
        Logging level (e.g. logging.INFO, logging.DEBUG).
    - stream: Stream 
        To output logs to (default: sys.stdout).
    
    Returns
    -------
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logger_level)
    
    # Clear existing handlers to avoid duplicates in Jupyter
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(formatter)
    
    # Stream handler (to notebook or console)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Suppress noisy libraries (optional)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    
    return logger