"""
Configuration module for managing environment variables and defaults.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for managing environment variables and defaults.
    """
    
    # Default values
    DEFAULTS = {
        # API Keys
        "ANTHROPIC_API_KEY": None,  # No default, must be provided
        "OPENAI_API_KEY": None,  # No default, must be provided
        
        # Application Settings
        "DATABASE_PATH": "data/database.sqlite",
        "PAPERS_DIR": "data/pdfs",
        "PROCESSED_DIR": "data/processed",
        "LOG_LEVEL": "INFO",
        
        # arXiv API Settings
        "ARXIV_MAX_RESULTS": "100",
        "ARXIV_WAIT_TIME": "3",
        
        # Processing Settings
        "BATCH_SIZE": "10",
        "MAX_WORKERS": "5",
        
        # Model Settings
        "CLAUDE_MODEL": "claude-3-sonnet-20240229",
        "OPENAI_MODEL": "gpt-4o",
        
        # Web Interface Settings
        "WEB_HOST": "0.0.0.0",
        "WEB_PORT": "5000",
        "DEBUG": "False"
    }
    
    # Required environment variables
    REQUIRED = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    def __init__(self):
        """
        Initialize the configuration.
        """
        # Load all configuration values
        self.config: Dict[str, Any] = {}
        
        # Load from environment
        for key, default in self.DEFAULTS.items():
            self.config[key] = os.getenv(key, default)
        
        # Validate required values
        self._validate()
        
        # Convert types for numeric values
        self._convert_types()
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.debug("Configuration loaded successfully")
    
    def _validate(self) -> None:
        """
        Validate that all required configuration values are present.
        Raises ValueError if any required values are missing.
        """
        missing = []
        
        for key in self.REQUIRED:
            if not self.config.get(key):
                missing.append(key)
        
        if missing:
            raise ValueError(f"Missing required configuration values: {', '.join(missing)}")
    
    def _convert_types(self) -> None:
        """
        Convert string values to appropriate types.
        """
        # Integer conversions
        int_keys = [
            "ARXIV_MAX_RESULTS",
            "ARXIV_WAIT_TIME",
            "BATCH_SIZE",
            "MAX_WORKERS",
            "WEB_PORT"
        ]
        
        for key in int_keys:
            if key in self.config and self.config[key] is not None:
                try:
                    self.config[key] = int(self.config[key])
                except ValueError:
                    logger.warning(f"Could not convert {key}={self.config[key]} to integer, using default")
                    self.config[key] = int(self.DEFAULTS[key])
        
        # Boolean conversions
        bool_keys = [
            "DEBUG"
        ]
        
        for key in bool_keys:
            if key in self.config and self.config[key] is not None:
                self.config[key] = self.config[key].lower() in ("true", "yes", "1", "t", "y")
    
    def _ensure_directories(self) -> None:
        """
        Ensure that all required directories exist.
        """
        directory_keys = [
            "DATABASE_PATH",  # Parent directory
            "PAPERS_DIR",
            "PROCESSED_DIR"
        ]
        
        for key in directory_keys:
            if key in self.config and self.config[key] is not None:
                path = Path(self.config[key])
                
                # For database, get the parent directory
                if key == "DATABASE_PATH":
                    path = path.parent
                
                path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value, or the default if not found
        """
        return self.config.get(key, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get a configuration value as an integer.
        
        Args:
            key: The configuration key
            default: Default value to return if the key is not found or not an integer
            
        Returns:
            The configuration value as an integer, or the default if not found or not an integer
        """
        value = self.get(key, default)
        
        if value is None:
            return default
        
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get a configuration value as a boolean.
        
        Args:
            key: The configuration key
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value as a boolean, or the default if not found
        """
        value = self.get(key, default)
        
        if value is None:
            return default
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "t", "y")
        
        return bool(value)
    
    def get_path(self, key: str, default: Optional[str] = None) -> Optional[Path]:
        """
        Get a configuration value as a Path.
        
        Args:
            key: The configuration key
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value as a Path, or the default if not found
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        return Path(value)
    
    def get_list(self, key: str, default: Optional[List[str]] = None, separator: str = ",") -> Optional[List[str]]:
        """
        Get a configuration value as a list.
        
        Args:
            key: The configuration key
            default: Default value to return if the key is not found
            separator: Separator to split the string by (default: comma)
            
        Returns:
            The configuration value as a list, or the default if not found
        """
        value = self.get(key, default)
        
        if value is None:
            return default
        
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator)]
        
        return default


# Create a singleton instance
config = Config() 