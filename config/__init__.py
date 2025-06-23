# config/__init__.py
"""
Configuration module for REVERSED TADF System.
"""

from .excitation_settings import (
    EXCITATION_SETTINGS, 
    VISUALIZATION_SETTINGS,
    DATABASE_SETTINGS
)

__all__ = [
    'EXCITATION_SETTINGS',
    'VISUALIZATION_SETTINGS', 
    'DATABASE_SETTINGS'
]