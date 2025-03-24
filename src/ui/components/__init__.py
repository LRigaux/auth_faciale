"""
UI components for facial authentication.

This package contains reusable UI components for the
facial authentication application.
"""

from .home import create_home_component
from .brute_force import create_brute_force_component
from .eigenfaces import create_eigenfaces_component
from .comparison import create_comparison_component
from .common import (
    create_header, create_footer, create_card, create_progress_bar, 
    create_image_display, create_confusion_matrix_display
)

__all__ = [
    'create_home_component',
    'create_brute_force_component',
    'create_eigenfaces_component',
    'create_comparison_component',
    'create_header',
    'create_footer',
    'create_card',
    'create_progress_bar',
    'create_image_display',
    'create_confusion_matrix_display'
] 