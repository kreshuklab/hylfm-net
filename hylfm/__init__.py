import warnings

try:
    from .settings import settings
except ImportError as e:
    warnings.warn(f"{e}\nUsing default settings instead.")
    from .settings_default import Settings

    settings = Settings()
