import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='__', case_sensitive=False, env_file='.env'
    )

    device: str = 'mps' if torch.mps.is_available() else 'cpu'
    hello: str


config = Config()
