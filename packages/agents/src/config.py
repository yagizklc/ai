from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='__', case_sensitive=False, env_file='.env'
    )

    logfire_token: str


config = Config()  # type: ignore
