from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

    hf_token: str

    postgres_user: str = "sentiment"
    postgres_password: str = "sentiment"
    postgres_db: str = "sentiment"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_host: str = "localhost"
    redis_port: int = 6379

    max_file_size_mb: int = 50
    min_duration_seconds: float = 1.0
    temp_dir: str = "/tmp/sentiment_audio"

    dsp_whisper_threshold: float = -45.0
    dsp_quiet_threshold: float = -30.0
    dsp_normal_threshold: float = -18.0
    dsp_loud_threshold: float = -8.0

    llm_timeout_seconds: int = 300
    llm_reasoning_budget: int = 4096
    llm_max_retries: int = 2
    llm_temperature: float = 0.1

    chunk_max_duration_s: int = 120

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "forbid",
    }

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}?sslmode=disable"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()
