import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- NVIDIA API ---
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

    # --- Hugging Face ---
    hf_token: str

    # --- PostgreSQL ---
    postgres_user: str = "sentiment"
    postgres_password: str = "sentiment"
    postgres_db: str = "sentiment"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # --- Audio ---
    max_file_size_mb: int = 50
    min_duration_seconds: float = 1.0
    temp_dir: str = "/tmp/sentiment_audio"

    # --- DSP thresholds (dBFS) ---
    dsp_whisper_threshold: float = -40.0
    dsp_quiet_threshold: float = -30.0
    dsp_normal_threshold: float = -20.0
    dsp_loud_threshold: float = -10.0

    # --- LLM ---
    llm_timeout_seconds: int = 300
    llm_reasoning_budget: int = 4096
    llm_max_retries: int = 2
    llm_temperature: float = 0.1

    # --- Chunking ---
    chunk_min_duration_s: int = 60
    chunk_max_duration_s: int = 120

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()