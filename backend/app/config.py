import re
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    environment: str = "development"
    public_url: str = "http://localhost:3000"
    extra_allowed_hosts: list[str] = Field(default_factory=list)
    render_external_hostname: str = ""
    database_url: str = "sqlite:///./docuchat.db"

    openai_model: str = "gpt-4.1-mini"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "docuchat_chunks"

    @field_validator("extra_allowed_hosts")
    @classmethod
    def validate_extra_hosts(cls, hosts: list[str]) -> list[str]:
        normalized = []
        for host in hosts:
            host = host.strip().lower()
            if len(host) > 253 or not all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")
            ):
                raise ValueError("EXTRA_ALLOWED_HOSTS must contain exact hostnames without URLs, ports or wildcards")
            normalized.append(host)
        return list(dict.fromkeys(normalized))

    @field_validator("render_external_hostname")
    @classmethod
    def validate_render_host(cls, host: str) -> str:
        return cls.validate_extra_hosts([host])[0] if host else ""

    @property
    def trusted_hosts(self) -> list[str]:
        hosts = [urlparse(self.public_url).hostname, "127.0.0.1", self.render_external_hostname, *self.extra_allowed_hosts]
        if self.environment != "production":
            hosts.extend(["localhost", "testserver"])
        return list(dict.fromkeys(host for host in hosts if host))

    @property
    def secure(self) -> bool:
        return self.public_url.startswith("https://")

    @model_validator(mode="after")
    def production_settings(self):
        self.public_url = self.public_url.rstrip("/")
        parsed = urlparse(self.public_url)
        if not parsed.hostname or parsed.path or parsed.scheme not in {"http", "https"}:
            raise ValueError("PUBLIC_URL must be an HTTP(S) origin with no path")
        if self.environment == "production":
            problems = []
            if not self.secure:
                problems.append("PUBLIC_URL must use HTTPS and point to the frontend")
            if not self.database_url.startswith("postgresql"):
                problems.append("DATABASE_URL must use PostgreSQL")
            if not self.qdrant_url:
                problems.append("QDRANT_URL is required")
            if problems:
                raise ValueError("Invalid production settings: " + "; ".join(problems))
        return self


@lru_cache
def settings() -> Settings:
    return Settings()
