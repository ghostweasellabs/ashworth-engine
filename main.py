"""Ashworth Engine - Multi-agent Financial Intelligence Platform."""

from src.config.settings import settings


def main() -> None:
    """Main entry point for the Ashworth Engine."""
    print(f"Ashworth Engine v0.1.0")
    print(f"Environment: {settings.environment}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Debug Mode: {settings.debug}")
    print("Project structure initialized successfully!")


if __name__ == "__main__":
    main()
