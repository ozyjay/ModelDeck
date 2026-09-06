from __future__ import annotations

import uvicorn

from modeldeck.config import Settings
from modeldeck.main import create_app


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.management_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
