"""Temporal worker entry point."""

import asyncio
from temporalio import Worker
from temporalio.client import Client

from .config import get_settings


async def main():
    """Main worker entry point."""
    settings = get_settings()

    client = await Client.connect(
        f"{settings.temporal_host}:{settings.temporal_port}",
        namespace="default",
    )

    worker = Worker(
        client=client,
        task_queue="upload-queue",
        workflows=[],
        activities=[],
    )

    print("Worker started, listening for tasks...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
