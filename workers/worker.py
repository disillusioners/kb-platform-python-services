"""Temporal worker entry point."""

import asyncio
from temporalio import Worker
from temporalio.client import Client

from shared.config import get_settings
from workers.activities import (
    delete_from_s3,
    download_from_s3,
    parse_document,
    generate_embeddings,
    upsert_to_qdrant,
    update_status
)
from workers.workflows import IndexingWorkflow, UploadWorkflow


async def main():
    """Main worker entry point."""
    settings = get_settings()

    client = await Client.connect(
        f"{settings.temporal_host}:{settings.temporal_port}",
        namespace="default",
    )

    worker = Worker(
        client=client,
        task_queue="indexing-queue",
        workflows=[UploadWorkflow, IndexingWorkflow],
        activities=[
            delete_from_s3,
            download_from_s3,
            parse_document,
            generate_embeddings,
            upsert_to_qdrant,
            update_status
        ],
    )

    print("Worker started, listening for tasks...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
