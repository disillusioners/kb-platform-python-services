from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports():
    from workers.activities import (
        delete_from_s3,
        download_from_s3,
        parse_document,
        generate_embeddings,
        upsert_to_qdrant,
        update_status
    )
    from shared.models.schemas import DocumentStatus


@workflow.defn
class UploadWorkflow:
    def __init__(self):
        self._upload_complete = False

    @workflow.signal(name="upload-complete")
    async def signal_upload_complete(self):
        self._upload_complete = True

    @workflow.run
    async def run(self, document_id: str, s3_key: str):
        """Wait for client to complete upload or timeout."""
        # Wait for "upload-complete" signal with timeout (15 minutes)
        try:
            await workflow.wait_condition(
                lambda: self._upload_complete,
                timeout=timedelta(minutes=15)
            )

            # Signal received - start indexing
            await workflow.execute_child_workflow(
                IndexingWorkflow.run,
                args=[document_id],
                task_queue="indexing-queue",
            )
            return "Upload completed, indexing started"

        except TimeoutError:
            # Upload timed out - cleanup S3 file and mark as failed
            workflow.logger.error(f"Upload timed out for document {document_id}, cleaning up S3")

            await workflow.execute_activity(
                delete_from_s3,
                args=[s3_key],
                start_to_close_timeout=timedelta(minutes=2),
            )

            await workflow.execute_activity(
                update_status,
                args=[document_id, DocumentStatus.FAILED.value, "Upload timed out"],
                start_to_close_timeout=timedelta(seconds=10),
            )

            raise TimeoutError(f"Upload timed out for document {document_id}")

        except Exception as e:
            # Workflow failed for other reasons - cleanup S3 file
            workflow.logger.error(f"Upload workflow failed for document {document_id}: {e}, cleaning up S3")

            await workflow.execute_activity(
                delete_from_s3,
                args=[s3_key],
                start_to_close_timeout=timedelta(minutes=2),
            )

            await workflow.execute_activity(
                update_status,
                args=[document_id, DocumentStatus.FAILED.value, str(e)],
                start_to_close_timeout=timedelta(seconds=10),
            )

            raise e


@workflow.defn
class IndexingWorkflow:
    @workflow.run
    async def run(self, document_id: str):
        # 1. Update status to INDEXING
        await workflow.execute_activity(
            update_status,
            args=[document_id, DocumentStatus.INDEXING.value],
            start_to_close_timeout=timedelta(seconds=10),
        )

        try:
            # 2. Download from S3
            file_data, filename = await workflow.execute_activity(
                download_from_s3,
                args=[document_id],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            
            # 3. Parse
            chunks = await workflow.execute_activity(
                parse_document,
                args=[document_id, file_data, filename],
                start_to_close_timeout=timedelta(minutes=5),
            )
            
            # 4. Generate Embeddings
            embeddings = await workflow.execute_activity(
                generate_embeddings,
                args=[chunks],
                start_to_close_timeout=timedelta(minutes=10),
            )
            
            # 5. Upsert
            await workflow.execute_activity(
                upsert_to_qdrant,
                args=[document_id, embeddings],
                start_to_close_timeout=timedelta(minutes=5),
            )
            
            # 6. Complete
            await workflow.execute_activity(
                update_status,
                args=[document_id, DocumentStatus.COMPLETE.value, None, len(embeddings)],
                start_to_close_timeout=timedelta(seconds=10),
            )
            
            return f"Indexed {len(embeddings)} chunks"

        except Exception as e:
            workflow.logger.error(f"Workflow failed: {e}")
            await workflow.execute_activity(
                update_status,
                args=[document_id, DocumentStatus.FAILED.value, str(e)],
                start_to_close_timeout=timedelta(seconds=10),
            )
            raise e
