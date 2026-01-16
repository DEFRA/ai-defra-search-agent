import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient

from app.chat import job_models


class AbstractJobRepository(ABC):
    @abstractmethod
    async def create(self, job: job_models.ChatJob) -> job_models.ChatJob:
        pass

    @abstractmethod
    async def get(self, job_id: uuid.UUID) -> job_models.ChatJob | None:
        pass

    @abstractmethod
    async def update(
        self,
        job_id: uuid.UUID,
        status: job_models.JobStatus,
        result: dict | None = None,
        error_message: str | None = None,
    ) -> job_models.ChatJob | None:
        pass


class MongoJobRepository(AbstractJobRepository):
    def __init__(self, client: AsyncIOMotorClient, database_name: str):
        self.client = client
        self.database = client[database_name]
        self.collection = self.database.chat_jobs

    async def create(self, job: job_models.ChatJob) -> job_models.ChatJob:
        await self.collection.insert_one(job.model_dump(mode="json"))
        return job

    async def get(self, job_id: uuid.UUID) -> job_models.ChatJob | None:
        doc = await self.collection.find_one({"job_id": str(job_id)})
        if not doc:
            return None
        return job_models.ChatJob(**doc)

    async def update(
        self,
        job_id: uuid.UUID,
        status: job_models.JobStatus,
        result: dict | None = None,
        error_message: str | None = None,
    ) -> job_models.ChatJob | None:
        update_data = {"status": status, "updated_at": datetime.now(UTC)}
        if result is not None:
            update_data["result"] = result
        if error_message is not None:
            update_data["error_message"] = error_message

        await self.collection.update_one(
            {"job_id": str(job_id)}, {"$set": update_data}
        )
        return await self.get(job_id)
