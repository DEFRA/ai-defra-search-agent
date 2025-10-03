from pathlib import Path

from app.config import config as settings
from app.lib.s3.s3_service import S3Service

PROMPT_DIR = Path(__file__).parent
PROMPT_S3_BUCKET = settings.prompt_s3_bucket


def load_prompt(name: str) -> str:
    if PROMPT_S3_BUCKET:
        s3 = S3Service(bucket_name=PROMPT_S3_BUCKET)
        key = f"prompts/{name}.txt"
        return s3.get_file(key).decode("utf-8")
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
