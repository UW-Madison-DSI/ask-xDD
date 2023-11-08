import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("RETRIEVER_APIKEY")
API_KEY_NAME = "Api-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME)


async def has_valid_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header
