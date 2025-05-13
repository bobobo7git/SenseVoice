from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    audio_url: str