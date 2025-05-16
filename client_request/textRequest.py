from pydantic import BaseModel

class TextAnalyzeRequest(BaseModel):
    text: str