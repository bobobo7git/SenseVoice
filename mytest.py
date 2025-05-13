from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from exceptions import *
from client_request.analyzeRequest import AnalyzeRequest
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess, format_str_v2, sentence_postprocess_sentencepiece, sentence_postprocess
from pprint import pprint
import tempfile
import os

app = FastAPI()

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

label2scheme = {
    'HAPPY': 'happiness',
    'SAD': 'sadness',
    'NEUTRAL': 'neutral',
    'ANGRY': 'angry',
    'EMO_UNKNOWN': 'other'
}
prob2scheme = {
    'happy': 'happiness',
    'sad': 'sadness',
    'neutral': 'neutral',
    'angry': 'angry',
    'unk': 'other'
}

@app.get("/")
def read_root():
    return {"message": "Hello audio-AI server!"}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    url = req.audio_url
    filename = url.split("/")[-1]
    
    try:
        res = m.inference(
            data_in=url,
            language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            ban_emo_unk=False,
            **kwargs,
        )
    except HTTPException as e:
        raise e
    except ValueError:
        raise InvalidAudioFormatError(filename.split('.')[-1])
    except Exception as e:
        raise InternalSenseVociceError("inference failed")

    emotion_scores = {
        prob2scheme[label]: v
        for label, v in res[0][0]['emotion_probs'].items()
    }

    # final emotion tag
    res = res[0][0]["text"].split("<|woitn|>")
    li = res[0].replace("<", "").replace(">", "").split("|")
    li = [e for e in li if e]

    final_language, final_emotion, final_event = li
    
    return JSONResponse(content={
        "message": "success",
        "filename": filename,
        "result": {
            "emotion": label2scheme[final_emotion],
            "emotion_scores": emotion_scores,
            "language": final_language,
            "event": final_event
        },
        
    }, status_code=200)
    

@app.post("/legacy-analyze")
async def legacy_analyze(file: UploadFile=File(...)):
    filename = file.filename
    content_type = file.content_type

    #---------- 음성 파일 임시 저장 후 피처 추출
    ext = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name
        if not content_type.startswith('audio'):
            raise InvalidAudioFormatError(content_type)
    try:
        res = m.inference(
            data_in=tmp_path,
            language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            ban_emo_unk=False,
            **kwargs,
        )
    except Exception:
        raise InternalSenseVociceError("inference failed")
    finally:
        os.remove(tmp_path)

    # emotion score 
    emotion_scores = {
        prob2scheme[label]: v
        for label, v in res[0][0]['emotion_probs'].items()
    }

    # final emotion tag
    res = res[0][0]["text"].split("<|woitn|>")
    li = res[0].replace("<", "").replace(">", "").split("|")
    li = [e for e in li if e]

    final_language, final_emotion, final_event = li
    
    return JSONResponse(content={
        "filename": filename,
        "content_type": content_type,
        "message": "success",
        "result": {
            "emotion": label2scheme[final_emotion],
            "emotion_scores": emotion_scores,
            "language": final_language,
            "event": final_event
        },
        
    }, status_code=200)

@app.exception_handler(InvalidAudioFormatError)
async def invalid_audio_format_handler(request: Request, exc: InvalidAudioFormatError):
    return JSONResponse(content={
            "message": "input file format error",
            "detail": exc.detail
        }, status_code=400)

@app.exception_handler(InternalSenseVociceError)
async def invalid_audio_format_handler(request: Request, exc: InternalSenseVociceError):
    return JSONResponse(content={
            "message": "audio model error",
            "detail": exc.detail
        }, status_code=500)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status": exc.status_code,
        },
    )

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()

    PORT = int(os.getenv('PORT'))
    HOST = os.getenv('HOST')

    uvicorn.run(app, host=HOST, port=PORT)
