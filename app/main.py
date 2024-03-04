from fastapi import FastAPI
from pydantic import BaseModel
from app.model import ParaphraseModel

paraphrasing_pipeline = ParaphraseModel()

app = FastAPI()

class ParaphraseInput(BaseModel):
    text: str

class ModelResponse(BaseModel):
    result: str

@app.get("/")
def home():
    return "Head over to http://localhost:80/docs to paraphrase your text!"

@app.post("/paraphrase")
def paraphrase(input_data: ParaphraseInput):
    paraphrased_text = paraphrasing_pipeline.paraphrase(input_data.text)
    return ModelResponse( result = paraphrased_text )
