import os
import json
from Contract_generation import rag_pipeline_with_prompt, get_text_from_Pdf, Generation_of_Contract
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pydantic import BaseModel
app = FastAPI()

origins = [
    "http://localhost:3000",        # If frontend is running locally
    "http://127.0.0.1:3000",
    "http://your-frontend-domain.com",  # Replace with your actual frontend domain
    "*"  # Allow all origins (not recommended in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Or ["*"] for public access
    allow_credentials=True,
    allow_methods=["*"],     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],     # Allow all headers
)

class Question(BaseModel):
    filename: str

class User_input(BaseModel):
    filename: str
    Placeholders: str


@app.post("/Contract_template/")
async def Document_proc(file: UploadFile = File(...)):
    file_location = f"./original_contracts/{file.filename}"
    # Save the file directly without reading it
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    context=get_text_from_Pdf(file_location)
    output=rag_pipeline_with_prompt(context)
    output_path=f"/root/Contract_generation/Contract_templates/{file.filename}.json"
    with open(output_path, 'w', encoding='utf-8') as file:
       file.write(output)    
    return {"summary": "The contract is sucessfully processed"}

@app.post("/template_generation/")
async def placeholder(data: Question):
    path=f"/root/Contract_generation/Contract_templates/{data.filename}.json"
    with open(path, 'r', encoding='utf-8') as file:
    	data = json.load(file)
    Placeholders=data['Placeholders']
    return {"Placeholders":Placeholders}

@app.post("/Contract_generation/")
async def Contract_generation(data: User_input):
        path=f"/root/Contract_generation/Contract_templates/{data.filename}.json"
        with open(path, 'r', encoding='utf-8') as file:
             datas = json.load(file)
        Context=datas['Template']
        Final_output=Generation_of_Contract(Context,data.Placeholders)
        return {"Generated_Contract":Final_output}
