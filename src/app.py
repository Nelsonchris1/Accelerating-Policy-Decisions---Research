from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.index import ChatModel
from retriever import chatPrompt
import asyncio
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
async def displayChatBot(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post('/chat')
async def chat(userQuery:ChatModel):
    input=userQuery.query
    return await asyncio.to_thread(chatPrompt,input)