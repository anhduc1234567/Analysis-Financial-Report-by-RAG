from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from core.output_generator import respond_user
import tempfile
import shutil
import os
app = FastAPI()
saved_file_path = None
# Cho phép frontend truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc thay "*" bằng "http://localhost:3000" nếu cần giới hạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: str = Form(...), file: Optional[UploadFile] = File(None)):
    global saved_file_path
    print(f"Message: {message}")
    # if message:
    if file:
        filename = file.filename.lower()
        print(f"Received file: {file.filename}")
        if filename.endswith('.doc') or filename.endswith('.docx'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                saved_file_path = tmp.name
        elif filename.endswith('.pdf'):
            # Xử lý file PDF riêng biệt, ví dụ lưu tạm và gọi hàm OCR riêng
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                saved_file_path = tmp.name
                print("PDF file nhận được, xử lý riêng tại đây")
        elif filename.endswith('.jpg'):
            # Xử lý file PDF riêng biệt, ví dụ lưu tạm và gọi hàm OCR riêng
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                shutil.copyfileobj(file.file, tmp)
                saved_file_path = tmp.name
                print("file ảnh nhận được, xử lý riêng tại đây")
        elif filename.endswith('.png'):
            # Xử lý file PDF riêng biệt, ví dụ lưu tạm và gọi hàm OCR riêng
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(file.file, tmp)
                saved_file_path = tmp.name
                print("file ảnh nhận được, xử lý riêng tại đây")
        elif filename.endswith('.csv'):
            # Xử lý file PDF riêng biệt, ví dụ lưu tạm và gọi hàm OCR riêng
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                shutil.copyfileobj(file.file, tmp)
                saved_file_path = tmp.name
                print("CSV file nhận được, xử lý riêng tại đây")
    if not saved_file_path:
        return {"response": "Bạn chưa upload file tài liệu nào."}
    
    if message:
        res = respond_user(user_question=message, temp_path=saved_file_path)
        return {"response": res}
    else:
        return {"response": "❗Bạn chưa nhập câu hỏi."}

    # return {"response": f" {message}"}
