from receiver import find_information
from huggingface_hub import InferenceClient
from google import genai
import os
import json
from receiver import retrieve
from typing import List
GOOGLE_API = "AIzaSyD8I_NEiSOc-VSxJpwpafPTiotiarIVMlE" #https://aistudio.google.com/apikey truy cập để lấy API
#get user question and similar information -> create a prompt for LLM
SYSTEM_PROMPT = "Hãy tưởng tượng bạn là một chuyên gia trong lĩnh vực tài chính \
        hãy trả lời chính xác và đưa ra dẫn chứng cho câu hỏi [user_question] [] \
            .....\
         nếu những thông tin vừa rồi không liên quan đến câu hỏi không cố trả lời hãy trả về không có thông tin cho câu hỏi \
        "
def get_user_prompt_docx(user_question, similar_infos: list[dict]):
    user_prompt = 'Từ các thông tin sau:\n'
    for similar_info in similar_infos:
        user_prompt += f"Tiêu đề: {similar_info['title']}\nNội dung: {similar_info['text']}\n"
    user_prompt += 'Hãy trả lời câu hỏi:\n'
    user_prompt += user_question
    return user_prompt

def get_user_prompt_csv(user_question, similar_infos: list[dict]):
    user_prompt = 'Từ các thông tin sau:\n'
    for similar_info in similar_infos:
        user_prompt += f"Hạng mục: {similar_info['Hạng mục']}\nMã số: {similar_info['Mã số']}\nGiá trị: {similar_info['Giá trị']}\n"
    user_prompt += 'Hãy trả lời câu hỏi:\n'
    user_prompt += user_question
    return user_prompt

def get_user_prompt_md(user_question, similar_infos: list[dict]):
    user_prompt = 'dựa vào những thông tin sau: \n'
    for similar_info in similar_infos:
        user_prompt += f"Trang: {similar_info['Page']}\nNội dung: {similar_info['content']}\n"
    # user_prompt += 'Hãy trả lời câu hỏi:\n'
    
    final_prompt = f"Hãy tưởng tượng bạn là một chuyên gia trong lĩnh vực tài chính \
        hãy trả lời chính xác và đưa ra dẫn chứng cho câu hỏi {user_question} {user_prompt} \
            .....\
         nếu những thông tin vừa rồi không liên quan đến câu hỏi không cố trả lời hãy trả về không có thông tin cho câu hỏi. \
        "
    # user_prompt += user_question
    return final_prompt

def get_user_prompt(input_model='all-MiniLM-L6-v2', num_sim_docx=20, user_question = '',temp_path = None) ->str:
    user_question, similar_infos = find_information(input_model=input_model, k=num_sim_docx,user_question=user_question,temp_path=temp_path)
    #todo: identify if metadatas if of docx or csv or md
    similar_k = retrieve(query=user_question,temp_path=temp_path,top_k=5)
    keys = list(similar_infos[0].keys())
    
    if 'title' in keys:
        user_prompt = get_user_prompt_docx(user_question=user_question, similar_infos=similar_infos)
    elif 'Mã số' in keys:
        user_prompt = get_user_prompt_csv(user_question=user_question, similar_infos=similar_infos)
    else:
        user_prompt = get_user_prompt_md(user_question=user_question, similar_infos=similar_k)

    return user_prompt, similar_k

def evaluate_LLM(question,answer,contexts):
    contexts = [doc["content"] for doc in contexts]
    log_entry = {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }
    log_file = 'data_test.json'
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return answer, contexts

def respond_user(user_question,temp_path):
    user_prompt, contexts = get_user_prompt(input_model='all-MiniLM-L6-v2', num_sim_docx=20, user_question=user_question,temp_path=temp_path)
    client = genai.Client(api_key= GOOGLE_API)
    os.environ["GEMINI_API_KEY"] = GOOGLE_API
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=user_prompt
    )
    evaluate_LLM(user_question, response.text, contexts)
    return response.text

temp_path_short = 'ACB _3reports.pdf'
temp_path_short2 = 'congtythuysan4report.pdf'
temp_path_long = '20250724 - ACB - BCTC hop nhat Quy 02 nam 2025 (2).pdf'
 
print(respond_user("Tài sản cố định có giá trị bao nhiêu", temp_path=temp_path_long))