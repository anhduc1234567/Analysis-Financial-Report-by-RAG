from docx_database_creater import create_database
from csv_database_creater import create_database_for_csv
from pdf_database_creater import create_database_for_pdf
from sentence_transformers import SentenceTransformer
from LLama_parse import pdf_to_md
import os
from pdf_database_creater import keyword_search
from rank_bm25 import BM25Okapi
#read input file
def get_input_path() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        input_path = os.path.abspath(os.path.join(base_dir, '..', '000000014601738_VI_BaoCaoTaiChinh_KiemToan_2024_HopNhat_14032025110908.docx'))
        return input_path
    
    except Exception as e:
        print('cant locate docx file')
        print(f'error: {e}')

#get vector database and metadatas
def get_database(input_model='all-MiniLM-L6-v2',temp_path=None) -> tuple:
    if temp_path is None:
        input_path = get_input_path()
    input_path = temp_path
    
    #checking file type
    file_ext = os.path.splitext(input_path)[-1].lower()

    #get vector embedding for csv
    if file_ext == '.csv':
        try:
            index, metadatas = create_database_for_csv(input_path=input_path, embedding_model=input_model)
            return index, metadatas

        except Exception as e:
            print('cant get database for csv file')
            print(f'error: {e}')
            
    elif file_ext == '.pdf' or file_ext == ".png" or file_ext == ".jpg":
        try:
            pdf_to_md(input_path,file_ext)
            index, metadatas = create_database_for_pdf(embedding_model=input_model)
            return index, metadatas
        
        except Exception as e:
            print('cant get database for markdown file')
            print(f'error: {e}')

    #get vector embedding for docx
    else:
        try:
            index, metadatas = create_database(input_path=input_path, input_model=input_model)
            return index, metadatas

        except Exception as e:
            print('cant get database for docx file')
            print(f'error: {e}')
    print("None")

#get user prompt
def get_user_question() -> str:
    user_question = input('What do you want to know: ')
    return user_question

#embedd user question into a vector
def get_user_question_embedding(input_model='all-MiniLM-L6-v2', user_question = '') -> tuple:
    user_question = user_question
    model = SentenceTransformer(input_model)
    user_question_embeddings = model.encode([user_question])
    return user_question, user_question_embeddings[0]


#find k similar information in vector database
def find_information(input_model='all-MiniLM-L6-v2', k=20, user_question = '',temp_path = None) -> tuple:
    user_question, user_question_vector = get_user_question_embedding(input_model=input_model, user_question = user_question)
    index, metadatas = get_database(input_model='all-MiniLM-L6-v2',temp_path = temp_path)

    #search for similar information in FAISS database
    D, I = index.search(user_question_vector.reshape(1, -1), k)  # D: distances, I: indices
    
    similar_info = []
    for idx in I[0]: 
        similar_info.append(metadatas[idx])
    return user_question, similar_info

def hybrid_search(query,temp_path, top_k):
    user_q, semantic_results = find_information(user_question=query,temp_path = temp_path)
    keyword_results = keyword_search(query, top_k)
    
    # Ghép kết quả, loại trùng (theo content)
    seen = set()
    candidates = []
    for item in semantic_results + keyword_results:
        if item["content"] not in seen:
            candidates.append(item)
            seen.add(item["content"])
    return candidates

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=5):
    pairs = [[query, doc["content"]] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]

def retrieve(query, temp_path, top_k=5):
    # 1. Hybrid search
    candidates = hybrid_search(query,temp_path, top_k=5)
    
    # 2. Re-rank
    final_docs = rerank(query, candidates, top_k)
    return final_docs
