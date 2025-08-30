import os
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
import json
from ragas.metrics import faithfulness, answer_relevancy
import asyncio
# set OpenAI key
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBpAFCUa0UGLvZh2lTCgz_T0BDsNkN96YU"

from ragas.embeddings.base import Embeddings
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceEndpoint

# tạo llm Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = LangchainLLMWrapper(llm)
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]):
        return [self.model.encode(t).tolist() for t in texts]
emb = HuggingFaceEmbeddings()
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = LangchainLLMWrapper(llm)
with open("data_test copy.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
samples = []
for item in data:
    samples.append({
        "question": item["question"],
        "answer": item["answer"],
        "contexts": item["contexts"]
    })
# Chuyển sang Dataset của Hugging Face
# dataset = Dataset.from_list(data)
print(len(samples))
# Chọn metrics bạn muốn tính
metrics = [faithfulness, answer_relevancy]
results = []
# Chạy đánh giá
# try:
import grpc
async def evalution():
    try:
        for sample in samples:
            dataset = Dataset.from_dict({
                "question": [sample["question"]],
                "answer": [sample["answer"]],
                "contexts": [sample["contexts"]],
            })
            print(dataset)
            score = evaluate(dataset, llm=llm, embeddings=emb, metrics=metrics)
            results.append(score)
    except Exception as e:
        print(e)
    finally:
            try:
                await grpc.aio.shutdown()   # nếu có dùng grpc async
            except Exception as e:
                print(f"Bỏ qua lỗi shutdown grpc: {e}")

try:
    asyncio.run(evalution())
    print(results)
except Exception as e:
    print(f"Lỗi tổng: {e}")
