# https://python.langchain.com/en/latest/getting_started/getting_started.html
# https://python.langchain.com/en/latest/use_cases/question_answering.html
# https://python.langchain.com/en/latest/modules/indexes/getting_started.html
# https://python.langchain.com/en/latest/use_cases/question_answering/semantic-search-over-chat.html

from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI
from langchain.schema import Generation
from langchain.schema import PromptValue, LLMResult
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()


# langchain.document_loaders.DataFrameLoader has a quite a limited functionality
class DataFrameLoader(BaseLoader):
    def __init__(self, data_frame: Any, page_content_columns: List[str]):
        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        self.data_frame = data_frame
        self.page_content_columns = page_content_columns

    def load(self) -> List[Document]:
        result = []
        for i, row in self.data_frame.iterrows():
            text = ""
            metadata = {}
            for col in self.page_content_columns:
                data = row[col]
                if isinstance(data, list):
                    text += "".join(data) + "\n"
                elif isinstance(data, str):
                    text += data + "\n"
                else:
                    print(f"[IGNORED] [{i}] [{col}] {data}")

            metadata_temp = row.to_dict()
            for col in self.page_content_columns:
                metadata_temp.pop(col)
            # Metadata is a dict where a value can only be str, int, or float. Delete other types.
            for key, value in metadata_temp.items():
                if isinstance(value, (str, int, float)):
                    metadata[key] = value

            result.append(Document(page_content=text, metadata=metadata))
        return result


class MyLanguageModel(BaseLanguageModel):
    def generate_prompt(self, prompts: List[PromptValue], stop: Optional[List[str]] = None,
                        callbacks: Callbacks = None) -> LLMResult:
        generation = Generation(text="Hello World!")
        result = LLMResult(generations=[[generation]])
        return result

    async def agenerate_prompt(self, prompts: List[PromptValue], stop: Optional[List[str]] = None,
                               callbacks: Callbacks = None) -> LLMResult:
        pass  # "whatever dude"


# NOTE: the OpenAIEmbeddings embeddings have the dimensionality of 1536
class MyEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0] * 1536, [2.0] * 1536]

    def embed_query(self, text: str) -> List[float]:
        return [1.0] * 1536


df = pd.read_pickle('../data/course_info.pkl')

loader = DataFrameLoader(df, ["title_en", "overview_objectives",
                              "overview_learning_outcomes", "overview_description.en"])
# NOTE: if the chunk size is too small, then there will be no way to tell which course this information corresponds to
# Ideally, every chunk should contain information about a single course OR every chunk should have some essential
# information about a course
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# embeddings = OpenAIEmbeddings()
embeddings = MyEmbeddings()
# llm = OpenAI(temperature=0.9)
llm = MyLanguageModel()

documents = loader.load()
texts = text_splitter.split_documents(documents)
# db = Chroma.from_documents(texts, embeddings, persist_directory="../data/chroma")
db = Chroma(persist_directory="../data/chroma", embedding_function=embeddings)  # load from disk
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What is the purpose of the Private International Law course"
print(qa.run(query))
