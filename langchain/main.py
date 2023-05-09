# https://python.langchain.com/en/latest/getting_started/getting_started.html
# https://python.langchain.com/en/latest/use_cases/question_answering.html
# https://python.langchain.com/en/latest/modules/indexes/getting_started.html
# https://python.langchain.com/en/latest/use_cases/question_answering/semantic-search-over-chat.html

from typing import Any, List

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
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


df = pd.read_pickle('../data/course_info.pkl')

loader = DataFrameLoader(df, ["title_en", "overview_objectives",
                              "overview_learning_outcomes", "overview_description.en"])
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0.9)

documents = loader.load()
texts = text_splitter.split_documents(documents)
db = Chroma.from_documents(texts, embeddings, persist_directory="../data/chroma")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What is the purpose of the Private International Law course"
print(qa.run(query))
