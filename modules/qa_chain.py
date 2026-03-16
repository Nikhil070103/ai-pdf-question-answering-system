from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


def create_qa_chain(retriever):

    pipe = pipeline(
        "text2text-generation",
       model="google/flan-t5-large",
        max_length=512,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain