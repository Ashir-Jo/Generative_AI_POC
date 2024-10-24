import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

pdf_data = read_pdf(r"C:\MyStuff\stories.pdf")

ollama_llm = Ollama(model="phi")

prompt_template = """
Use the following data to answer the question:
Data: {data}
Question: {question}
"""
prompt = PromptTemplate(input_variables=["data", "question"], template=prompt_template)

result = prompt | ollama_llm
response = result.invoke({
    "data": pdf_data[:2000],
    "question": "Summarize SANTA’S GIFTS"
})

print(response)
