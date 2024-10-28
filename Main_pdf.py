import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

#Function to read from the PDF
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

#Path to the pdf file
pdf_data = read_pdf(r"C:\MyStuff\stories.pdf")

#Initializing ollama LLM
ollama_llm = Ollama(model="phi")

# Zero shot prompt template
prompt_template = """
Use the following data to answer the question:
Data: {data}
Question: {question}
"""

#prompt template with the input variables
prompt = PromptTemplate(input_variables=["data", "question"], template=prompt_template)

#Call Ollama LLM
result = prompt | ollama_llm
response = result.invoke({
    "data": pdf_data[:2000],
    "question": "Summarize SANTA’S GIFTS"
})

#print the output
print(response)
