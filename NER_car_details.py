import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def read_pdf(file_path):
    # Function to extract text from the provided PDF file
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Path to the uploaded PDF file
pdf_file_path = r'C:\MyStuff\comprehensive_car_brochure.pdf'

# Extract data from the PDF
pdf_data = read_pdf(pdf_file_path)

# Initialize the Ollama LLM
ollama_llm = Ollama(model="phi")

# Define a few-shot prompt with multiple examples
prompt_template = """
Here are some examples of how to extract car models based on specifications:

Example 1:
Data: {example_data_1}
Question: Show me all the models which have hybrid engines.
Answer: {example_answer_1}

Example 2:
Data: {example_data_2}
Question: Which models come with FWD drive?
Answer: {example_answer_2}

Example 3:
Data: {example_data_3}
Question: Which models have horsepower greater than 300?
Answer: {example_answer_3}

Now, use the following data to answer the question:
Data: {data}
Question: {question}
"""

# Provide example data for few-shot learning
example_data_1 = """
The Ford Mustang exemplifies bold performance and modern innovation with a V8 3.0L engine that delivers 300 HP and
400 Nm of torque. Its AWD system ensures stability on all terrains, while the hybrid fuel system supports a top speed of
200 km/h and impressive acceleration, going from 0 to 100 km/h in just 4.0 seconds. Achieving a fuel efficiency of 10
km/l, the Mustang comes equipped with an automatic transmission for smooth handling. Available in a selection of
colors such as Red, Blue, Black, White, Silver, and Gray, the Mustang features a luxurious leather interior, Bose
Premium Sound System, and an advanced infotainment system supporting both Apple CarPlay and Android Auto.
Additional features include a sunroof, adaptive cruise control, heated seats, and blind spot monitoring, making this used
model with a mileage of 5000 km a thrilling yet practical choice at $30k.
"""
example_answer_1 = "Ford Mustang"

example_data_2 = """
Combining power and elegance, the Dodge Charger is driven by a V8 4.0L engine, producing 310 HP and 406 Nm of
torque. Its FWD drive type and automatic transmission make it a breeze to handle, achieving a top speed of 204 km/h
and accelerating from 0 to 100 km/h in just 3.4 seconds. This gasoline-powered Charger offers a fuel efficiency of 12
km/l. Available in a range of colors, including Red, Blue, Black, White, Silver, and Gray, it boasts a sophisticated leather
interior with a Harman Kardon sound system. Equipped with an infotainment system supporting Apple CarPlay and
Android Auto, this brand-new Charger, priced at $32k, features heated seats, adaptive cruise control, a sunroof, and
blind spot monitoring
"""
example_answer_2 = "Dodge Charger"

example_data_3 = """
The Chevrolet Camaro stands out with its aggressive style and powerful V6 3.5L engine, delivering 305 HP and 403 Nm
of torque. Its FWD drive type, coupled with a manual transmission, offers an engaging driving experience. This
gasoline-powered Camaro boasts a top speed of 202 km/h and accelerates from 0-100 km/h in just 3.7 seconds, with a
fuel efficiency of 11 km/l. Available in colors like Red, Blue, Black, White, Silver, and Gray, the Camaro features an
Alcantara interior, a premium Harman Kardon sound system, and a touchscreen infotainment system with Apple
CarPlay and Android Auto. Priced at $31k, this brand-new model includes advanced features like adaptive cruise
control, a sunroof, and heated seats, promising both luxury and performance.
"""
example_answer_3 = "Chevrolet Camaro"

# Create the prompt template with input examples
prompt = PromptTemplate(
    input_variables=["example_data_1", "example_answer_1", "example_data_2", "example_answer_2", "example_data_3", "example_answer_3", "data", "question"],
    template=prompt_template
)

# Format the prompt with all the extracted PDF content
formatted_prompt = prompt.format(
    example_data_1=example_data_1,
    example_answer_1=example_answer_1,
    example_data_2=example_data_2,
    example_answer_2=example_answer_2,
    example_data_3=example_data_3,
    example_answer_3=example_answer_3,
    data=pdf_data,
    question="Show me all the models which have a top speed more than 250"
)

# Call the LLM with the formatted prompt
response = ollama_llm.invoke(formatted_prompt)

# Print the response from the model
print(response)
