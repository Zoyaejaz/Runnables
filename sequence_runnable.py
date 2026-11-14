from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #this is used to organize the output of the model into a string format

# 1. Prompt Template
prompt=ChatPromptTemplate.from_template("Explain {topic} in simple words")

# 2. Model
model=ChatGroq(model="openai/gpt-oss-120b",temperature=0)

# 3. Output Parser
parser=StrOutputParser()

#Step-by-step manual flow

#Format the prompt with the topic
'''formatted_prompt=prompt.format(topic="Machine Learning")

#Call the model manually
response = model.invoke(formatted_prompt)

#Parse the output
parsed_response = output_parser.parse(response) 

print(parsed_response)'''

#instead of performing the above steps manually, we can use the run method to do all the steps in one go that is called runnables

chain=prompt | model | parser

result=chain.invoke("Machine Learning")

print(result)
