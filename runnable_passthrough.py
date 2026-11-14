from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser #this is used to organize the output of the model into a string format
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


#Model
model=ChatGroq(model="openai/gpt-oss-120b",temperature=0)

#parser
parser=StrOutputParser()

#prompt
code_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])

seq = code_prompt | model | parser

seq2 = RunnableParallel(
    {"code" : RunnablePassthrough(), #Passes the input data unchanged to the next step in a LangChain chain without modifying it.
     "explanation" : explain_prompt | model | parser}
)

chain= seq | seq2

result = chain.invoke({"topic" : "please write a code of palindrome in python "})

print(result['code'])
print(result['explanation'])