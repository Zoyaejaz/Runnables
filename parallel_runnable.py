from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

#Components of the runnable
model=ChatGroq(model="openai/gpt-oss-120b",temperature=0)
parser=StrOutputParser()

#Two different prompts
short_prompt=ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")

detailed_prompt=ChatPromptTemplate.from_template("Explain {topic} in detail")

#Input
topic="Machine Learning"

chain=RunnableParallel({
    "short":RunnableLambda(lambda x :x['short']) |short_prompt | model | parser , 
    "detailed":RunnableLambda(lambda x: x['detailed']) |detailed_prompt | model | parser }) 
#RunnableParallel: Executes multiple chains simultaneously and returns their outputs in a dictionary.
#RunnableLambda: Converts a normal Python function (lambda or custom function) into a LangChain Runnable so it can be used in a chain.


result=chain.invoke({
    "short":{"topic":"Machine Learning"},
    "detailed":{"topic":"Deep Learning"}
})

print(result['short'])
print(result['detailed'])