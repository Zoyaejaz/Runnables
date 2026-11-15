from dotenv import load_dotenv #dotenv Loads environment variables
import os #Accesses environment variables
import requests #Makes HTTP requests

load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage # HumanMessage Represents a message sent by a human user., ToolMessage Represents the result returned by a tool.
from tavily import TavilyClient
from rich import print #This imports the print function from the Rich library.Rich makes terminal output more attractive and readable.Instead of using the normal Python print function, you're using Rich's print function.
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

# =========================
# 🌦️ Weather Tool
# =========================

@tool #This converts your Python function into a LangChain tool.It allows the LLM to discover and call this function.
def get_weather(city: str) -> str:
    """Get current weather of a city""" #A docstring describes what the function does.The LLM uses the tool's description to understand when it should call that tool.
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json() #.json() converts the JSON response into a Python dictionary.
    
    if str(data.get("cod")) != "200": #HTTP/API response code 200 generally indicates a successful request.
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city""" #this written statement is called docstring and its function is to tell the LLM what this function does
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)
# =========================
# 🧠 LLM Setup
# =========================

llm = ChatGroq(model="openai/gpt-oss-120b")

#Middleware is an intermediate layer that allows you to perform additional operations before or around an action.

@wrap_tool_call #This decorator converts your function into tool-call middleware.
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

#create_agent() creates an agent using LangChain's agent framework.Under the hood, it uses LangGraph to manage the agent's execution loop.The agent is responsible for coordinating the LLM and tools.
agent = create_agent(
    llm,
    tools = [get_weather,get_news],
    system_prompt= """
You are a helpful city assistant.
You can answer questions about weather and latest city news.
Use the available tools whenever live information is required.
Do not invent weather information or news.
""",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )