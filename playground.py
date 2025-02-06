from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from fin_agent import web_agent, fin_agent
import os
import phi 

from dotenv import load_dotenv
load_dotenv()
Groq.api_key = os.getenv("GROQ_API_KEY")


### Agent for interacting with webpages, collecting data
web_agent = Agent(
    name = "Web page Agent",
    role = "responsible for searching web for information and collecting valuable data from web",
    model = Groq(id = "deepseek-r1-distill-llama-70b"),
    tools= [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tools_calls = True,
    markdown= True,
)

### web searching agent

fin_agent = Agent(
    name= "Financial AI agent",
    role= "responsible for interact with agents and recommend stocks to buy based on there performance.",
    model = Groq(id="gemma2-9b-it"),
    tools = [
        YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        key_financial_ratios= True,
        company_news=True,
        technical_indicators=True,
        historical_prices=True,
        stock_fundamentals=True)
        ],
    instructions= ["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents = [fin_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground: app", reload = True)