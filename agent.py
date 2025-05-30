import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

load_dotenv()

@tool
def add(x: int, y: int) -> int:
    """Adds two numbers.
    :arg x: The first number.
    :arg y: The second number.
    """
    return x + y

@tool
def subtract(x: int, y: int) -> int:
    """Subtracts two numbers.
    :arg x: The first number.
    :arg y: The second number.
    """
    return x - y
@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers.
    :arg x: The first number.
    :arg y: The second number.
    """
    return x * y

@tool
def divide(x: int, y: int) -> float:
    """Divides two numbers.
    :arg x: The first number.
    :arg y: The second number.
    :raises ValueError: If y is zero.
    """
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

@tool
def modulus(x: int, y: int) -> int:
    """Calculates the modulus of two numbers.
    :arg x: The first number.
    :arg y: The second number.
    :raises ValueError: If y is zero.
    """
    return x % y
@tool
def wiki_search(query: str) -> str:
    """Searches Wikipedia for the given query and returns the top results.
    :arg query: The search query.
    """
    loader = WikipediaLoader(query=query, load_max_docs=2)
    docs = loader.load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Searches the web for the given query using Tavily and returns the top results.
    :arg query: The search query.
    """
    tavily_search = TavilySearchResults(query=query, num_results=3)
    results = tavily_search.run()
    formatted_results = "\n\n---\n\n".join(
        [f'<Document source="{result["source"]}" page="{result.get("page", "")}"/>\n{result["content"]}\n</Document>'
         for result in results])
    return {"web_results": formatted_results}


@tool
def arvix_search(query: str) -> str:
    """Searches Arxiv for the given query and returns the top results.
    :arg query: The search query.
    """
    loader = ArxivLoader(query=query, load_max_docs=3)
    docs = loader.load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ])
    return {"arxiv_results": formatted_search_docs}


with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY"))
print("Supabase client created.")
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
print("Vector store initialized with Supabase.")
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)
print("Retriever tool created.")
tools = [
    add,
    subtract,
    multiply,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
]

def build_graph(provider: str = "huggingface") -> StateGraph:
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)  # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
        # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="huggingface")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()