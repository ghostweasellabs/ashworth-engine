# Phase 8: Testing and Containerization - Implementing the LangGraph Multi-Agent System

## Objective

Construct the core intelligence of the platform: a multi-agent system powered by LangGraph. This system will reason, delegate tasks between specialized agents, and utilize the knowledge base created in the previous phase.

## Key Technologies

- LangGraph for agentic workflows
- LangChain for AI integrations
- FastAPI for API exposure
- Supabase vector store for RAG

## Implementation Steps

### 4.1 Designing the Agentic Graph with LangGraph

1. Model the agent system as a stateful graph with a state object that persists information:
   - Initial user query
   - Conversational history
   - Retrieved documents
   - Intermediate agent outputs

2. Define key nodes and edges:
   
   **Nodes (Agents and Tools):**
   - Router: Conditional node that examines current state and decides next action
   - Retriever_Agent: Specialized agent to interact with knowledge base
   - Financial_Analyst_Agent: Higher-level reasoning agent for financial analysis
   
   **Edges:**
   - Conditional edges from Router for dynamic execution paths

### 4.2 Tooling the Retriever Agent

1. Create a custom tool using LangChain abstractions for similarity search:
   ```python
   # Function encapsulating similarity search logic
   def similarity_search_tool(query):
       # Instantiate SupabaseVectorStore
       # Connect to local Supabase instance
       # Use .as_retriever() method
       # Execute match_documents function
   ```

2. Integrate the tool with the Retriever_Agent:
   - The agent invokes the tool when needed
   - Pass user query to retriever
   - Return relevant document chunks

### 4.3 Exposing the Graph via a Streaming FastAPI Endpoint

1. Create a new streaming API endpoint `POST /chat` in the FastAPI application

2. Implement streaming responses:
   - Use FastAPI's StreamingResponse
   - Invoke LangGraph agent using .stream() method
   - Format yielded data as Server-Sent Events
   - Send chunks immediately to client

3. Implement session management:
   - Use unique session ID from client
   - Retrieve conversation history for given session
   - Include history in agent's state for context

## Checkpoint 8

The backend "engine" should be fully operational:
- Use an API client to send JSON payload with query to http://localhost:8001/chat
- Trigger the LangGraph execution flow
- Observe execution path through logs or streamed response:
  - Router directing query to retriever
  - Retriever tool fetching relevant document chunks
  - Financial analyst agent synthesizing final answer
- All steps should stream back in near real-time

## Success Criteria

- [ ] LangGraph stateful graph designed with proper nodes and edges
- [ ] Router node correctly directing workflow
- [ ] Retriever_Agent with custom similarity search tool
- [ ] Financial_Analyst_Agent for higher-level reasoning
- [ ] Conditional edges implemented for dynamic execution
- [ ] Custom tool for Supabase vector store queries
- [ ] POST /chat endpoint with streaming responses
- [ ] Session management for multi-turn conversations
- [ ] End-to-end testing successful with API client