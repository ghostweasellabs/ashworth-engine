# ADR-001: Adopt LangGraph for Orchestration

## Status
Accepted

## Context
The Ashworth Engine requires a robust, stateful multi-agent orchestration system to manage complex financial data processing workflows. The system needs to coordinate multiple agents (data fetcher, cleaner, processor, categorizer, report generator) while maintaining state across the workflow.

## Decision
We will use LangGraph's StateGraph for agent workflow management and orchestration.

## Rationale
- **Stateful Workflows**: LangGraph provides built-in state management across multi-step workflows
- **Agent Coordination**: Native support for @task decorated functions and agent interactions  
- **Error Handling**: Robust error handling and recovery mechanisms
- **Observability**: Built-in tracing and monitoring capabilities
- **Scalability**: Can handle complex branching logic and conditional workflows

## Consequences
### Positive
- Clear separation of concerns between agents
- Reliable state persistence across workflow steps
- Built-in retry and error handling mechanisms
- Excellent debugging and monitoring capabilities

### Negative
- Additional dependency on LangGraph framework
- Learning curve for team members unfamiliar with LangGraph
- Potential vendor lock-in to LangSmith ecosystem

## Implementation
- Configure `langgraph.json` for project structure
- Define StateGraph in `src/workflows/financial_analysis.py`
- Implement @task decorated functions in agent modules
- Use TypedDict for state schema definitions