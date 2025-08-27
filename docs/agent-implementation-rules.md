# Agent Implementation Rules for Ashworth Engine

## Overview
Rules and guidelines for implementing agents in the Ashworth Engine that should be stored in the RAG system for dynamic retrieval by agents involved in development and orchestration.

## Agent-Persona Alignment

### Persona Requirements

1. **Exact Correspondence**
   - Each agent persona in configuration must correspond exactly to an implemented agent
   - Remove any non-existent or duplicate personas (e.g., 'orchestrator')
   - Maintain code consistency and avoid confusion

2. **Persona Definition**
   - Include clear role and responsibilities
   - Specify domain expertise and capabilities
   - Define interaction patterns with other agents

### Implementation Standards

1. **Agent Structure**
   - Follow consistent class structure across all agents
   - Implement standardized initialization patterns
   - Include proper error handling and logging

2. **Interface Compliance**
   - Adhere to defined agent interfaces
   - Implement required methods and properties
   - Maintain backward compatibility

## LangGraph Workflow Patterns

### Sequential Workflow Pattern

1. **When to Use**
   - For financial analysis workflows involving IRS compliance
   - When deterministic execution is required
   - For processes needing data dependency integrity

2. **Implementation**
   - Use StateGraph pattern instead of supervisor agent pattern
   - Ensure auditability of all steps
   - Maintain compliance with tax regulations

### Conditional Workflow Pattern

1. **When to Use**
   - For decision-making processes
   - When branching logic is required
   - For complex problem-solving scenarios

2. **Implementation**
   - Use conditional edges for dynamic execution paths
   - Implement proper state management
   - Include fallback mechanisms

## State Management Guidelines

### State Design

1. **Information Persistence**
   - Persist relevant information throughout execution
   - Include initial user query and conversational history
   - Store retrieved documents and intermediate outputs

2. **State Updates**
   - Update state only with verified information
   - Avoid storing sensitive or unnecessary data
   - Maintain consistency across agent interactions

### Error Handling

1. **Graceful Degradation**
   - Handle errors without crashing the workflow
   - Provide meaningful error messages to users
   - Log errors for debugging and improvement

2. **Recovery Mechanisms**
   - Implement retry logic for transient failures
   - Provide alternative approaches when primary methods fail
   - Allow users to correct input and retry

## RAG Integration Standards

### Rule Retrieval

1. **Context Identification**
   - Identify the domain and context of the request
   - Determine which rules are relevant
   - Formulate appropriate queries to the RAG system

2. **Rule Application**
   - Apply retrieved rules to guide responses
   - Cite rules when appropriate for transparency
   - Handle cases where no relevant rules are found

### Knowledge Base Interaction

1. **Query Formulation**
   - Create specific, targeted queries
   - Use clear and concise language
   - Include context when necessary

2. **Result Processing**
   - Filter and prioritize relevant information
   - Synthesize information from multiple sources
   - Verify accuracy and currency of information

## Implementation Guidance

### For Orchestrator Agent

1. **Workflow Management**
   - Coordinate agent execution in proper sequence
   - Handle data flow between agents
   - Monitor workflow progress and status

2. **Error Handling**
   - Detect and respond to agent failures
   - Provide fallback mechanisms
   - Communicate issues to users clearly

### For Specialized Agents

1. **Domain Expertise**
   - Focus on specific areas of expertise
   - Maintain deep knowledge in their domain
   - Collaborate effectively with other agents

2. **Quality Standards**
   - Provide accurate and reliable outputs
   - Follow domain-specific best practices
   - Continuously improve through feedback

## Examples

### Correct Agent Implementation
```python
class TaxCategorizerAgent:
    def __init__(self):
        # Initialize with IRS compliance rules from RAG
        self.rules = self.retrieve_rules("IRS compliance")
    
    def categorize_expense(self, expense_data):
        # Apply official IRS expense categories
        # Follow "Ordinary and Necessary" standard
        # Flag potential audit triggers
        pass
```

### Incorrect Agent Implementation
```python
class TaxCategorizerAgent:
    def __init__(self):
        # Hard-coded rules instead of RAG retrieval
        self.rules = {
            "meals": "50% deductible",
            # ... other hard-coded rules
        }
```

## References

- LangGraph documentation
- Project architecture documentation
- Agent persona definitions
- RAG system documentation