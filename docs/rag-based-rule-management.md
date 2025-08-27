# RAG-Based Rule Management for Ashworth Engine v2

## Objective

Instead of hard-coding rules, guidelines, and policies into the agent implementations or documentation, store them as documents in the RAG system. This allows the LLM to dynamically retrieve and apply relevant rules based on context, making the system more flexible and maintainable.

## Approach

All project rules, guidelines, policies, and best practices should be ingested into the RAG knowledge base as structured documents. The agents can then query the RAG system to retrieve relevant rules when needed, rather than having them hard-coded in prompts or system instructions.

## Implementation Strategy

### 1. Document Structure for Rules

Create structured documents for different categories of rules:

```
# [Category] Rules

## Overview
Brief description of the rule category

## Specific Rules
1. **Rule Name**: Description of the rule
   - Context when to apply
   - Implementation guidance
   - Examples if applicable

2. **Rule Name**: Description of the rule
   - Context when to apply
   - Implementation guidance
   - Examples if applicable
```

### 2. Categories of Rules to Include in RAG

1. **IRS Compliance Rules**
   - Official IRS Publication 334 guidelines
   - Tax deduction rules and limitations
   - Audit trigger identification
   - Business expense categorization standards
   - See: [irs-compliance-rules.md](irs-compliance-rules.md)

2. **Development Process Rules**
   - GitHub CLI workflow requirements
   - PR documentation standards
   - Testing requirements
   - Code review guidelines
   - See: [development-process-rules.md](development-process-rules.md)

3. **Agent Implementation Rules**
   - Persona alignment requirements
   - LangGraph workflow patterns
   - State management guidelines
   - Error handling standards
   - See: [agent-implementation-rules.md](agent-implementation-rules.md)

4. **Technology Stack Rules**
   - Library usage policies (e.g., pyecharts vs matplotlib)
   - Package manager preferences (yarn vs pnpm)
   - Database integration standards
   - Security best practices
   - See: [technology-stack-rules.md](technology-stack-rules.md)

5. **Project Structure Rules**
   - Directory organization standards
   - File naming conventions
   - Configuration management
   - Environment variable usage

### 3. Ingestion Process

1. Create markdown documents for each rule category
2. Use the existing RAG ingestion pipeline to process these documents
3. Ensure documents are properly chunked for optimal retrieval
4. Add metadata to facilitate filtering and retrieval

### 4. Agent Usage Pattern

Agents should follow this pattern when processing requests:

1. Identify the context and domain of the request
2. Query the RAG system for relevant rules:
   ```
   "Retrieve rules for {domain} related to {specific_context}"
   ```
3. Apply the retrieved rules to guide the response
4. Cite the rules when appropriate for transparency

## Benefits

1. **Flexibility**: Rules can be updated without changing agent code
2. **Maintainability**: Centralized rule management
3. **Context Awareness**: Agents can retrieve only relevant rules
4. **Scalability**: Easy to add new rule categories
5. **Auditability**: Clear record of which rules were applied

## Example Rule Document

# IRS Compliance Rules

## Overview
Rules and guidelines for ensuring all tax-related processing follows official IRS guidelines with zero tolerance for hallucinations.

## Specific Rules

1. **Zero Hallucination Policy**: 
   - All tax-related processing must be grounded in verifiable tax rules and documentation
   - Cross-reference all tax advice with official IRS publications
   - Never generate tax advice not explicitly supported by official documentation

2. **IRS Publication Compliance**: 
   - Use current IRS Publication 334 (Tax Guide for Small Business) as the authoritative source
   - Verify publication currency before implementing tax-related features
   - Discontinue use of outdated publications (e.g., Publication 535)

3. **Expense Categorization Standards**: 
   - Implement all 11 official IRS expense categories with proper codes
   - Enforce the "Ordinary and Necessary" standard for all categorizations
   - Apply specific deduction rules (e.g., 50% business meal deduction)

## Implementation

These rules should be retrieved by any agent processing tax-related information and applied to ensure compliance with IRS guidelines.

## Success Criteria

- [ ] All project rules documented as RAG-compatible documents
- [ ] Agents successfully retrieving and applying rules from RAG
- [ ] No hard-coded rules in agent implementations
- [ ] Dynamic rule application based on context
- [ ] Rule updates possible without agent code changes