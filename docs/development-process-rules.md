# Development Process Rules for Ashworth Engine

## Overview
Development process rules and guidelines that should be stored in the RAG system for dynamic retrieval by agents involved in code generation, testing, and project management.

## GitHub Workflow Standards

### Pull Request Requirements

1. **PR Creation**
   - Use GitHub CLI for all non-trivial changes
   - Create feature branches from main for all work
   - Include detailed description of completed work

2. **PR Content**
   - Document acceptance criteria status
   - Include implementation statistics
   - Describe next-phase readiness
   - Ensure traceability and knowledge sharing

3. **PR Merging**
   - All PRs must be reviewed before merging
   - Update .gitignore to exclude unintended files
   - Verify all tests pass before merging

### Commit Message Standards

1. **Format**
   - Use conventional commit format: `<type>(<scope>): <subject>`
   - Types: feat, fix, chore, docs, style, refactor, test, perf
   - Scope: Optional, but recommended for context

2. **Content**
   - Be concise but descriptive
   - Focus on what changed and why
   - Reference issues or tickets when applicable

## Testing Requirements

### Unit Testing

1. **Coverage Standards**
   - Minimum 80% code coverage for new features
   - Test all error conditions and edge cases
   - Include both positive and negative test cases

2. **Test Structure**
   - Use pytest as the testing framework
   - Follow AAA pattern: Arrange, Act, Assert
   - Keep tests independent and isolated

### Integration Testing

1. **Workflow Testing**
   - Test sequential StateGraph pattern execution
   - Verify data dependency integrity between agents
   - Validate error handling and recovery mechanisms

2. **Compliance Testing**
   - Verify IRS compliance adherence
   - Test tax deduction rule application
   - Validate zero hallucination assurance

## Code Review Guidelines

### Review Process

1. **Pre-Review Checklist**
   - Code follows project style guidelines
   - All tests pass and new tests are included
   - Documentation is updated
   - No sensitive information in code

2. **Review Focus Areas**
   - Correctness and functionality
   - Performance and efficiency
   - Security and privacy
   - Maintainability and readability

### Approval Criteria

1. **Minimum Requirements**
   - At least one approval from team member
   - All critical and high severity issues addressed
   - Successful CI/CD pipeline execution

2. **Quality Standards**
   - Code is clear and well-documented
   - Follows established patterns and practices
   - No redundant or dead code

## Documentation Standards

### Technical Documentation

1. **Code Comments**
   - Explain why, not what
   - Focus on complex or non-obvious logic
   - Keep comments up to date with code changes

2. **API Documentation**
   - Document all public interfaces
   - Include examples for complex APIs
   - Specify parameter types and return values

### Project Documentation

1. **Phase Documentation**
   - Include clear objectives and success criteria
   - Document implementation steps in detail
   - Provide verification checkpoints

2. **Architecture Documentation**
   - Update when making significant changes
   - Include diagrams for complex systems
   - Document design decisions and rationale

## Implementation Guidance

### For Code Generation Agents

1. Always follow established coding patterns
2. Include appropriate error handling
3. Write tests for new functionality
4. Document complex logic and decisions

### For Testing Agents

1. Create comprehensive test suites
2. Include edge case testing
3. Validate compliance with project rules
4. Ensure test independence and repeatability

### For Documentation Agents

1. Keep documentation synchronized with code
2. Use clear and concise language
3. Include examples where helpful
4. Update documentation when code changes

## Examples

### Good Commit Message
```
feat(tax-categorizer): implement IRS Publication 334 compliance
- Add 11 official IRS expense categories
- Implement 50% business meal deduction rule
- Add Form 8300 warning for transactions ≥$10,000
```

### Poor Commit Message
```
Fixed stuff
```

## References

- GitHub CLI documentation
- Conventional Commits specification
- pytest documentation
- Project coding standards