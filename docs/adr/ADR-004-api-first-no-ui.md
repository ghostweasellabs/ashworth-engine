# ADR-004: API-First, No Dedicated UI

## Status
Accepted

## Context
The Ashworth Engine is designed to serve small and medium businesses (SMBs) who may want to integrate financial intelligence capabilities into their existing tools and workflows rather than adopt another standalone application.

## Decision
We will provide functionality exclusively via REST API initially, without developing a dedicated user interface.

## Rationale
- **Integration Focus**: SMB clients often prefer to integrate capabilities into existing tools
- **Resource Efficiency**: Allows focus on core AI and data processing capabilities
- **Flexibility**: Different clients can build custom UIs suited to their workflows  
- **Faster Time-to-Market**: Eliminates frontend development complexity
- **API-First Design**: Ensures robust, well-documented APIs as primary interface

## Consequences
### Positive
- Faster development and deployment
- More flexible integration options for clients
- Forces good API design and documentation
- Lower maintenance overhead
- Can focus resources on AI/ML capabilities

### Negative
- Higher barrier to entry for non-technical users
- Relies on clients or partners to build UI layers
- More complex user onboarding process
- May limit adoption for some customer segments

## Implementation
- Build comprehensive REST API with FastAPI
- Provide detailed API documentation (OpenAPI/Swagger)
- Include example integration code and SDKs
- Design APIs with frontend integration in mind
- Consider future UI development as separate project

## Future Considerations
- Monitor customer feedback on UI requirements
- Consider partnering with UI/UX teams for client projects
- Evaluate building reference UI implementations
- Assess market demand for standalone application