# ADR-002: Use uv for Python Dependency Management

## Status
Accepted

## Context
The Ashworth Engine requires fast, reliable Python dependency management with modern tooling. Traditional pip and requirements.txt approaches are slower and less reliable for dependency resolution.

## Decision
We will adopt uv as the exclusive Python package manager, replacing pip and requirements.txt workflows.

## Rationale
- **Performance**: uv is significantly faster than pip for installs and dependency resolution
- **Better Dependency Resolution**: More reliable conflict resolution than pip
- **Modern Tooling**: Built on Rust with modern package management best practices
- **Lock Files**: Generates `uv.lock` files for reproducible builds
- **Project Integration**: Seamless integration with `pyproject.toml`

## Consequences
### Positive
- Faster dependency installations (10-100x faster than pip)
- More reliable dependency resolution
- Better developer experience with modern tooling
- Reproducible builds across environments

### Negative
- Team needs to learn new tool and commands
- Potential compatibility issues with legacy Python tooling
- Additional tool requirement in CI/CD pipelines

## Implementation
- Use `uv init .` for project initialization
- Use `uv venv` for virtual environment creation  
- Use `uv add <package>` for adding dependencies (without version pinning)
- Manually update `pyproject.toml` with specific versions after installation
- Eliminate all references to `requirements.txt` and `pip install`

## Migration Steps
1. Initialize project with `uv init .`
2. Create virtual environment with `uv venv` 
3. Install all dependencies using `uv add`
4. Update `pyproject.toml` with pinned versions from `uv pip list`
5. Remove any existing `requirements.txt` files