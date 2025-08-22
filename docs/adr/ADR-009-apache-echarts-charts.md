# ADR-009: Use Apache ECharts (pyecharts) for Chart Generation

## Status
Accepted

## Context
The Ashworth Engine requires professional chart generation for financial reports and visualizations. Charts need to be high-quality, interactive when possible, and suitable for embedding in both web interfaces and PDF reports.

## Decision
We will use pyecharts (Apache ECharts Python wrapper) as the primary chart generation library, replacing matplotlib.

## Rationale
- **Superior Visual Quality**: Apache ECharts produces more professional, modern-looking charts
- **Interactive Charts**: Built-in support for interactive features when displayed in web contexts
- **Better Theming**: Rich set of professional themes suitable for business reports
- **Performance**: Lightweight and fast rendering for web applications
- **Consistency**: Charts maintain consistent styling across different output formats
- **Business Focus**: Better suited for financial and business intelligence visualizations

## Consequences
### Positive
- Professional-quality charts that enhance report credibility
- Interactive features improve user experience in web contexts
- Better integration with modern web technologies
- Rich theming options for brand consistency
- Lightweight library with good performance

### Negative
- Team needs to learn new charting library syntax
- Different API compared to matplotlib ecosystem
- May require additional configuration for PDF embedding
- Less extensive scientific plotting features than matplotlib

## Implementation
- Use `pyecharts` for all chart generation in `src/utils/chart_generation.py`
- Use `snapshot-selenium` for converting charts to PNG format for PDF embedding
- Generate charts as HTML/JavaScript for web display
- Convert to PNG/SVG for PDF embedding when needed
- Maintain consistency with seaborn and plotly for specialized use cases
- Configure professional themes suitable for financial reports

## Migration Impact
- Replace existing matplotlib chart generation code
- Update documentation to reference Apache ECharts/pyecharts
- Ensure chart outputs are compatible with PDF generation pipeline
- Test chart rendering in both web and PDF contexts