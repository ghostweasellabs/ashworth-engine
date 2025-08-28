"""
Professional chart generation for executive reports using Apache ECharts patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
import json


logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates professional charts and visualizations for executive reports.
    Uses Apache ECharts configuration patterns for consistency.
    """
    
    def __init__(self):
        """Initialize chart generator with professional themes."""
        self.professional_theme = {
            "color_palette": [
                "#2E86AB",  # Professional blue
                "#A23B72",  # Deep magenta
                "#F18F01",  # Warm orange
                "#C73E1D",  # Professional red
                "#6A994E",  # Forest green
                "#577590",  # Slate blue
                "#F8961E",  # Amber
                "#90323D"   # Burgundy
            ],
            "background_color": "#FFFFFF",
            "text_color": "#333333",
            "grid_color": "#E5E5E5",
            "font_family": "Arial, sans-serif",
            "title_font_size": 18,
            "label_font_size": 12,
            "legend_font_size": 12
        }
        
        self.chart_defaults = {
            "animation": True,
            "responsive": True,
            "maintain_aspect_ratio": True,
            "plugins": {
                "legend": {
                    "display": True,
                    "position": "bottom"
                },
                "tooltip": {
                    "enabled": True,
                    "mode": "index",
                    "intersect": False
                }
            }
        }
    
    async def create_pie_chart(
        self,
        data: Dict[str, Union[float, Decimal]],
        title: str,
        theme: str = "professional"
    ) -> Dict[str, Any]:
        """
        Create a professional pie chart configuration.
        
        Args:
            data: Dictionary mapping labels to values
            title: Chart title
            theme: Theme to apply
            
        Returns:
            Chart configuration dictionary
        """
        try:
            # Convert data to the format expected by ECharts
            chart_data = []
            total_value = sum(float(v) for v in data.values())
            
            for label, value in data.items():
                percentage = (float(value) / total_value * 100) if total_value > 0 else 0
                chart_data.append({
                    "name": label,
                    "value": float(value),
                    "percentage": round(percentage, 1),
                    "formatted_value": f"${float(value):,.2f}"
                })
            
            # Sort by value descending
            chart_data.sort(key=lambda x: x["value"], reverse=True)
            
            config = {
                "type": "pie",
                "title": {
                    "text": title,
                    "left": "center",
                    "textStyle": {
                        "fontSize": self.professional_theme["title_font_size"],
                        "color": self.professional_theme["text_color"],
                        "fontFamily": self.professional_theme["font_family"]
                    }
                },
                "tooltip": {
                    "trigger": "item",
                    "formatter": "{a} <br/>{b}: ${c} ({d}%)"
                },
                "legend": {
                    "orient": "horizontal",
                    "bottom": "10%",
                    "textStyle": {
                        "fontSize": self.professional_theme["legend_font_size"],
                        "color": self.professional_theme["text_color"]
                    }
                },
                "series": [{
                    "name": title,
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "center": ["50%", "45%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 5,
                        "borderColor": "#fff",
                        "borderWidth": 2
                    },
                    "label": {
                        "show": True,
                        "position": "outside",
                        "formatter": "{b}\n{d}%",
                        "fontSize": self.professional_theme["label_font_size"]
                    },
                    "emphasis": {
                        "label": {
                            "show": True,
                            "fontSize": 14,
                            "fontWeight": "bold"
                        },
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)"
                        }
                    },
                    "data": chart_data
                }],
                "color": self.professional_theme["color_palette"]
            }
            
            return {
                "data": chart_data,
                "config": config,
                "type": "pie",
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
            return self._create_error_chart("pie", title, str(e))
    
    async def create_line_chart(
        self,
        data: Dict[str, Union[float, Decimal]],
        title: str,
        theme: str = "professional",
        x_axis_label: str = "Period",
        y_axis_label: str = "Amount ($)"
    ) -> Dict[str, Any]:
        """
        Create a professional line chart configuration.
        
        Args:
            data: Dictionary mapping x-axis labels to values
            title: Chart title
            theme: Theme to apply
            x_axis_label: X-axis label
            y_axis_label: Y-axis label
            
        Returns:
            Chart configuration dictionary
        """
        try:
            # Sort data by key (assuming date format YYYY-MM)
            sorted_data = dict(sorted(data.items()))
            
            x_data = list(sorted_data.keys())
            y_data = [float(v) for v in sorted_data.values()]
            
            # Format data for display
            chart_data = [
                {
                    "period": period,
                    "value": value,
                    "formatted_value": f"${value:,.2f}"
                }
                for period, value in zip(x_data, y_data)
            ]
            
            config = {
                "type": "line",
                "title": {
                    "text": title,
                    "left": "center",
                    "textStyle": {
                        "fontSize": self.professional_theme["title_font_size"],
                        "color": self.professional_theme["text_color"],
                        "fontFamily": self.professional_theme["font_family"]
                    }
                },
                "tooltip": {
                    "trigger": "axis",
                    "formatter": "{b}<br/>{a}: ${c}"
                },
                "legend": {
                    "bottom": "5%",
                    "textStyle": {
                        "fontSize": self.professional_theme["legend_font_size"],
                        "color": self.professional_theme["text_color"]
                    }
                },
                "grid": {
                    "left": "10%",
                    "right": "10%",
                    "bottom": "15%",
                    "top": "15%",
                    "containLabel": True
                },
                "xAxis": {
                    "type": "category",
                    "boundaryGap": False,
                    "data": x_data,
                    "name": x_axis_label,
                    "nameLocation": "middle",
                    "nameGap": 30,
                    "axisLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"]
                        }
                    },
                    "axisLabel": {
                        "color": self.professional_theme["text_color"],
                        "fontSize": self.professional_theme["label_font_size"]
                    }
                },
                "yAxis": {
                    "type": "value",
                    "name": y_axis_label,
                    "nameLocation": "middle",
                    "nameGap": 50,
                    "axisLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"]
                        }
                    },
                    "axisLabel": {
                        "color": self.professional_theme["text_color"],
                        "fontSize": self.professional_theme["label_font_size"],
                        "formatter": "${value}"
                    },
                    "splitLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"],
                            "type": "dashed"
                        }
                    }
                },
                "series": [{
                    "name": y_axis_label,
                    "type": "line",
                    "smooth": True,
                    "symbol": "circle",
                    "symbolSize": 6,
                    "lineStyle": {
                        "width": 3,
                        "color": self.professional_theme["color_palette"][0]
                    },
                    "itemStyle": {
                        "color": self.professional_theme["color_palette"][0],
                        "borderColor": "#fff",
                        "borderWidth": 2
                    },
                    "areaStyle": {
                        "opacity": 0.1,
                        "color": self.professional_theme["color_palette"][0]
                    },
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0, 0, 0, 0.3)"
                        }
                    },
                    "data": y_data
                }],
                "color": self.professional_theme["color_palette"]
            }
            
            return {
                "data": chart_data,
                "config": config,
                "type": "line",
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Failed to create line chart: {e}")
            return self._create_error_chart("line", title, str(e))
    
    async def create_bar_chart(
        self,
        data: Dict[str, Union[float, Decimal]],
        title: str,
        theme: str = "professional",
        x_axis_label: str = "Category",
        y_axis_label: str = "Amount ($)"
    ) -> Dict[str, Any]:
        """
        Create a professional bar chart configuration.
        
        Args:
            data: Dictionary mapping categories to values
            title: Chart title
            theme: Theme to apply
            x_axis_label: X-axis label
            y_axis_label: Y-axis label
            
        Returns:
            Chart configuration dictionary
        """
        try:
            # Sort data by value descending
            sorted_items = sorted(data.items(), key=lambda x: float(x[1]), reverse=True)
            
            x_data = [item[0] for item in sorted_items]
            y_data = [float(item[1]) for item in sorted_items]
            
            # Format data for display
            chart_data = [
                {
                    "category": category,
                    "value": value,
                    "formatted_value": f"${value:,.2f}"
                }
                for category, value in zip(x_data, y_data)
            ]
            
            config = {
                "type": "bar",
                "title": {
                    "text": title,
                    "left": "center",
                    "textStyle": {
                        "fontSize": self.professional_theme["title_font_size"],
                        "color": self.professional_theme["text_color"],
                        "fontFamily": self.professional_theme["font_family"]
                    }
                },
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {
                        "type": "shadow"
                    },
                    "formatter": "{b}<br/>{a}: ${c}"
                },
                "grid": {
                    "left": "10%",
                    "right": "10%",
                    "bottom": "15%",
                    "top": "15%",
                    "containLabel": True
                },
                "xAxis": {
                    "type": "category",
                    "data": x_data,
                    "name": x_axis_label,
                    "nameLocation": "middle",
                    "nameGap": 30,
                    "axisLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"]
                        }
                    },
                    "axisLabel": {
                        "color": self.professional_theme["text_color"],
                        "fontSize": self.professional_theme["label_font_size"],
                        "rotate": 45,
                        "interval": 0
                    }
                },
                "yAxis": {
                    "type": "value",
                    "name": y_axis_label,
                    "nameLocation": "middle",
                    "nameGap": 50,
                    "axisLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"]
                        }
                    },
                    "axisLabel": {
                        "color": self.professional_theme["text_color"],
                        "fontSize": self.professional_theme["label_font_size"],
                        "formatter": "${value}"
                    },
                    "splitLine": {
                        "lineStyle": {
                            "color": self.professional_theme["grid_color"],
                            "type": "dashed"
                        }
                    }
                },
                "series": [{
                    "name": y_axis_label,
                    "type": "bar",
                    "barWidth": "60%",
                    "itemStyle": {
                        "borderRadius": [4, 4, 0, 0],
                        "color": {
                            "type": "linear",
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 1,
                            "colorStops": [
                                {
                                    "offset": 0,
                                    "color": self.professional_theme["color_palette"][0]
                                },
                                {
                                    "offset": 1,
                                    "color": self._lighten_color(self.professional_theme["color_palette"][0], 0.3)
                                }
                            ]
                        }
                    },
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0, 0, 0, 0.3)"
                        }
                    },
                    "data": y_data
                }],
                "color": self.professional_theme["color_palette"]
            }
            
            return {
                "data": chart_data,
                "config": config,
                "type": "bar",
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            return self._create_error_chart("bar", title, str(e))
    
    def _create_error_chart(self, chart_type: str, title: str, error: str) -> Dict[str, Any]:
        """
        Create an error chart when chart generation fails.
        
        Args:
            chart_type: Type of chart that failed
            title: Original chart title
            error: Error message
            
        Returns:
            Error chart configuration
        """
        return {
            "data": [],
            "config": {
                "type": "text",
                "title": {
                    "text": f"{title} (Error)",
                    "left": "center",
                    "textStyle": {
                        "color": "#FF0000"
                    }
                },
                "graphic": {
                    "type": "text",
                    "left": "center",
                    "top": "middle",
                    "style": {
                        "text": f"Chart generation failed:\n{error}",
                        "fontSize": 14,
                        "fill": "#666666"
                    }
                }
            },
            "type": chart_type,
            "title": title,
            "error": error
        }
    
    def _lighten_color(self, color: str, factor: float) -> str:
        """
        Lighten a hex color by a factor.
        
        Args:
            color: Hex color string
            factor: Lightening factor (0.0 to 1.0)
            
        Returns:
            Lightened hex color
        """
        try:
            # Remove # if present
            color = color.lstrip('#')
            
            # Convert to RGB
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            
            # Lighten
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception:
            # Return original color if lightening fails
            return color
    
    def export_chart_config(self, chart_config: Dict[str, Any], format: str = "json") -> str:
        """
        Export chart configuration in specified format.
        
        Args:
            chart_config: Chart configuration dictionary
            format: Export format ("json", "js")
            
        Returns:
            Formatted configuration string
        """
        try:
            if format.lower() == "json":
                return json.dumps(chart_config, indent=2, default=str)
            elif format.lower() == "js":
                return f"const chartConfig = {json.dumps(chart_config, indent=2, default=str)};"
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export chart config: {e}")
            return "{}"
    
    def get_theme_colors(self, theme: str = "professional") -> List[str]:
        """
        Get color palette for a theme.
        
        Args:
            theme: Theme name
            
        Returns:
            List of hex color codes
        """
        if theme == "professional":
            return self.professional_theme["color_palette"].copy()
        else:
            return self.professional_theme["color_palette"].copy()


# Global chart generator instance
_chart_generator = None


def get_chart_generator() -> ChartGenerator:
    """Get the global chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator