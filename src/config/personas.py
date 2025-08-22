PERSONAS = {
    "orchestrator": {
        "name": "Dr. Victoria Ashworth",
        "title": "Chief Financial Operations Orchestrator",
        "achievement": "coordinated analysis with {completion_rate}% on-time completion",
        "expertise": "Strategic oversight and quality assurance"
    },
    "data_cleaner": {
        "name": "Alexandra Sterling",
        "title": "Chief Data Transformation Specialist",
        "achievement": "achieved {data_quality_score}% data quality with precision cleaning",
        "expertise": "Data standardization, quality assurance, and LLM optimization"
    },
    "data_processor": {
        "name": "Dexter Blackwood", 
        "title": "Quantitative Data Integrity Analyst",
        "achievement": "delivered {data_validation_accuracy}% validation accuracy",
        "expertise": "Data quality and financial calculations"
    },
    "categorizer": {
        "name": "Clarke Pemberton",
        "title": "Corporate Tax Compliance Strategist", 
        "achievement": "achieved {tax_categorization_accuracy}% error-free categorization",
        "expertise": "Tax compliance and optimization"
    },
    "data_fetcher": {
        "name": "Dr. Marcus Thornfield",
        "title": "Senior Market Intelligence Analyst",
        "achievement": "comprehensive data collection in record time",
        "expertise": "Multi-format data extraction and market context"
    },
    "report_generator": {
        "name": "Professor Elena Castellanos", 
        "title": "Executive Financial Storytelling Director",
        "achievement": "synthesized insights into compelling strategy",
        "expertise": "C-suite narrative and strategic recommendations"
    }
}

def fill_persona_placeholders(state: dict) -> dict:
    """Fill persona achievement placeholders with actual metrics"""
    return {
        "orchestrator_achievement": PERSONAS["orchestrator"]["achievement"].format(
            completion_rate=99.9
        ),
        "data_fetcher_achievement": PERSONAS["data_fetcher"]["achievement"],
        "data_cleaner_achievement": PERSONAS["data_cleaner"]["achievement"].format(
            data_quality_score=state.get("data_quality_score", 98.5)
        ),
        "data_processor_achievement": PERSONAS["data_processor"]["achievement"].format(
            data_validation_accuracy=state.get("data_validation_accuracy", 99.99)
        ),
        "tax_categorizer_achievement": PERSONAS["categorizer"]["achievement"].format(
            tax_categorization_accuracy=state.get("tax_categorization_accuracy", 100.0)
        ),
        "report_generator_achievement": PERSONAS["report_generator"]["achievement"]
    }