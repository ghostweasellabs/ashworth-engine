"""
Agentic reasoning engine for intelligent rule management and contextual understanding.
Provides context-aware query formulation, rule conflict resolution, and citation tracking.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.utils.rag.similarity_search import SearchConfig, SearchStrategy, SimilaritySearchEngine
from src.utils.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for rule applications."""
    HIGH = "high"        # 0.8-1.0: Clear, unambiguous rule application
    MEDIUM = "medium"    # 0.6-0.8: Some interpretation required
    LOW = "low"          # 0.4-0.6: Significant ambiguity or uncertainty
    VERY_LOW = "very_low"  # 0.0-0.4: Highly uncertain, recommend professional consultation


class RuleConflictType(str, Enum):
    """Types of conflicts between rules."""
    CONTRADICTORY = "contradictory"    # Rules directly contradict each other
    OVERLAPPING = "overlapping"        # Rules have overlapping scope
    HIERARCHICAL = "hierarchical"      # One rule supersedes another
    CONTEXTUAL = "contextual"          # Conflict depends on specific context


class Citation(BaseModel):
    """Citation for a rule or regulation."""
    document_id: str
    document_title: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    publication_year: Optional[int] = None
    url: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)


class RuleInterpretation(BaseModel):
    """An interpretation of a rule with confidence and reasoning."""
    rule_text: str
    interpretation: str
    confidence: ConfidenceLevel
    reasoning: str
    citations: List[Citation]
    applicable_contexts: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class RuleConflict(BaseModel):
    """Represents a conflict between multiple rules."""
    conflict_type: RuleConflictType
    conflicting_rules: List[RuleInterpretation]
    resolution_strategy: str
    recommended_interpretation: Optional[RuleInterpretation] = None
    confidence: ConfidenceLevel
    reasoning: str


class AgenticQuery(BaseModel):
    """Context-aware query formulated by the agentic system."""
    original_query: str
    enhanced_query: str
    context_factors: List[str]
    domain_keywords: List[str]
    search_strategy: SearchStrategy
    expected_rule_types: List[str] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    """Result from the agentic reasoning engine."""
    query: AgenticQuery
    rule_interpretations: List[RuleInterpretation]
    conflicts: List[RuleConflict] = Field(default_factory=list)
    synthesized_guidance: str
    overall_confidence: ConfidenceLevel
    recommendations: List[str] = Field(default_factory=list)
    audit_trail: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class AgenticReasoningEngine:
    """
    Agentic reasoning engine that provides intelligent rule retrieval and reasoning.
    Goes beyond simple similarity search to understand context and resolve conflicts.
    """
    
    def __init__(self):
        self.search_engine = SimilaritySearchEngine()
        
        # Domain-specific knowledge for query enhancement
        self.tax_domains = {
            "business_expenses": ["deduction", "ordinary", "necessary", "business purpose"],
            "depreciation": ["asset", "recovery", "useful life", "section 179"],
            "meals_entertainment": ["50%", "business meal", "entertainment", "substantiation", "meal"],
            "travel": ["business travel", "transportation", "lodging", "per diem"],
            "home_office": ["exclusive use", "regular use", "principal place", "deduction", "home office"]
        }
        
        # Rule hierarchy for conflict resolution
        self.rule_hierarchy = {
            "irs_code": 1,           # Highest authority
            "irs_regulation": 2,      # Treasury regulations
            "irs_publication": 3,     # IRS publications and guidance
            "court_decision": 4,      # Tax court decisions
            "irs_ruling": 5,         # Private letter rulings, etc.
        }
    
    async def reason_about_query(
        self, 
        query: str, 
        context: Dict[str, any] = None
    ) -> ReasoningResult:
        """
        Main reasoning function that processes a query with full agentic capabilities.
        
        Args:
            query: The original query
            context: Additional context (transaction details, client info, etc.)
            
        Returns:
            ReasoningResult with comprehensive analysis
        """
        try:
            logger.info(f"Starting agentic reasoning for query: {query[:100]}...")
            
            # Step 1: Formulate context-aware query
            enhanced_query = await self._formulate_context_aware_query(query, context)
            
            # Step 2: Retrieve relevant rules using multiple strategies
            search_results = await self._retrieve_rules_multi_strategy(enhanced_query)
            
            # Step 3: Analyze and interpret rules
            rule_interpretations = await self._analyze_and_interpret_rules(
                search_results, enhanced_query, context
            )
            
            # Step 4: Detect and resolve conflicts
            conflicts = await self._detect_and_resolve_conflicts(rule_interpretations)
            
            # Step 5: Synthesize guidance
            synthesized_guidance = await self._synthesize_guidance(
                rule_interpretations, conflicts, enhanced_query
            )
            
            # Step 6: Determine overall confidence
            overall_confidence = self._calculate_overall_confidence(
                rule_interpretations, conflicts
            )
            
            # Step 7: Generate recommendations
            recommendations = await self._generate_recommendations(
                rule_interpretations, conflicts, overall_confidence
            )
            
            # Step 8: Create audit trail
            audit_trail = self._create_audit_trail(
                enhanced_query, rule_interpretations, conflicts
            )
            
            result = ReasoningResult(
                query=enhanced_query,
                rule_interpretations=rule_interpretations,
                conflicts=conflicts,
                synthesized_guidance=synthesized_guidance,
                overall_confidence=overall_confidence,
                recommendations=recommendations,
                audit_trail=audit_trail
            )
            
            logger.info(f"Completed agentic reasoning with {len(rule_interpretations)} interpretations")
            return result
            
        except Exception as e:
            logger.error(f"Error in agentic reasoning: {e}")
            # Return conservative fallback result
            return self._create_fallback_result(query, str(e))
    
    async def _formulate_context_aware_query(
        self, 
        query: str, 
        context: Dict[str, any] = None
    ) -> AgenticQuery:
        """Enhance the original query with context and domain knowledge."""
        
        # Identify domain from query
        domain = self._identify_domain(query)
        
        # Extract context factors
        context_factors = []
        if context:
            if "transaction_amount" in context:
                context_factors.append(f"amount: ${context['transaction_amount']}")
            if "transaction_date" in context:
                context_factors.append(f"date: {context['transaction_date']}")
            if "business_type" in context:
                context_factors.append(f"business: {context['business_type']}")
            if "client_size" in context:
                context_factors.append(f"size: {context['client_size']}")
        
        # Get domain-specific keywords
        domain_keywords = self.tax_domains.get(domain, [])
        
        # Enhance query with domain knowledge
        enhanced_query_parts = [query]
        if domain_keywords:
            enhanced_query_parts.extend(domain_keywords[:3])  # Add top 3 keywords
        
        enhanced_query = " ".join(enhanced_query_parts)
        
        # Determine search strategy based on query complexity
        search_strategy = self._determine_search_strategy(query, context)
        
        # Identify expected rule types
        expected_rule_types = self._identify_expected_rule_types(query, domain)
        
        return AgenticQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            context_factors=context_factors,
            domain_keywords=domain_keywords,
            search_strategy=search_strategy,
            expected_rule_types=expected_rule_types
        )
    
    def _identify_domain(self, query: str) -> str:
        """Identify the tax domain from the query."""
        query_lower = query.lower()
        
        # Check for specific domains first (more specific matches)
        if "meal" in query_lower or "entertainment" in query_lower:
            return "meals_entertainment"
        elif "home office" in query_lower:
            return "home_office"
        elif "travel" in query_lower:
            return "travel"
        elif "depreciation" in query_lower or "asset" in query_lower:
            return "depreciation"
        elif "deduction" in query_lower or "expense" in query_lower:
            return "business_expenses"
        
        return "general"
    
    def _determine_search_strategy(
        self, 
        query: str, 
        context: Dict[str, any] = None
    ) -> SearchStrategy:
        """Determine the best search strategy based on query and context."""
        
        # Use strict strategy for compliance-critical queries
        compliance_keywords = ["deduction", "irs", "tax", "compliance", "audit"]
        if any(keyword in query.lower() for keyword in compliance_keywords):
            return SearchStrategy.STRICT
        
        # Use adaptive for complex queries
        if len(query.split()) > 10 or (context and len(context) > 3):
            return SearchStrategy.ADAPTIVE
        
        return SearchStrategy.BALANCED
    
    def _identify_expected_rule_types(self, query: str, domain: str) -> List[str]:
        """Identify what types of rules we expect to find."""
        rule_types = ["irs_publication"]
        
        if "code" in query.lower() or "section" in query.lower():
            rule_types.append("irs_code")
        
        if "regulation" in query.lower() or "reg" in query.lower():
            rule_types.append("irs_regulation")
        
        return rule_types
    
    async def _retrieve_rules_multi_strategy(
        self, 
        enhanced_query: AgenticQuery
    ) -> List[SearchResult]:
        """Retrieve rules using multiple search strategies for comprehensive coverage."""
        
        all_results = []
        
        # Primary search with enhanced query
        config = SearchConfig(
            strategy=enhanced_query.search_strategy,
            max_results=15,
            document_type_filter="irs_publication",
            rerank_results=True,
            include_context=True
        )
        
        primary_results = await self.search_engine.search(
            enhanced_query.enhanced_query, config
        )
        all_results.extend(primary_results)
        
        # Secondary search with original query for comparison
        if enhanced_query.enhanced_query != enhanced_query.original_query:
            secondary_results = await self.search_engine.search(
                enhanced_query.original_query, config
            )
            
            # Add unique results
            existing_chunks = {(r.document_id, r.chunk_index) for r in all_results}
            for result in secondary_results:
                if (result.document_id, result.chunk_index) not in existing_chunks:
                    all_results.append(result)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit to top results
        return all_results[:20]
    
    async def _analyze_and_interpret_rules(
        self,
        search_results: List[SearchResult],
        query: AgenticQuery,
        context: Dict[str, any] = None
    ) -> List[RuleInterpretation]:
        """Analyze search results and create rule interpretations."""
        
        interpretations = []
        
        for result in search_results:
            try:
                # Create citation
                citation = Citation(
                    document_id=result.document_id,
                    document_title=result.metadata.get("title", "Unknown Document"),
                    section=result.metadata.get("section"),
                    publication_year=result.metadata.get("publication_year"),
                    confidence_score=result.similarity_score
                )
                
                # Determine confidence level based on similarity score and content quality
                confidence = self._determine_confidence_level(result, query, context)
                
                # Generate interpretation
                interpretation_text = self._generate_interpretation(result, query, context)
                
                # Identify applicable contexts
                applicable_contexts = self._identify_applicable_contexts(result, context)
                
                # Identify limitations
                limitations = self._identify_limitations(result, query)
                
                # Generate reasoning
                reasoning = self._generate_reasoning(result, query, confidence)
                
                rule_interpretation = RuleInterpretation(
                    rule_text=result.content,
                    interpretation=interpretation_text,
                    confidence=confidence,
                    reasoning=reasoning,
                    citations=[citation],
                    applicable_contexts=applicable_contexts,
                    limitations=limitations
                )
                
                interpretations.append(rule_interpretation)
                
            except Exception as e:
                logger.error(f"Error analyzing rule result: {e}")
                continue
        
        return interpretations
    
    def _determine_confidence_level(
        self,
        result: SearchResult,
        query: AgenticQuery,
        context: Dict[str, any] = None
    ) -> ConfidenceLevel:
        """Determine confidence level for a rule interpretation."""
        
        score = result.similarity_score
        
        # Adjust based on content quality indicators
        content = result.content.lower()
        
        # Higher confidence for specific regulatory language
        if any(phrase in content for phrase in ["shall", "must", "required", "prohibited"]):
            score += 0.1
        
        # Lower confidence for ambiguous language
        if any(phrase in content for phrase in ["may", "generally", "typically", "usually"]):
            score -= 0.1
        
        # Higher confidence for exact keyword matches
        query_words = set(query.original_query.lower().split())
        content_words = set(content.split())
        overlap_ratio = len(query_words.intersection(content_words)) / len(query_words)
        score += overlap_ratio * 0.1
        
        # Map to confidence levels
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_interpretation(
        self,
        result: SearchResult,
        query: AgenticQuery,
        context: Dict[str, any] = None
    ) -> str:
        """Generate a human-readable interpretation of the rule."""
        
        # For now, provide a structured interpretation
        # In a full implementation, this could use an LLM for natural language generation
        
        content = result.content
        
        # Extract key points
        key_phrases = []
        if "deductible" in content.lower():
            key_phrases.append("This expense may be deductible")
        if "not deductible" in content.lower():
            key_phrases.append("This expense is not deductible")
        if "50%" in content:
            key_phrases.append("Subject to 50% limitation")
        if "ordinary and necessary" in content.lower():
            key_phrases.append("Must be ordinary and necessary business expense")
        
        if key_phrases:
            interpretation = ". ".join(key_phrases) + "."
        else:
            # Fallback to first sentence of content
            sentences = content.split('.')
            interpretation = sentences[0].strip() + "." if sentences else content[:200] + "..."
        
        return interpretation
    
    def _identify_applicable_contexts(
        self,
        result: SearchResult,
        context: Dict[str, any] = None
    ) -> List[str]:
        """Identify contexts where this rule applies."""
        
        contexts = []
        content = result.content.lower()
        
        # Business contexts
        if "business" in content:
            contexts.append("business expenses")
        if "personal" in content:
            contexts.append("personal expenses")
        if "employee" in content:
            contexts.append("employee expenses")
        
        # Amount contexts
        if context and "transaction_amount" in context:
            amount = context["transaction_amount"]
            if amount >= 5000:
                contexts.append("large expenses")
            elif amount < 100:
                contexts.append("small expenses")
        
        return contexts
    
    def _identify_limitations(self, result: SearchResult, query: AgenticQuery) -> List[str]:
        """Identify limitations or exceptions to the rule."""
        
        limitations = []
        content = result.content.lower()
        
        # Common limitation patterns
        if "except" in content or "unless" in content:
            limitations.append("Contains exceptions - review full text")
        if "subject to" in content:
            limitations.append("Subject to additional requirements")
        if "may" in content and "deduct" in content:
            limitations.append("Deductibility depends on specific circumstances")
        if "substantiation" in content:
            limitations.append("Requires proper documentation")
        
        return limitations
    
    def _generate_reasoning(
        self,
        result: SearchResult,
        query: AgenticQuery,
        confidence: ConfidenceLevel
    ) -> str:
        """Generate reasoning for the interpretation."""
        
        reasoning_parts = []
        
        # Similarity reasoning
        reasoning_parts.append(
            f"Based on {result.similarity_score:.1%} similarity to query"
        )
        
        # Source reasoning
        doc_title = result.metadata.get("title", "IRS document")
        reasoning_parts.append(f"Found in {doc_title}")
        
        # Confidence reasoning
        if confidence == ConfidenceLevel.HIGH:
            reasoning_parts.append("High confidence due to clear regulatory language")
        elif confidence == ConfidenceLevel.LOW:
            reasoning_parts.append("Lower confidence due to ambiguous language")
        
        return ". ".join(reasoning_parts) + "."
    
    async def _detect_and_resolve_conflicts(
        self,
        interpretations: List[RuleInterpretation]
    ) -> List[RuleConflict]:
        """Detect conflicts between rule interpretations and provide resolution strategies."""
        
        conflicts = []
        
        # Group interpretations by topic/domain
        topic_groups = self._group_interpretations_by_topic(interpretations)
        
        for topic, group_interpretations in topic_groups.items():
            if len(group_interpretations) < 2:
                continue
            
            # Check for conflicts within the group
            topic_conflicts = self._find_conflicts_in_group(group_interpretations, topic)
            conflicts.extend(topic_conflicts)
        
        return conflicts
    
    def _group_interpretations_by_topic(
        self,
        interpretations: List[RuleInterpretation]
    ) -> Dict[str, List[RuleInterpretation]]:
        """Group interpretations by topic for conflict detection."""
        
        groups = {}
        
        for interpretation in interpretations:
            # Simple topic identification based on content
            content = interpretation.rule_text.lower()
            
            topic = "general"
            if "deduction" in content:
                topic = "deductions"
            elif "meal" in content:
                topic = "meals"
            elif "travel" in content:
                topic = "travel"
            elif "depreciation" in content:
                topic = "depreciation"
            
            if topic not in groups:
                groups[topic] = []
            groups[topic].append(interpretation)
        
        return groups
    
    def _find_conflicts_in_group(
        self,
        interpretations: List[RuleInterpretation],
        topic: str
    ) -> List[RuleConflict]:
        """Find conflicts within a group of interpretations."""
        
        conflicts = []
        
        # Simple conflict detection based on contradictory language
        for i, interp1 in enumerate(interpretations):
            for j, interp2 in enumerate(interpretations[i+1:], i+1):
                
                conflict_type = self._detect_conflict_type(interp1, interp2)
                
                if conflict_type:
                    # Resolve the conflict
                    resolution_strategy, recommended = self._resolve_conflict(
                        interp1, interp2, conflict_type
                    )
                    
                    conflict = RuleConflict(
                        conflict_type=conflict_type,
                        conflicting_rules=[interp1, interp2],
                        resolution_strategy=resolution_strategy,
                        recommended_interpretation=recommended,
                        confidence=self._calculate_conflict_confidence(interp1, interp2),
                        reasoning=f"Conflict detected in {topic} rules: {resolution_strategy}"
                    )
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_conflict_type(
        self,
        interp1: RuleInterpretation,
        interp2: RuleInterpretation
    ) -> Optional[RuleConflictType]:
        """Detect if two interpretations conflict and determine the type."""
        
        content1 = interp1.interpretation.lower()
        content2 = interp2.interpretation.lower()
        
        # Check for contradictory statements
        if ("deductible" in content1 and "not deductible" in content2) or \
           ("not deductible" in content1 and "deductible" in content2):
            return RuleConflictType.CONTRADICTORY
        
        # Check for overlapping scope with different requirements
        if ("50%" in content1 and "100%" in content2) or \
           ("100%" in content1 and "50%" in content2):
            return RuleConflictType.OVERLAPPING
        
        # Check for hierarchical conflicts (different authority levels)
        doc1_type = interp1.citations[0].document_id if interp1.citations else ""
        doc2_type = interp2.citations[0].document_id if interp2.citations else ""
        
        if "code" in doc1_type and "publication" in doc2_type:
            return RuleConflictType.HIERARCHICAL
        
        return None
    
    def _resolve_conflict(
        self,
        interp1: RuleInterpretation,
        interp2: RuleInterpretation,
        conflict_type: RuleConflictType
    ) -> Tuple[str, Optional[RuleInterpretation]]:
        """Resolve a conflict between two interpretations."""
        
        if conflict_type == RuleConflictType.HIERARCHICAL:
            # Higher authority wins
            if interp1.citations and interp2.citations:
                doc1_id = interp1.citations[0].document_id
                doc2_id = interp2.citations[0].document_id
                
                # Determine hierarchy
                hierarchy1 = self._get_document_hierarchy(doc1_id)
                hierarchy2 = self._get_document_hierarchy(doc2_id)
                
                if hierarchy1 < hierarchy2:  # Lower number = higher authority
                    return "Higher authority rule takes precedence", interp1
                else:
                    return "Higher authority rule takes precedence", interp2
        
        elif conflict_type == RuleConflictType.CONTRADICTORY:
            # Choose higher confidence interpretation
            if interp1.confidence.value == "high" and interp2.confidence.value != "high":
                return "Higher confidence interpretation selected", interp1
            elif interp2.confidence.value == "high" and interp1.confidence.value != "high":
                return "Higher confidence interpretation selected", interp2
            else:
                return "Conservative interpretation recommended - consult professional", None
        
        elif conflict_type == RuleConflictType.OVERLAPPING:
            # Apply more restrictive rule
            return "More restrictive rule applied for compliance", None
        
        return "Manual review required", None
    
    def _get_document_hierarchy(self, document_id: str) -> int:
        """Get hierarchy level for a document type."""
        
        doc_id_lower = document_id.lower()
        
        if "code" in doc_id_lower:
            return self.rule_hierarchy["irs_code"]
        elif "regulation" in doc_id_lower:
            return self.rule_hierarchy["irs_regulation"]
        elif "pub" in doc_id_lower or "publication" in doc_id_lower:
            return self.rule_hierarchy["irs_publication"]
        else:
            return 10  # Unknown, lowest priority
    
    def _calculate_conflict_confidence(
        self,
        interp1: RuleInterpretation,
        interp2: RuleInterpretation
    ) -> ConfidenceLevel:
        """Calculate confidence in conflict resolution."""
        
        # If both interpretations have high confidence, conflict resolution is medium
        if interp1.confidence == ConfidenceLevel.HIGH and interp2.confidence == ConfidenceLevel.HIGH:
            return ConfidenceLevel.MEDIUM
        
        # If one has low confidence, resolution confidence is higher
        if interp1.confidence == ConfidenceLevel.LOW or interp2.confidence == ConfidenceLevel.LOW:
            return ConfidenceLevel.MEDIUM
        
        return ConfidenceLevel.LOW
    
    async def _synthesize_guidance(
        self,
        interpretations: List[RuleInterpretation],
        conflicts: List[RuleConflict],
        query: AgenticQuery
    ) -> str:
        """Synthesize comprehensive guidance from all interpretations and conflicts."""
        
        if not interpretations:
            return "No relevant rules found. Recommend consulting with a tax professional."
        
        guidance_parts = []
        
        # Start with highest confidence interpretation
        high_confidence_interps = [
            i for i in interpretations if i.confidence == ConfidenceLevel.HIGH
        ]
        
        if high_confidence_interps:
            primary_interp = high_confidence_interps[0]
            guidance_parts.append(f"Primary guidance: {primary_interp.interpretation}")
        else:
            # Use highest scoring interpretation
            primary_interp = max(interpretations, key=lambda x: x.citations[0].confidence_score)
            guidance_parts.append(f"Based on available rules: {primary_interp.interpretation}")
        
        # Add conflict resolutions
        if conflicts:
            guidance_parts.append("Important considerations:")
            for conflict in conflicts:
                guidance_parts.append(f"- {conflict.resolution_strategy}")
        
        # Add limitations and warnings
        all_limitations = []
        for interp in interpretations:
            all_limitations.extend(interp.limitations)
        
        if all_limitations:
            unique_limitations = list(set(all_limitations))
            guidance_parts.append("Limitations: " + "; ".join(unique_limitations))
        
        return " ".join(guidance_parts)
    
    def _calculate_overall_confidence(
        self,
        interpretations: List[RuleInterpretation],
        conflicts: List[RuleConflict]
    ) -> ConfidenceLevel:
        """Calculate overall confidence in the reasoning result."""
        
        if not interpretations:
            return ConfidenceLevel.VERY_LOW
        
        # If there are unresolved conflicts, reduce confidence
        if conflicts:
            unresolved_conflicts = [c for c in conflicts if c.recommended_interpretation is None]
            if unresolved_conflicts:
                return ConfidenceLevel.LOW
        
        # Calculate based on interpretation confidences
        confidence_scores = {
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.VERY_LOW: 1
        }
        
        avg_score = sum(confidence_scores[i.confidence] for i in interpretations) / len(interpretations)
        
        if avg_score >= 3.5:
            return ConfidenceLevel.HIGH
        elif avg_score >= 2.5:
            return ConfidenceLevel.MEDIUM
        elif avg_score >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _generate_recommendations(
        self,
        interpretations: List[RuleInterpretation],
        conflicts: List[RuleConflict],
        overall_confidence: ConfidenceLevel
    ) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        
        recommendations = []
        
        # Confidence-based recommendations
        if overall_confidence == ConfidenceLevel.VERY_LOW:
            recommendations.append("Consult with a qualified tax professional before proceeding")
        elif overall_confidence == ConfidenceLevel.LOW:
            recommendations.append("Consider professional consultation for complex situations")
        
        # Conflict-based recommendations
        if conflicts:
            recommendations.append("Review all applicable rules carefully due to potential conflicts")
            
            for conflict in conflicts:
                if conflict.conflict_type == RuleConflictType.CONTRADICTORY:
                    recommendations.append("Apply conservative interpretation due to contradictory rules")
        
        # Documentation recommendations
        substantiation_needed = any(
            "substantiation" in interp.rule_text.lower() 
            for interp in interpretations
        )
        if substantiation_needed:
            recommendations.append("Ensure proper documentation and substantiation")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Apply rule as interpreted, maintaining proper documentation")
        
        return recommendations
    
    def _create_audit_trail(
        self,
        query: AgenticQuery,
        interpretations: List[RuleInterpretation],
        conflicts: List[RuleConflict]
    ) -> List[str]:
        """Create comprehensive audit trail for compliance purposes."""
        
        trail = []
        
        # Query information
        trail.append(f"Original query: {query.original_query}")
        trail.append(f"Enhanced query: {query.enhanced_query}")
        trail.append(f"Search strategy: {query.search_strategy}")
        
        # Rule sources
        trail.append(f"Retrieved {len(interpretations)} rule interpretations")
        for i, interp in enumerate(interpretations):
            for citation in interp.citations:
                trail.append(
                    f"Rule {i+1}: {citation.document_title} "
                    f"(confidence: {citation.confidence_score:.1%})"
                )
        
        # Conflicts
        if conflicts:
            trail.append(f"Detected {len(conflicts)} rule conflicts")
            for i, conflict in enumerate(conflicts):
                trail.append(f"Conflict {i+1}: {conflict.conflict_type} - {conflict.resolution_strategy}")
        
        # Timestamp
        trail.append(f"Analysis completed: {datetime.now().isoformat()}")
        
        return trail
    
    def _create_fallback_result(self, query: str, error: str) -> ReasoningResult:
        """Create a conservative fallback result when reasoning fails."""
        
        fallback_query = AgenticQuery(
            original_query=query,
            enhanced_query=query,
            context_factors=[],
            domain_keywords=[],
            search_strategy=SearchStrategy.BALANCED
        )
        
        return ReasoningResult(
            query=fallback_query,
            rule_interpretations=[],
            conflicts=[],
            synthesized_guidance="Unable to analyze rules due to system error. Recommend consulting with a tax professional.",
            overall_confidence=ConfidenceLevel.VERY_LOW,
            recommendations=["Consult with qualified tax professional", "Do not proceed without professional guidance"],
            audit_trail=[f"System error: {error}", f"Fallback result generated: {datetime.now().isoformat()}"]
        )


# Convenience functions for common use cases
async def analyze_tax_rule(query: str, context: Dict[str, any] = None) -> ReasoningResult:
    """Analyze a tax rule query with full agentic reasoning."""
    engine = AgenticReasoningEngine()
    return await engine.reason_about_query(query, context)


async def get_expense_deduction_guidance(
    expense_description: str,
    amount: float,
    business_context: str = None
) -> ReasoningResult:
    """Get guidance on expense deductibility."""
    context = {
        "transaction_amount": amount,
        "business_type": business_context or "general business"
    }
    
    query = f"Is {expense_description} deductible as a business expense?"
    
    engine = AgenticReasoningEngine()
    return await engine.reason_about_query(query, context)


# Test function
async def test_agentic_reasoning():
    """Test the agentic reasoning engine."""
    engine = AgenticReasoningEngine()
    
    # Test query with context
    query = "business meal deduction rules"
    context = {
        "transaction_amount": 150.00,
        "business_type": "consulting",
        "transaction_date": "2024-01-15"
    }
    
    result = await engine.reason_about_query(query, context)
    
    print(f"Query: {result.query.original_query}")
    print(f"Enhanced: {result.query.enhanced_query}")
    print(f"Confidence: {result.overall_confidence}")
    print(f"Guidance: {result.synthesized_guidance}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Conflicts: {len(result.conflicts)}")
    print(f"Interpretations: {len(result.rule_interpretations)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agentic_reasoning())