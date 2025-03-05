"""
OpenAI client for analyzing papers.
Uses OpenAI o1 to plan implementation.
"""
import os
import sys
import json
import logging
import asyncio
import base64
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pypdf

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import config

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Pydantic models for structured output
class InnovationAssessment(BaseModel):
    summary: str = Field(description="Brief description of the innovation")
    novelty: str = Field(description="Analysis of what makes this research novel or incremental")
    comparison: str = Field(description="Comparison to existing state-of-the-art approaches")
    impact: str = Field(description="Assessment of potential impact on the field")
    layman_explanation: str = Field(description="Simple explanation of the research in first principles for non-experts")
    innovation_score: float = Field(description="Innovation score from 1.0-10.0 with decimal precision")

class CommercialOpportunity(BaseModel):
    applications: List[str] = Field(description="List of potential applications and use cases")
    target_markets: List[str] = Field(description="List of target industries or markets")
    competitive_advantage: str = Field(description="Evaluation of advantage over existing solutions")
    time_to_market: str = Field(description="Estimate in months/years")
    barriers: List[str] = Field(description="List of barriers to commercialization")
    market_potential_score: float = Field(description="Market potential score from 1.0-10.0 with decimal precision")

class ImplementationFeasibility(BaseModel):
    complexity: str = Field(description="Assessment of technical complexity")
    required_expertise: List[str] = Field(description="List of required expertise areas")
    technical_challenges: List[str] = Field(description="List of key technical challenges")
    alternative_approaches: List[str] = Field(description="List of alternative implementation approaches")
    simplification_opportunities: List[str] = Field(description="Areas where implementation can be simplified")
    technical_feasibility_score: float = Field(description="Technical feasibility score from 1.0-10.0 with decimal precision")

class ImplementationPlan(BaseModel):
    steps: List[str] = Field(description="Step-by-step implementation plan")
    computational_resources: List[str] = Field(description="List of required computational resources")
    hardware_requirements: List[str] = Field(description="List of special hardware requirements")
    software_requirements: List[str] = Field(description="List of software requirements")
    timeline: str = Field(description="Estimated timeline for implementation")
    tech_stack: List[str] = Field(description="Recommended technologies and tools")

class RiskAssessment(BaseModel):
    technical_risks: List[str] = Field(description="List of technical risks")
    market_risks: List[str] = Field(description="List of market risks")
    regulatory_concerns: List[str] = Field(description="List of regulatory or ethical considerations")
    mitigations: List[str] = Field(description="List of potential mitigations")

class Scores(BaseModel):
    innovation_score: float = Field(description="Innovation score from 1.0-10.0 with decimal precision")
    technical_feasibility_score: float = Field(description="Technical feasibility score from 1.0-10.0 with decimal precision")
    market_potential_score: float = Field(description="Market potential score from 1.0-10.0 with decimal precision")
    impact_score: float = Field(description="Impact score from 1.0-10.0 with decimal precision")
    opportunity_score: float = Field(description="Overall opportunity score from 1.0-10.0 with decimal precision")

class PaperAssessment(BaseModel):
    innovation_assessment: InnovationAssessment
    commercial_opportunity: CommercialOpportunity
    implementation_feasibility: ImplementationFeasibility
    implementation_plan: ImplementationPlan
    risk_assessment: RiskAssessment
    scores: Scores
    summary: str = Field(description="Overall assessment of the commercial potential")


class OpenAIClient:
    """
    Client for interacting with OpenAI API to analyze papers.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model or os.getenv("OPENAI_MODEL", "o1-2024-12-17")
        
        # Initialize the client
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
    
    async def plan_implementation(self, paper_metadata: Dict[str, Any], pdf_path: str, claude_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan implementation for a paper using OpenAI API with structured output.
        
        Args:
            paper_metadata: Paper metadata
            pdf_path: Path to PDF file
            claude_analysis: Optional Claude analysis results
            
        Returns:
            Implementation plan
        """
        logger.info(f"Planning implementation for paper: {paper_metadata.get('paper_id', 'unknown')}")
        
        # Extract text from PDF
        pdf_text = self._extract_text_from_pdf(Path(pdf_path))
        # Limit text length to avoid token issues
        if len(pdf_text) > 30000:
            pdf_text = pdf_text[:30000] + "... [truncated for token limit]"
        
        # Prepare the system prompt
        system_prompt = """You are an expert AI researcher and engineer specializing in assessing research papers for their commercial potential and implementation feasibility. Your task is to analyze the provided paper and create a detailed structured assessment.

Be truly critical in your assessment - scores should vary significantly between different papers. Be honest when a paper has limited commercial potential or significant implementation challenges. Not every paper deserves high scores.

When providing scores, use precise decimal values (e.g., 7.3, 8.6, 5.2) rather than just integers to better capture nuanced differences between papers. This precision helps distinguish between papers with similar qualities.

For each paper, provide a "layman's explanation" that breaks down the research into simple concepts using first principles. This should help non-experts understand what the research is about, why it matters, and how it might be used in the real world. Avoid jargon and use analogies when helpful.

You will provide your analysis in a structured format with scores for innovation, technical feasibility, market potential, impact, and overall opportunity (all on a scale of 1.0-10.0 with decimal precision).
"""
        
        # Prepare the user content
        user_content = f"""Paper Title: {paper_metadata.get('title', '(Unknown title)')}
Paper ID: {paper_metadata.get('paper_id', '(Unknown ID)')}
Authors: {paper_metadata.get('authors', '(Unknown authors)')}
Categories: {paper_metadata.get('categories', '(Unknown categories)')}

Abstract:
{paper_metadata.get('abstract', '(No abstract available)')}

PDF Content:
{pdf_text}

Please provide a comprehensive assessment, evaluating this paper's commercial potential, technical feasibility, and providing a detailed implementation roadmap.
"""
        
        try:
            # Create a synchronous request
            completion = self.sync_client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=PaperAssessment,
                reasoning_effort="high" if "o1" in self.model else None
            )
            
            # Get the structured output
            assessment = completion.choices[0].message.parsed
            
            # Format the assessment as readable text
            formatted_plan = self._format_implementation_plan(assessment)
            
            # Process the response
            implementation_plan = {
                "paper_id": paper_metadata.get("paper_id"),
                "plan": formatted_plan,
                "steps": assessment.implementation_plan.steps,
                "time_estimate": assessment.implementation_plan.timeline,
                "resources": (
                    assessment.implementation_plan.computational_resources + 
                    assessment.implementation_plan.hardware_requirements + 
                    assessment.implementation_plan.software_requirements
                ),
                "challenges": assessment.risk_assessment.technical_risks,
                "raw_plan": json.dumps(assessment.dict(), indent=2),
                # Extract scores directly from the structured output
                "innovation_score": assessment.scores.innovation_score,
                "technical_feasibility_score": assessment.scores.technical_feasibility_score,
                "market_potential_score": assessment.scores.market_potential_score,
                "impact_score": assessment.scores.impact_score,
                "opportunity_score": assessment.scores.opportunity_score,
                "target_markets": assessment.commercial_opportunity.target_markets,
                "time_to_market": assessment.commercial_opportunity.time_to_market,
                "layman_explanation": assessment.innovation_assessment.layman_explanation,
                "analyzed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Implementation planning complete for paper: {paper_metadata.get('paper_id', 'unknown')}")
            return implementation_plan
        
        except Exception as e:
            logger.error(f"Failed to plan implementation: {e}", exc_info=True)
            
            # Fallback to traditional approach
            try:
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                }
                
                if "o1" in self.model:
                    api_params["reasoning_effort"] = "high"
                else:
                    api_params["temperature"] = 0.3
                
                # Call the OpenAI API
                response = await self.client.chat.completions.create(**api_params)
                
                # Extract the content
                content = response.choices[0].message.content
                
                # Process the response with fallback
                implementation_plan = {
                    "paper_id": paper_metadata.get("paper_id"),
                    "plan": content,
                    "steps": self._extract_steps(content),
                    "time_estimate": self._extract_time_estimate(content),
                    "resources": self._extract_resources(content),
                    "challenges": self._extract_challenges(content),
                    "raw_plan": content
                }
                
                logger.info(f"Implementation planning complete with fallback for paper: {paper_metadata.get('paper_id', 'unknown')}")
                return implementation_plan
            except Exception as inner_e:
                logger.error(f"Fallback also failed: {inner_e}", exc_info=True)
                raise
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            text = ""
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text[:50000]  # Limit to first 50k characters to avoid token limits
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}", exc_info=True)
            return "Error extracting text from PDF."

    def _format_implementation_plan(self, assessment: PaperAssessment) -> str:
        """Format the structured implementation plan as a readable text."""
        sections = []
        
        # Innovation Assessment
        ia = assessment.innovation_assessment
        sections.append(f"# INNOVATION ASSESSMENT (Score: {ia.innovation_score}/10)\n")
        sections.append(f"{ia.summary}\n")
        sections.append(f"- **Novelty**: {ia.novelty}")
        sections.append(f"- **Comparison to SOTA**: {ia.comparison}")
        sections.append(f"- **Potential Impact**: {ia.impact}\n")
        
        # Commercial Opportunity
        co = assessment.commercial_opportunity
        sections.append(f"# COMMERCIAL OPPORTUNITY (Score: {co.market_potential_score}/10)\n")
            
        # Applications
        sections.append("## Potential Applications")
        for app in co.applications:
            sections.append(f"- {app}")
            
        # Target Markets
        sections.append("\n## Target Markets")
        for market in co.target_markets:
            sections.append(f"- {market}")
            
        sections.append(f"\n- **Competitive Advantage**: {co.competitive_advantage}")
        sections.append(f"- **Time to Market**: {co.time_to_market}")
            
        # Barriers
        sections.append("\n## Commercialization Barriers")
        for barrier in co.barriers:
            sections.append(f"- {barrier}")
        
        # Implementation Feasibility
        impl = assessment.implementation_feasibility
        sections.append(f"\n# IMPLEMENTATION FEASIBILITY (Score: {impl.technical_feasibility_score}/10)\n")
        sections.append(f"**Complexity**: {impl.complexity}\n")
            
        # Required Expertise
        sections.append("## Required Expertise")
        for exp in impl.required_expertise:
            sections.append(f"- {exp}")
            
        # Technical Challenges
        sections.append("\n## Technical Challenges")
        for challenge in impl.technical_challenges:
            sections.append(f"- {challenge}")
            
        # Simplification Opportunities
        sections.append("\n## Simplification Opportunities")
        for simp in impl.simplification_opportunities:
            sections.append(f"- {simp}")
        
        # Implementation Plan
        plan = assessment.implementation_plan
        sections.append("\n# IMPLEMENTATION PLAN\n")
            
        # Steps
        sections.append("## Implementation Steps")
        for i, step in enumerate(plan.steps, 1):
            sections.append(f"{i}. {step}")
            
        # Resources
        sections.append("\n## Required Resources")
            
        # Computational
        sections.append("\n### Computational Resources")
        for res in plan.computational_resources:
            sections.append(f"- {res}")
            
        # Hardware
        sections.append("\n### Hardware Requirements")
        for hw in plan.hardware_requirements:
            sections.append(f"- {hw}")
            
        # Software
        sections.append("\n### Software Requirements")
        for sw in plan.software_requirements:
            sections.append(f"- {sw}")
            
        # Timeline and Tech Stack
        sections.append(f"\n**Timeline**: {plan.timeline}")
            
        # Tech Stack
        sections.append("\n### Recommended Tech Stack")
        for tech in plan.tech_stack:
            sections.append(f"- {tech}")
        
        # Risk Assessment
        risk = assessment.risk_assessment
        sections.append("\n# RISK ASSESSMENT\n")
            
        # Technical Risks
        sections.append("## Technical Risks")
        for tr in risk.technical_risks:
            sections.append(f"- {tr}")
            
        # Market Risks
        sections.append("\n## Market Risks")
        for mr in risk.market_risks:
            sections.append(f"- {mr}")
            
        # Regulatory Concerns
        sections.append("\n## Regulatory & Ethical Concerns")
        for rc in risk.regulatory_concerns:
            sections.append(f"- {rc}")
            
        # Mitigations
        sections.append("\n## Risk Mitigations")
        for mit in risk.mitigations:
            sections.append(f"- {mit}")
        
        # Overall Score
        scores = assessment.scores
        sections.append("\n# OVERALL ASSESSMENT\n")
        sections.append(f"- **Innovation Score**: {scores.innovation_score}/10")
        sections.append(f"- **Technical Feasibility**: {scores.technical_feasibility_score}/10")
        sections.append(f"- **Market Potential**: {scores.market_potential_score}/10")
        sections.append(f"- **Impact Score**: {scores.impact_score}/10")
        sections.append(f"- **Overall Opportunity Score**: {scores.opportunity_score}/10")
        
        # Summary
        sections.append(f"\n## Executive Summary\n{assessment.summary}")
        
        # Generated timestamp
        sections.append(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(sections)
        
    # Keep the extraction methods for fallback
    def _extract_steps(self, text: str) -> List[str]:
        """Extract implementation steps from response text."""
        steps = []
        in_steps_section = False
        
        for line in text.split("\n"):
            line = line.strip()
            
            # Detect steps section
            if "implementation steps" in line.lower() or "implementation steps:" in line.lower():
                in_steps_section = True
                continue
            
            # Exit steps section when we hit another major section
            if in_steps_section and line and line[0].isdigit() and ":" in line and not (line[0:2].isdigit() and line[2] == "."):
                in_steps_section = False
            
            # Extract steps
            if in_steps_section and line and (
                (line[0].isdigit() and "." in line[:3]) or  # numbered lists like "1. Step one"
                (line.startswith("Step") and ":" in line)    # or "Step 1: Do something"
            ):
                # Clean up the step text
                step = line.split(".", 1)[-1].split(":", 1)[-1].strip()
                if step:
                    steps.append(step)
        
        # If we couldn't find specific steps, try to find numbered items
        if not steps:
            for line in text.split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and ". " in line[:4]:
                    step = line.split(". ", 1)[-1].strip()
                    if step:
                        steps.append(step)
        
        return steps
    
    def _extract_time_estimate(self, text: str) -> str:
        """Extract time estimate from response text."""
        # Look for total time estimate
        for line in text.split("\n"):
            line = line.lower().strip()
            if "total time" in line or "estimated total time" in line or "total implementation time" in line:
                return line
            
        # Look for time-related sentences
        for line in text.split("\n"):
            if "week" in line.lower() and "time" in line.lower():
                return line.strip()
            if "day" in line.lower() and "time" in line.lower():
                return line.strip()
            if "hour" in line.lower() and "time" in line.lower():
                return line.strip()
        
        return "Time estimate not found"
    
    def _extract_resources(self, text: str) -> List[str]:
        """Extract resources from response text."""
        resources = []
        in_resources_section = False
        
        for line in text.split("\n"):
            line = line.strip()
            
            # Detect resources section
            if "resources required" in line.lower() or "resources needed" in line.lower() or "required resources" in line.lower():
                in_resources_section = True
                continue
            
            # Exit resources section when we hit another major section
            if in_resources_section and line and line.endswith(":") and len(line) < 30:
                in_resources_section = False
            
            # Extract resources
            if in_resources_section and line and (line.startswith("-") or line.startswith("*")):
                resource = line[1:].strip()
                if resource:
                    resources.append(resource)
        
        return resources
    
    def _extract_challenges(self, text: str) -> List[str]:
        """Extract challenges from response text."""
        challenges = []
        in_challenges_section = False
        
        for line in text.split("\n"):
            line = line.strip()
            
            # Detect challenges section
            if "challenges" in line.lower() and line.endswith(":"):
                in_challenges_section = True
                continue
            
            # Exit challenges section when we hit another major section
            if in_challenges_section and line and line.endswith(":") and len(line) < 30:
                in_challenges_section = False
            
            # Extract challenges
            if in_challenges_section and line and (line.startswith("-") or line.startswith("*")):
                challenge = line[1:].strip()
                if challenge:
                    challenges.append(challenge)
        
        return challenges 