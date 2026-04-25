"""
AI Maturity Framework
=====================
Conceptual framework for measuring AI adoption maturity.

Based on:
- Gartner AI Maturity Model (5 levels)
- MIT Sloan Digital Maturity Framework
- Deloitte State of AI in the Enterprise (2023)
- Harvard Business Review: "Building the AI-Powered Organization" (2019)

The framework defines 5 dimensions, each containing 5 sub-dimensions
measured via Likert scales (1-5). Total: 25 questions.

Dimension weights are based on weighted aggregation of empirical
findings from the cited literature.
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class MaturityLevel(str, Enum):
    """Five levels of AI maturity (Gartner-inspired)."""
    INITIAL = "Initial"          # 0.0 - 1.5
    EXPLORING = "Exploring"      # 1.5 - 2.5
    DEVELOPING = "Developing"    # 2.5 - 3.5
    SCALING = "Scaling"          # 3.5 - 4.3
    OPTIMIZING = "Optimizing"    # 4.3 - 5.0


@dataclass
class Question:
    """A single Likert-scale question."""
    id: str
    text: str
    helper: str  # tooltip explanation
    scale_labels: Dict[int, str]  # 1-5 with labels


@dataclass
class SubDimension:
    """A sub-dimension containing one question."""
    id: str
    name: str
    description: str
    question: Question


@dataclass
class Dimension:
    """A main dimension of AI maturity."""
    id: str
    name: str
    description: str
    weight: float  # contribution to overall score
    sub_dimensions: List[SubDimension]


# ---------------------------------------------------------------------------
# DIMENSION 1: DATA (weight: 0.25)
# ---------------------------------------------------------------------------
DIMENSION_DATA = Dimension(
    id="data",
    name="Data",
    description="Quality, governance, accessibility and integration of data assets.",
    weight=0.25,
    sub_dimensions=[
        SubDimension(
            id="data_quality",
            name="Data Quality",
            description="Cleanliness, completeness and consistency of data.",
            question=Question(
                id="d1_q1",
                text="How would you describe the quality of data your company works with?",
                helper="Quality refers to how clean, consistent and reliable your data is for analysis.",
                scale_labels={
                    1: "Inconsistent and difficult to use",
                    2: "Usable but requires frequent manual cleaning",
                    3: "Reasonably reliable for basic analysis",
                    4: "High quality with established validation processes",
                    5: "Excellent, governed, with automated pipelines",
                },
            ),
        ),
        SubDimension(
            id="data_governance",
            name="Data Governance",
            description="Policies, ownership and compliance frameworks.",
            question=Question(
                id="d1_q2",
                text="How mature is your data governance (policies, ownership, GDPR/regulation)?",
                helper="Governance includes who owns data, how it's protected and regulatory compliance.",
                scale_labels={
                    1: "No formal governance, ad-hoc decisions",
                    2: "Basic policies but inconsistently applied",
                    3: "Documented policies, partial compliance",
                    4: "Strong governance with assigned data owners",
                    5: "Best-in-class governance with audit trails and proactive compliance",
                },
            ),
        ),
        SubDimension(
            id="data_accessibility",
            name="Data Accessibility",
            description="Ease of access for those who need data.",
            question=Question(
                id="d1_q3",
                text="How easily can teams across your company access the data they need?",
                helper="Considers data silos, APIs, self-service tools and request friction.",
                scale_labels={
                    1: "Data is heavily siloed, access is slow",
                    2: "Some access, but requires manual requests",
                    3: "Centralized for some teams, others struggle",
                    4: "Self-service access for most use cases",
                    5: "Democratized access with strong access controls",
                },
            ),
        ),
        SubDimension(
            id="data_integration",
            name="Data Integration",
            description="Integration across systems (ERP, CRM, data lakes).",
            question=Question(
                id="d1_q4",
                text="How well are your different data sources integrated?",
                helper="Integration means data flows between systems without manual reconciliation.",
                scale_labels={
                    1: "Systems are isolated, manual exports needed",
                    2: "Some integrations, mostly batch and brittle",
                    3: "Core systems integrated, others not",
                    4: "Most systems integrated via APIs or data platform",
                    5: "Unified data platform with real-time integration",
                },
            ),
        ),
        SubDimension(
            id="data_volume",
            name="Data Volume & Variety",
            description="Volume and variety of data (structured, unstructured).",
            question=Question(
                id="d1_q5",
                text="How would you assess the volume and variety of data you collect?",
                helper="Variety includes structured (DB), semi-structured (logs) and unstructured (text, images).",
                scale_labels={
                    1: "Minimal data collected, mostly structured",
                    2: "Moderate volume, limited variety",
                    3: "Substantial volume, some unstructured data",
                    4: "Large volume with diverse data types",
                    5: "Massive multi-modal datasets continuously growing",
                },
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# DIMENSION 2: TALENT & CULTURE (weight: 0.20)
# ---------------------------------------------------------------------------
DIMENSION_TALENT = Dimension(
    id="talent",
    name="Talent & Culture",
    description="Internal AI/ML skills, data-driven culture and change readiness.",
    weight=0.20,
    sub_dimensions=[
        SubDimension(
            id="talent_skills",
            name="AI/ML Skills",
            description="Internal expertise in data science and ML.",
            question=Question(
                id="d2_q1",
                text="What level of AI/ML expertise exists internally?",
                helper="Includes data scientists, ML engineers and analytics talent.",
                scale_labels={
                    1: "No AI/ML talent, fully reliant on vendors",
                    2: "1-2 generalists with basic skills",
                    3: "Small team with mixed expertise",
                    4: "Established AI/ML team with senior talent",
                    5: "Mature team including research-level expertise",
                },
            ),
        ),
        SubDimension(
            id="talent_culture",
            name="Data-Driven Culture",
            description="How decisions are made (gut vs. data).",
            question=Question(
                id="d2_q2",
                text="How data-driven is decision-making at your company?",
                helper="From gut-feeling to dashboards to predictive analytics in every meeting.",
                scale_labels={
                    1: "Decisions are mostly intuition-based",
                    2: "Data referenced occasionally for big decisions",
                    3: "Dashboards consulted for routine decisions",
                    4: "Data is central to most strategic decisions",
                    5: "Data-driven culture embedded at all levels",
                },
            ),
        ),
        SubDimension(
            id="talent_upskilling",
            name="Active Upskilling",
            description="Investment in training and AI literacy.",
            question=Question(
                id="d2_q3",
                text="How much does your company invest in AI training and upskilling?",
                helper="Includes courses, certifications, internal academies and learning hours.",
                scale_labels={
                    1: "No formal AI training programs",
                    2: "Occasional ad-hoc training",
                    3: "Some structured programs for select teams",
                    4: "Company-wide AI literacy programs",
                    5: "Continuous learning culture with measurable AI fluency",
                },
            ),
        ),
        SubDimension(
            id="talent_collaboration",
            name="Cross-Functional Collaboration",
            description="Collaboration between business and technical teams.",
            question=Question(
                id="d2_q4",
                text="How well do business and technical teams collaborate on AI projects?",
                helper="Indicates whether AI projects are co-owned by business stakeholders.",
                scale_labels={
                    1: "Teams operate in silos, little collaboration",
                    2: "Occasional collaboration on specific projects",
                    3: "Regular touchpoints, but ownership unclear",
                    4: "Strong collaboration with shared KPIs",
                    5: "Fully embedded cross-functional product teams",
                },
            ),
        ),
        SubDimension(
            id="talent_change",
            name="Change Readiness",
            description="Openness to adopting new tools and processes.",
            question=Question(
                id="d2_q5",
                text="How open is your organization to adopting new AI tools and processes?",
                helper="Considers cultural resistance, change management capabilities.",
                scale_labels={
                    1: "Strong resistance to change",
                    2: "Slow adoption with significant pushback",
                    3: "Mixed reception depending on the team",
                    4: "Generally open with effective change management",
                    5: "Eager adoption, change is part of identity",
                },
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# DIMENSION 3: TECHNOLOGY (weight: 0.20)
# ---------------------------------------------------------------------------
DIMENSION_TECHNOLOGY = Dimension(
    id="technology",
    name="Technology",
    description="Infrastructure, MLOps, cloud and analytics tooling.",
    weight=0.20,
    sub_dimensions=[
        SubDimension(
            id="tech_cloud",
            name="Cloud Infrastructure",
            description="Adoption of cloud and modern infra.",
            question=Question(
                id="d3_q1",
                text="How modern is your technology infrastructure (cloud, APIs, microservices)?",
                helper="Modern infra is needed to deploy and scale AI workloads efficiently.",
                scale_labels={
                    1: "Mostly on-premises legacy systems",
                    2: "Hybrid with significant legacy",
                    3: "Cloud-first for new projects",
                    4: "Predominantly cloud-native",
                    5: "Fully cloud-native, multi-cloud or modern stack",
                },
            ),
        ),
        SubDimension(
            id="tech_mlops",
            name="MLOps Maturity",
            description="Operational practices for ML.",
            question=Question(
                id="d3_q2",
                text="How mature are your MLOps practices (model deployment, monitoring, CI/CD)?",
                helper="MLOps determines whether models stay accurate and useful in production.",
                scale_labels={
                    1: "No MLOps; models exist only in notebooks",
                    2: "Manual deployment, no monitoring",
                    3: "Some automation, basic monitoring",
                    4: "Established MLOps with CI/CD and monitoring",
                    5: "Best-in-class MLOps with automated retraining",
                },
            ),
        ),
        SubDimension(
            id="tech_analytics",
            name="Analytics Tools",
            description="BI and analytics platforms in use.",
            question=Question(
                id="d3_q3",
                text="What analytics and BI tools are used across the company?",
                helper="From spreadsheets to enterprise BI platforms with self-service.",
                scale_labels={
                    1: "Mostly spreadsheets and ad-hoc analyses",
                    2: "Basic BI used by a few teams",
                    3: "Centralized BI platform for key metrics",
                    4: "Self-service BI widely adopted",
                    5: "Advanced analytics integrated into operations",
                },
            ),
        ),
        SubDimension(
            id="tech_debt",
            name="Technical Debt",
            description="Accumulated debt limiting agility (inverse).",
            question=Question(
                id="d3_q4",
                text="How would you assess the level of technical debt limiting your AI ambitions?",
                helper="High debt slows down everything; low debt enables agility.",
                scale_labels={
                    1: "Crippling technical debt, blocks most initiatives",
                    2: "Significant debt, slows down delivery",
                    3: "Moderate debt, manageable but visible",
                    4: "Limited debt, actively managed",
                    5: "Minimal debt, modern stack across the board",
                },
            ),
        ),
        SubDimension(
            id="tech_security",
            name="Security & Reliability",
            description="Security practices for AI systems.",
            question=Question(
                id="d3_q5",
                text="How robust are your security and reliability practices for AI systems?",
                helper="Includes model security, data protection, uptime, fallbacks.",
                scale_labels={
                    1: "Security is reactive and inconsistent",
                    2: "Basic security but no AI-specific practices",
                    3: "General security strong, AI-specific emerging",
                    4: "Robust security including AI threat modeling",
                    5: "Industry-leading security with continuous testing",
                },
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# DIMENSION 4: STRATEGY (weight: 0.20)
# ---------------------------------------------------------------------------
DIMENSION_STRATEGY = Dimension(
    id="strategy",
    name="Strategy",
    description="Executive vision, budget, KPIs and use case prioritization.",
    weight=0.20,
    sub_dimensions=[
        SubDimension(
            id="strategy_vision",
            name="C-Suite AI Vision",
            description="Clarity of leadership's AI vision.",
            question=Question(
                id="d4_q1",
                text="How clear and committed is your C-suite to an AI strategy?",
                helper="Top-down commitment is the #1 predictor of AI success (HBR, 2019).",
                scale_labels={
                    1: "No clear AI vision from leadership",
                    2: "AI mentioned but not prioritized",
                    3: "AI is a stated priority, plans being developed",
                    4: "Clear AI strategy with executive sponsorship",
                    5: "AI is core to corporate strategy with CEO ownership",
                },
            ),
        ),
        SubDimension(
            id="strategy_budget",
            name="Dedicated Budget",
            description="Financial commitment to AI initiatives.",
            question=Question(
                id="d4_q2",
                text="How significant is the budget dedicated to AI initiatives?",
                helper="Sustained investment over multiple years is essential for ROI.",
                scale_labels={
                    1: "No dedicated AI budget",
                    2: "Small experimental budget",
                    3: "Moderate, project-by-project funding",
                    4: "Substantial multi-year committed budget",
                    5: "AI budget treated as strategic infrastructure",
                },
            ),
        ),
        SubDimension(
            id="strategy_use_cases",
            name="Use Case Pipeline",
            description="Identification and prioritization of AI use cases.",
            question=Question(
                id="d4_q3",
                text="How developed is your AI use case pipeline?",
                helper="Mature companies maintain prioritized backlogs of AI opportunities.",
                scale_labels={
                    1: "No identified use cases",
                    2: "Few ideas, no prioritization",
                    3: "Several use cases, some prioritized",
                    4: "Well-developed pipeline with ROI analysis",
                    5: "Continuous discovery process with measurable impact",
                },
            ),
        ),
        SubDimension(
            id="strategy_kpis",
            name="AI KPIs",
            description="Defined metrics to measure AI success.",
            question=Question(
                id="d4_q4",
                text="Are there defined KPIs to measure the success of AI initiatives?",
                helper="Without KPIs, AI projects drift and ROI cannot be demonstrated.",
                scale_labels={
                    1: "No KPIs defined for AI",
                    2: "KPIs exist but rarely tracked",
                    3: "Basic KPIs tracked for major projects",
                    4: "Comprehensive KPIs tied to business outcomes",
                    5: "Continuously refined KPIs driving decisions",
                },
            ),
        ),
        SubDimension(
            id="strategy_ethics",
            name="AI Ethics & Risk",
            description="Frameworks for responsible AI.",
            question=Question(
                id="d4_q5",
                text="How developed are your AI ethics and risk management practices?",
                helper="Includes bias testing, fairness, regulatory readiness (e.g., EU AI Act).",
                scale_labels={
                    1: "No ethics framework or risk processes",
                    2: "Aware of risks but no formal practices",
                    3: "Emerging guidelines, applied to some projects",
                    4: "Established framework with regular reviews",
                    5: "Industry-leading responsible AI practices",
                },
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# DIMENSION 5: PROCESSES (weight: 0.15)
# ---------------------------------------------------------------------------
DIMENSION_PROCESSES = Dimension(
    id="processes",
    name="Processes",
    description="Process automation, documentation and integration of AI in operations.",
    weight=0.15,
    sub_dimensions=[
        SubDimension(
            id="proc_automation",
            name="Current Automation",
            description="Existing level of process automation.",
            question=Question(
                id="d5_q1",
                text="What is the current level of automation in your core processes?",
                helper="Automation is the foundation on which AI augmentation builds.",
                scale_labels={
                    1: "Mostly manual processes",
                    2: "Some scripts and basic automation",
                    3: "RPA / workflow automation in key areas",
                    4: "Extensive automation across the organization",
                    5: "Intelligent automation pervasive",
                },
            ),
        ),
        SubDimension(
            id="proc_documentation",
            name="Process Documentation",
            description="Documented, standardized processes.",
            question=Question(
                id="d5_q2",
                text="How well documented and standardized are your business processes?",
                helper="Documented processes are easier to optimize and automate.",
                scale_labels={
                    1: "Mostly tribal knowledge",
                    2: "Some documentation, often outdated",
                    3: "Core processes documented",
                    4: "Comprehensive documentation, regularly updated",
                    5: "Living documentation with process mining",
                },
            ),
        ),
        SubDimension(
            id="proc_continuous",
            name="Continuous Improvement",
            description="Cycles of optimization and feedback.",
            question=Question(
                id="d5_q3",
                text="How embedded is continuous improvement in your operations?",
                helper="Includes Lean, Six Sigma, agile retrospectives, KPI review cycles.",
                scale_labels={
                    1: "No formal improvement process",
                    2: "Occasional improvement initiatives",
                    3: "Regular review cycles in some areas",
                    4: "Continuous improvement embedded in most teams",
                    5: "Improvement is part of organizational DNA",
                },
            ),
        ),
        SubDimension(
            id="proc_ai_integration",
            name="AI in Operations",
            description="Degree of AI integrated in daily ops.",
            question=Question(
                id="d5_q4",
                text="To what extent is AI integrated into daily operational processes?",
                helper="From experimental pilots to AI-augmented decisions in real time.",
                scale_labels={
                    1: "AI not integrated in any process",
                    2: "AI used in isolated pilots",
                    3: "AI integrated in select operational processes",
                    4: "AI integrated across multiple operations",
                    5: "AI is the default for most operational decisions",
                },
            ),
        ),
        SubDimension(
            id="proc_feedback",
            name="Feedback Loops",
            description="Mechanisms to capture outcomes and refine.",
            question=Question(
                id="d5_q5",
                text="How effective are your feedback loops for AI-driven decisions?",
                helper="Feedback loops are how AI systems learn and improve over time.",
                scale_labels={
                    1: "No feedback captured from AI outputs",
                    2: "Manual feedback for some use cases",
                    3: "Feedback captured but not systematically used",
                    4: "Structured feedback feeding model retraining",
                    5: "Real-time feedback loops driving continuous learning",
                },
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# FRAMEWORK ASSEMBLY
# ---------------------------------------------------------------------------
FRAMEWORK: List[Dimension] = [
    DIMENSION_DATA,
    DIMENSION_TALENT,
    DIMENSION_TECHNOLOGY,
    DIMENSION_STRATEGY,
    DIMENSION_PROCESSES,
]


# Validate weights sum to 1.0
_total_weight = sum(d.weight for d in FRAMEWORK)
assert abs(_total_weight - 1.0) < 1e-9, f"Weights sum to {_total_weight}, must be 1.0"


def get_all_questions() -> List[Question]:
    """Return a flat list of all questions in the framework."""
    questions = []
    for dim in FRAMEWORK:
        for sub in dim.sub_dimensions:
            questions.append(sub.question)
    return questions


def get_question_to_dimension_map() -> Dict[str, str]:
    """Return mapping of question_id -> dimension_id."""
    mapping = {}
    for dim in FRAMEWORK:
        for sub in dim.sub_dimensions:
            mapping[sub.question.id] = dim.id
    return mapping


def classify_maturity(score: float) -> MaturityLevel:
    """Classify a 1-5 score into a maturity level."""
    if score < 1.5:
        return MaturityLevel.INITIAL
    elif score < 2.5:
        return MaturityLevel.EXPLORING
    elif score < 3.5:
        return MaturityLevel.DEVELOPING
    elif score < 4.3:
        return MaturityLevel.SCALING
    else:
        return MaturityLevel.OPTIMIZING


def maturity_description(level: MaturityLevel) -> str:
    """Human-readable description of each maturity level."""
    descriptions = {
        MaturityLevel.INITIAL: (
            "AI is largely absent from the organization. Decisions rely on intuition, "
            "data is fragmented, and there is no clear strategy for adoption."
        ),
        MaturityLevel.EXPLORING: (
            "The organization is exploring AI through small pilots. Awareness is growing "
            "but capabilities are limited and impact is anecdotal."
        ),
        MaturityLevel.DEVELOPING: (
            "AI is gaining traction with multiple initiatives. Foundations (data, talent, "
            "infrastructure) are forming but inconsistent across the organization."
        ),
        MaturityLevel.SCALING: (
            "AI is producing measurable business value. The organization scales successful "
            "use cases and invests systematically in capabilities."
        ),
        MaturityLevel.OPTIMIZING: (
            "AI is core to the organization's competitive advantage. Practices are mature, "
            "feedback loops are tight, and innovation is continuous."
        ),
    }
    return descriptions[level]
