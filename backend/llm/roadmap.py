"""
Roadmap Generator
=================
Generates a personalized AI adoption roadmap based on the assessment.

Architecture:
- Rule-based skeleton (deterministic, always works)
- LLM augmentation layer (optional; falls back gracefully if no API key)
- Knowledge base of initiatives by maturity level + sector

The rule-based version produces professional output without external
dependencies. The LLM layer can enrich descriptions when available.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.framework import FRAMEWORK, MaturityLevel, classify_maturity


# ---------------------------------------------------------------------------
# Knowledge base: Initiatives by dimension and maturity level
# ---------------------------------------------------------------------------

INITIATIVES = {
    "data": {
        MaturityLevel.INITIAL: [
            {
                "title": "Data Inventory & Audit",
                "description": "Catalog all data sources, owners, and quality issues across the organization.",
                "effort": "low",
                "duration_weeks": 4,
                "roi": "foundational",
                "phase": 1,
            },
            {
                "title": "Establish Data Governance Council",
                "description": "Form a cross-functional team to define data ownership and policies.",
                "effort": "medium",
                "duration_weeks": 8,
                "roi": "foundational",
                "phase": 1,
            },
        ],
        MaturityLevel.EXPLORING: [
            {
                "title": "Implement Data Quality Monitoring",
                "description": "Deploy automated checks for completeness, consistency, and accuracy.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
            {
                "title": "Centralized Data Warehouse",
                "description": "Consolidate key data sources into a modern data warehouse (e.g. Snowflake, BigQuery).",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "very_high",
                "phase": 2,
            },
        ],
        MaturityLevel.DEVELOPING: [
            {
                "title": "Self-Service Data Access",
                "description": "Enable business users to query data via SQL editors, dashboards, and semantic layers.",
                "effort": "medium",
                "duration_weeks": 16,
                "roi": "high",
                "phase": 2,
            },
            {
                "title": "Real-Time Data Pipelines",
                "description": "Move from batch to streaming for high-value use cases (CDC, Kafka).",
                "effort": "high",
                "duration_weeks": 20,
                "roi": "high",
                "phase": 3,
            },
        ],
        MaturityLevel.SCALING: [
            {
                "title": "Data Mesh Architecture",
                "description": "Decentralize data ownership to domain teams while maintaining federated governance.",
                "effort": "very_high",
                "duration_weeks": 36,
                "roi": "high",
                "phase": 3,
            },
        ],
        MaturityLevel.OPTIMIZING: [
            {
                "title": "Data Product Marketplace",
                "description": "Treat data as products with SLAs, discovery, and usage analytics.",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "high",
                "phase": 3,
            },
        ],
    },
    "talent": {
        MaturityLevel.INITIAL: [
            {
                "title": "AI Literacy Program for Leadership",
                "description": "Run executive workshops to align on AI vocabulary, opportunities and risks.",
                "effort": "low",
                "duration_weeks": 6,
                "roi": "foundational",
                "phase": 1,
            },
            {
                "title": "Hire First Data Scientist / ML Engineer",
                "description": "Bring in senior talent to anchor your AI capability.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "very_high",
                "phase": 1,
            },
        ],
        MaturityLevel.EXPLORING: [
            {
                "title": "Company-Wide AI Fundamentals Course",
                "description": "Roll out a 4-6 week course on AI basics for all knowledge workers.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
            {
                "title": "Build a Data & AI Team",
                "description": "Grow a dedicated team of 4-8 specialists with mixed expertise.",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "very_high",
                "phase": 2,
            },
        ],
        MaturityLevel.DEVELOPING: [
            {
                "title": "AI Center of Excellence",
                "description": "Establish a CoE to drive standards, share knowledge, and incubate use cases.",
                "effort": "high",
                "duration_weeks": 20,
                "roi": "high",
                "phase": 2,
            },
        ],
        MaturityLevel.SCALING: [
            {
                "title": "Embedded AI Product Teams",
                "description": "Move from centralized CoE to embedded product teams with AI specialists.",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "high",
                "phase": 3,
            },
        ],
        MaturityLevel.OPTIMIZING: [
            {
                "title": "Internal AI Research Function",
                "description": "Invest in applied research to maintain competitive advantage.",
                "effort": "very_high",
                "duration_weeks": 36,
                "roi": "high",
                "phase": 3,
            },
        ],
    },
    "technology": {
        MaturityLevel.INITIAL: [
            {
                "title": "Cloud Migration Strategy",
                "description": "Define a multi-year roadmap to migrate from legacy on-prem to cloud.",
                "effort": "high",
                "duration_weeks": 16,
                "roi": "foundational",
                "phase": 1,
            },
        ],
        MaturityLevel.EXPLORING: [
            {
                "title": "ML Experimentation Platform",
                "description": "Provide notebooks, GPU access, and experiment tracking (e.g. Databricks, SageMaker, MLflow).",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
        ],
        MaturityLevel.DEVELOPING: [
            {
                "title": "Production MLOps Platform",
                "description": "Deploy CI/CD for models, monitoring, and automated retraining pipelines.",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "very_high",
                "phase": 2,
            },
            {
                "title": "Feature Store",
                "description": "Centralize feature engineering for reuse and consistency between training and serving.",
                "effort": "high",
                "duration_weeks": 20,
                "roi": "high",
                "phase": 3,
            },
        ],
        MaturityLevel.SCALING: [
            {
                "title": "LLM Platform & RAG Infrastructure",
                "description": "Build internal capabilities for fine-tuning, evaluation and grounded LLM applications.",
                "effort": "high",
                "duration_weeks": 24,
                "roi": "very_high",
                "phase": 3,
            },
        ],
        MaturityLevel.OPTIMIZING: [
            {
                "title": "Multi-Modal AI Infrastructure",
                "description": "Extend platforms to support vision, speech, and multi-modal models at scale.",
                "effort": "very_high",
                "duration_weeks": 36,
                "roi": "high",
                "phase": 3,
            },
        ],
    },
    "strategy": {
        MaturityLevel.INITIAL: [
            {
                "title": "Define AI Vision & Operating Principles",
                "description": "Hold a series of leadership workshops to articulate the role of AI in your strategy.",
                "effort": "low",
                "duration_weeks": 6,
                "roi": "foundational",
                "phase": 1,
            },
            {
                "title": "Identify Top 5 AI Use Cases",
                "description": "Run a structured discovery process to identify high-ROI opportunities.",
                "effort": "medium",
                "duration_weeks": 8,
                "roi": "high",
                "phase": 1,
            },
        ],
        MaturityLevel.EXPLORING: [
            {
                "title": "AI Investment Plan & Budget",
                "description": "Secure multi-year committed funding tied to a portfolio of initiatives.",
                "effort": "medium",
                "duration_weeks": 8,
                "roi": "foundational",
                "phase": 1,
            },
            {
                "title": "Define AI KPIs & Reporting Cadence",
                "description": "Establish business KPIs (not just model metrics) and a quarterly reporting cycle.",
                "effort": "low",
                "duration_weeks": 4,
                "roi": "high",
                "phase": 1,
            },
        ],
        MaturityLevel.DEVELOPING: [
            {
                "title": "Responsible AI Framework",
                "description": "Establish bias testing, explainability, and human-in-the-loop policies aligned with EU AI Act.",
                "effort": "medium",
                "duration_weeks": 16,
                "roi": "foundational",
                "phase": 2,
            },
        ],
        MaturityLevel.SCALING: [
            {
                "title": "Portfolio Management for AI",
                "description": "Adopt portfolio management practices: stage gates, kill criteria, value tracking.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
        ],
        MaturityLevel.OPTIMIZING: [
            {
                "title": "AI-First Product Strategy",
                "description": "Re-evaluate every product line through an AI-first lens; explore new business models.",
                "effort": "very_high",
                "duration_weeks": 36,
                "roi": "very_high",
                "phase": 3,
            },
        ],
    },
    "processes": {
        MaturityLevel.INITIAL: [
            {
                "title": "Process Documentation Sprint",
                "description": "Document the top 10 critical business processes as the basis for future automation.",
                "effort": "medium",
                "duration_weeks": 8,
                "roi": "foundational",
                "phase": 1,
            },
        ],
        MaturityLevel.EXPLORING: [
            {
                "title": "RPA / Workflow Automation Quick Wins",
                "description": "Automate 3-5 high-volume processes using low-code tools.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
        ],
        MaturityLevel.DEVELOPING: [
            {
                "title": "Process Mining Implementation",
                "description": "Use process mining to discover bottlenecks and AI augmentation opportunities.",
                "effort": "medium",
                "duration_weeks": 12,
                "roi": "high",
                "phase": 2,
            },
            {
                "title": "Embed AI in 2-3 Core Processes",
                "description": "Move beyond pilots and integrate AI into mission-critical workflows.",
                "effort": "high",
                "duration_weeks": 20,
                "roi": "very_high",
                "phase": 2,
            },
        ],
        MaturityLevel.SCALING: [
            {
                "title": "Closed-Loop Feedback Systems",
                "description": "Implement systems that capture outcomes from AI decisions to drive retraining.",
                "effort": "high",
                "duration_weeks": 16,
                "roi": "high",
                "phase": 3,
            },
        ],
        MaturityLevel.OPTIMIZING: [
            {
                "title": "Autonomous Process Optimization",
                "description": "Deploy systems that continuously optimize processes without human tuning.",
                "effort": "very_high",
                "duration_weeks": 36,
                "roi": "very_high",
                "phase": 3,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Roadmap generation
# ---------------------------------------------------------------------------

def _level_for_dimension(avg_score: float) -> MaturityLevel:
    """Same buckets as overall score."""
    return classify_maturity(avg_score)


def _select_initiatives(dim_id: str, dim_avg: float,
                        max_per_dim: int = 2) -> List[Dict]:
    """Pick relevant initiatives for a dimension based on its current level."""
    level = _level_for_dimension(dim_avg)
    level_initiatives = INITIATIVES.get(dim_id, {}).get(level, [])

    # If lower-level gaps exist, also pull a foundational item from the previous level
    if level != MaturityLevel.INITIAL and dim_avg < 3.0:
        levels_order = list(MaturityLevel)
        prev_level = levels_order[max(0, levels_order.index(level) - 1)]
        prev = INITIATIVES.get(dim_id, {}).get(prev_level, [])
        if prev:
            level_initiatives = prev[:1] + level_initiatives

    selected = []
    for init in level_initiatives[:max_per_dim]:
        item = init.copy()
        item["dimension_id"] = dim_id
        selected.append(item)
    return selected


def generate_roadmap(
    dimension_scores: Dict[str, float],
    overall_score: float,
    sector: Optional[str] = None,
    company_size: Optional[str] = None,
    weakest_dimensions: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a phased roadmap.

    Phases:
    - Phase 1: Quick wins & foundations (0-3 months)
    - Phase 2: Core capability building (3-9 months)
    - Phase 3: Scaling & differentiation (9-18 months)
    """
    # Sort dimensions by weakness (ascending score = highest priority)
    sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1])

    # Collect initiatives, prioritizing weaker dimensions
    all_initiatives = []
    for dim_id, dim_avg in sorted_dims:
        # Weaker dims get more initiatives
        max_per = 3 if dim_avg < 2.5 else 2 if dim_avg < 3.5 else 1
        all_initiatives.extend(_select_initiatives(dim_id, dim_avg, max_per_dim=max_per))

    # Group by phase
    phases = {1: [], 2: [], 3: []}
    for init in all_initiatives:
        phases[init["phase"]].append(init)

    # If phase 1 is empty, promote the lowest-effort phase-2 initiatives
    # of the two weakest dimensions to phase 1 as quick wins
    if not phases[1] and phases[2]:
        weakest_two = [d[0] for d in sorted_dims[:2]]
        candidates = [i for i in phases[2]
                      if i["dimension_id"] in weakest_two and
                      i["effort"] in ("low", "medium")]
        # Take up to 3, prefer shorter duration
        candidates.sort(key=lambda x: x["duration_weeks"])
        for c in candidates[:3]:
            promoted = c.copy()
            promoted["phase"] = 1
            promoted["promoted_from_phase_2"] = True
            phases[1].append(promoted)
            phases[2].remove(c)

    # Build phase descriptions adaptive to maturity level
    overall_level = classify_maturity(overall_score)
    phase_summaries = _phase_summaries(overall_level)

    return {
        "overall_score": overall_score,
        "overall_level": overall_level.value,
        "sector": sector,
        "company_size": company_size,
        "phases": [
            {
                "phase": 1,
                "title": "Foundations & Quick Wins (0-3 months)",
                "summary": phase_summaries[1],
                "initiatives": phases[1],
                "estimated_duration_weeks": _estimate_phase_duration(phases[1]),
            },
            {
                "phase": 2,
                "title": "Core Capability Building (3-9 months)",
                "summary": phase_summaries[2],
                "initiatives": phases[2],
                "estimated_duration_weeks": _estimate_phase_duration(phases[2]),
            },
            {
                "phase": 3,
                "title": "Scale & Differentiate (9-18 months)",
                "summary": phase_summaries[3],
                "initiatives": phases[3],
                "estimated_duration_weeks": _estimate_phase_duration(phases[3]),
            },
        ],
        "executive_summary": _executive_summary(
            overall_score, overall_level, sorted_dims, sector
        ),
    }


def _estimate_phase_duration(initiatives: List[Dict]) -> int:
    """Estimate phase duration as the max initiative duration (parallel execution)."""
    if not initiatives:
        return 0
    return max(i["duration_weeks"] for i in initiatives)


def _phase_summaries(level: MaturityLevel) -> Dict[int, str]:
    """Adaptive phase descriptions based on overall maturity."""
    if level in (MaturityLevel.INITIAL, MaturityLevel.EXPLORING):
        return {
            1: "Establish the basics: data inventory, leadership alignment, first hires.",
            2: "Build core platforms and run pilot use cases that prove value.",
            3: "Scale successful pilots and mature your AI operating model.",
        }
    elif level == MaturityLevel.DEVELOPING:
        return {
            1: "Close foundational gaps and secure executive commitment with a clear roadmap.",
            2: "Industrialize your ML and data platforms; embed AI in core processes.",
            3: "Move from project-based AI to product-led, continuously improving systems.",
        }
    elif level == MaturityLevel.SCALING:
        return {
            1: "Address remaining gaps to unlock the next phase of scale.",
            2: "Optimize MLOps, governance and cross-functional ways of working.",
            3: "Pursue differentiation through advanced capabilities and new business models.",
        }
    else:  # OPTIMIZING
        return {
            1: "Refine portfolio management and double down on highest-impact bets.",
            2: "Push the frontier with applied research and platform investments.",
            3: "Reinvent the business through AI-native products and services.",
        }


def _executive_summary(score: float, level: MaturityLevel,
                       sorted_dims: List, sector: Optional[str]) -> str:
    """Multi-sentence narrative summary."""
    weakest = sorted_dims[0][0] if sorted_dims else "data"
    strongest = sorted_dims[-1][0] if sorted_dims else "data"
    weakest_name = next(d.name for d in FRAMEWORK if d.id == weakest)
    strongest_name = next(d.name for d in FRAMEWORK if d.id == strongest)

    sector_phrase = f" Compared to peers in the {sector} sector," if sector else ""

    summary = (
        f"Your organization scores {score:.2f}/5.0 on the AI Maturity Index, "
        f"placing you at the {level.value} level. "
        f"Your strongest dimension is {strongest_name}, while {weakest_name} "
        f"represents the most significant opportunity for improvement.{sector_phrase} "
        f"the recommended path forward focuses on closing foundational gaps "
        f"in {weakest_name} during the first three months, while preparing "
        f"to scale capabilities in subsequent phases. "
        f"Each phase of the roadmap is sized to be achievable while maintaining "
        f"momentum toward the next maturity level."
    )
    return summary


# ---------------------------------------------------------------------------
# Health Summary Report
# ---------------------------------------------------------------------------

_DIM_HEALTH_THRESHOLDS = {
    "healthy":  3.5,
    "fair":     2.5,
    "at_risk":  1.5,
}

_HEALTH_COLORS = {
    "excellent": "#0d9488",
    "good":      "#16a34a",
    "fair":      "#ca8a04",
    "poor":      "#ea580c",
    "critical":  "#dc2626",
}

_DIM_HEALTH_COLORS = {
    "healthy":  "#16a34a",
    "fair":     "#ca8a04",
    "at_risk":  "#ea580c",
    "critical": "#dc2626",
}


def _dim_health_status(score: float) -> str:
    if score >= _DIM_HEALTH_THRESHOLDS["healthy"]:
        return "healthy"
    if score >= _DIM_HEALTH_THRESHOLDS["fair"]:
        return "fair"
    if score >= _DIM_HEALTH_THRESHOLDS["at_risk"]:
        return "at_risk"
    return "critical"


def _overall_health_status(score: float) -> str:
    if score >= 4.3:
        return "excellent"
    if score >= 3.5:
        return "good"
    if score >= 2.5:
        return "fair"
    if score >= 1.5:
        return "poor"
    return "critical"


def generate_health_summary(
    dimension_scores: Dict[str, float],
    overall_score: float,
    what_if_results: Optional[List[Dict]] = None,
    benchmark: Optional[Dict] = None,
    sector: Optional[str] = None,
) -> Dict:
    """
    Generate a concise health summary of an organization's AI maturity.

    Returns a structured snapshot with per-dimension health indicators,
    severity alerts, the single highest-impact action, and an overall
    health status.
    """
    health_status = _overall_health_status(overall_score)
    level = classify_maturity(overall_score)

    # Per-dimension health
    dim_health = []
    alerts = []
    for dim in FRAMEWORK:
        score = dimension_scores.get(dim.id, 0.0)
        status = _dim_health_status(score)
        dim_health.append({
            "dimension_id": dim.id,
            "dimension_name": dim.name,
            "score": round(score, 2),
            "status": status,
            "color": _DIM_HEALTH_COLORS[status],
        })
        if status in ("at_risk", "critical"):
            severity = "critical" if status == "critical" else "high"
            alerts.append({
                "severity": severity,
                "dimension_id": dim.id,
                "dimension_name": dim.name,
                "score": round(score, 2),
                "message": (
                    f"{dim.name} scores {score:.1f}/5 — "
                    f"{'critically low' if severity == 'critical' else 'below the recommended threshold'}. "
                    f"This is limiting your overall AI progress."
                ),
            })

    # Sort alerts: critical first
    alerts.sort(key=lambda a: 0 if a["severity"] == "critical" else 1)

    # Top recommended action: dimension with highest what-if gain, or lowest score
    top_action = None
    if what_if_results:
        best = max(what_if_results, key=lambda w: w.get("delta", 0))
        top_action = {
            "dimension_id": best["dimension_id"],
            "dimension_name": best["dimension_name"],
            "title": _top_initiative_title(best["dimension_id"], dimension_scores.get(best["dimension_id"], 1.0)),
            "expected_score_gain": round(best.get("delta", 0), 3),
        }
    else:
        weakest_id, weakest_score = min(dimension_scores.items(), key=lambda x: x[1])
        weakest_dim = next(d for d in FRAMEWORK if d.id == weakest_id)
        top_action = {
            "dimension_id": weakest_id,
            "dimension_name": weakest_dim.name,
            "title": _top_initiative_title(weakest_id, weakest_score),
            "expected_score_gain": None,
        }

    # Improvement potential: score gain if all at-risk dims move to 3.5
    improvement_potential = sum(
        max(0.0, 3.5 - dimension_scores.get(d.id, 0.0)) * d.weight
        for d in FRAMEWORK
    )

    # Headline
    at_risk_count = sum(1 for d in dim_health if d["status"] in ("at_risk", "critical"))
    if at_risk_count == 0:
        headline = f"Your AI health is {health_status} — all dimensions are on track"
    elif at_risk_count == 1:
        headline = f"Your AI health is {health_status} — 1 dimension needs attention"
    else:
        headline = f"Your AI health is {health_status} — {at_risk_count} dimensions need attention"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": round(overall_score, 3),
        "maturity_level": level.value,
        "health_status": health_status,
        "health_color": _HEALTH_COLORS[health_status],
        "summary_headline": headline,
        "dimension_health": dim_health,
        "alerts": alerts,
        "top_action": top_action,
        "improvement_potential": round(improvement_potential, 3),
        "benchmark_percentile": benchmark.get("your_percentile") if benchmark else None,
        "sector": sector,
    }


def _top_initiative_title(dim_id: str, dim_score: float) -> str:
    """Return the title of the highest-priority initiative for a dimension."""
    level = _level_for_dimension(dim_score)
    initiatives = INITIATIVES.get(dim_id, {}).get(level, [])
    if initiatives:
        return initiatives[0]["title"]
    # Fall back to previous level if current level has no initiatives
    levels_order = list(MaturityLevel)
    idx = levels_order.index(level)
    if idx > 0:
        prev = INITIATIVES.get(dim_id, {}).get(levels_order[idx - 1], [])
        if prev:
            return prev[0]["title"]
    return "Review and strengthen this dimension"


# ---------------------------------------------------------------------------
# Optional: LLM augmentation (graceful fallback)
# ---------------------------------------------------------------------------

def augment_with_llm(roadmap: Dict, api_key: Optional[str] = None) -> Dict:
    """
    Optional LLM augmentation step.
    If no API key is available, returns the roadmap unchanged.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return roadmap  # graceful no-op

    # Placeholder: in production we would call the Anthropic API to enrich
    # initiative descriptions with sector-specific context. The deterministic
    # version above is already production-quality.
    return roadmap


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_dim_scores = {
        "data": 3.6,
        "talent": 2.6,
        "technology": 3.8,
        "strategy": 3.2,
        "processes": 2.6,
    }
    overall = sum(s * d.weight for s, d in zip(sample_dim_scores.values(), FRAMEWORK))

    rm = generate_roadmap(
        dimension_scores=sample_dim_scores,
        overall_score=overall,
        sector="manufacturing",
        company_size="medium",
    )

    print(f"Overall: {rm['overall_score']:.2f} ({rm['overall_level']})")
    print(f"\nExecutive summary:\n{rm['executive_summary']}\n")
    for phase in rm["phases"]:
        print(f"\n=== {phase['title']} ===")
        print(f"  {phase['summary']}")
        print(f"  Estimated duration: ~{phase['estimated_duration_weeks']} weeks")
        for init in phase["initiatives"]:
            print(f"  • [{init['dimension_id']}] {init['title']}  "
                  f"(ROI: {init['roi']}, effort: {init['effort']})")
