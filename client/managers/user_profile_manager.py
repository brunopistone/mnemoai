"""Simplified user profile learning system with EMA-based tracking."""

from datetime import datetime
import json
import os
import re
import textwrap
from typing import Dict, List, Any, Optional
from utils.config import config
from utils.logger import logger
from utils.paths import profile_dir


class UserProfileManager:
    """Manages user interaction patterns with efficient EMA-based learning."""

    # Intent patterns for classification
    INTENT_PATTERNS = {
        "debug": r"\b(error|bug|issue|problem|not working|broken|fail|debug|fix)\b",
        "learn": r"\b(how|what|why|explain|understand|learn|teach|documentation)\b",
        "implement": r"\b(create|build|implement|make|write|develop|code|add)\b",
        "optimize": r"\b(improve|optimize|faster|better|performance|refactor|efficient)\b",
        "review": r"\b(review|check|look at|analyze|examine|audit)\b",
        "configure": r"\b(setup|configure|install|deploy|set up|initialize)\b",
        "search": r"\b(find|search|locate|where|look for)\b",
        "read": r"\b(read|show|display|get|fetch|retrieve)\b",
    }

    # Domain keywords for detection
    DOMAIN_KEYWORDS = {
        "python": ["python", "pip", "pytest", "django", "flask", "pandas", "numpy"],
        "javascript": ["javascript", "node", "npm", "react", "vue", "typescript"],
        "devops": ["docker", "kubernetes", "ci/cd", "jenkins", "terraform", "ansible"],
        "aws": ["aws", "ec2", "s3", "lambda", "sagemaker", "bedrock", "cloudformation"],
        "databases": ["sql", "postgres", "mysql", "mongodb", "redis", "database"],
    }

    # Technical terms for level detection
    TECHNICAL_TERMS = [
        "api",
        "async",
        "callback",
        "closure",
        "decorator",
        "dependency",
        "endpoint",
        "framework",
        "middleware",
        "microservice",
        "oauth",
        "polymorphism",
        "refactor",
        "repository",
        "scalability",
        "schema",
        "singleton",
        "threading",
        "validation",
        "webpack",
        "websocket",
    ]

    def __init__(self, profile_path: str = None) -> None:
        """Initialize user profile manager.

        Args:
            profile_path: Optional path to profile file
        """
        if profile_path is None:
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            profile_path = os.path.join(str(profile_dir()), f"{profile_name}.json")

        self.profile_path = profile_path
        self.profile = self._load_profile()

        # Migrate legacy profile if needed
        if not self.profile.get("_legacy_migrated", False):
            self._migrate_profile()

    def _load_profile(self) -> Dict:
        """Load existing profile or create new one.

        Returns:
            Profile dictionary
        """
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load profile: {e}")

        # New simplified profile structure
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "interaction_count": 0,
            # EMA-based metrics (0-1 scale)
            "verbosity": 0.5,  # 0=concise, 1=detailed
            "directness": 0.5,  # 0=conversational, 1=direct
            "technical_level": 0.5,  # 0=beginner, 1=expert
            "abstraction": 0.5,  # 0=examples, 1=concepts
            # Domain tracking
            "top_domains": [],  # [{name, score, last_seen}]
            # Tool success patterns
            "tool_patterns": {},  # intent -> tool -> {success, total}
            # Migration flag
            "_legacy_migrated": True,
        }

    def _migrate_profile(self) -> None:
        """Migrate legacy profile to new simplified schema."""
        logger.info("Migrating legacy profile to new schema...")

        # Initialize new fields if missing
        for field, default in [
            ("verbosity", 0.5),
            ("directness", 0.5),
            ("technical_level", 0.5),
            ("abstraction", 0.5),
            ("top_domains", []),
            ("tool_patterns", {}),
        ]:
            if field not in self.profile:
                self.profile[field] = default

        # Migrate from legacy complexity_preference
        complexity = self.profile.get("complexity_preference", {})
        simple = complexity.get("simple", 0)
        detailed = complexity.get("detailed", 0)
        if simple + detailed > 0:
            self.profile["verbosity"] = detailed / (simple + detailed)

        # Migrate from legacy sentiment
        sentiment = self.profile.get("sentiment", {})
        direct = sentiment.get("direct", 0)
        polite = sentiment.get("polite", 0) + sentiment.get("exploratory", 0)
        if direct + polite > 0:
            self.profile["directness"] = direct / (direct + polite)

        # Migrate from legacy communication_patterns
        comm = self.profile.get("communication_patterns", {})
        tech_depth = comm.get("technical_depth", 0)
        if tech_depth > 0:
            # Normalize: 0-5 = beginner, 5-10 = intermediate, 10+ = expert
            self.profile["technical_level"] = min(1.0, tech_depth / 15)

        # Migrate from legacy abstraction_level
        abstraction = self.profile.get("abstraction_level", {})
        concrete = abstraction.get("concrete", 0)
        abstract = abstraction.get("abstract", 0)
        if concrete + abstract > 0:
            self.profile["abstraction"] = abstract / (concrete + abstract)

        # Migrate from legacy domain_expertise
        domains = self.profile.get("domain_expertise", {})
        if domains:
            sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            max_score = max(d[1] for d in sorted_domains) if sorted_domains else 1
            self.profile["top_domains"] = [
                {
                    "name": name,
                    "score": score / max_score,  # Normalize to 0-1
                    "last_seen": datetime.now().isoformat(),
                }
                for name, score in sorted_domains
            ]

        self.profile["_legacy_migrated"] = True
        self._save_profile()
        logger.info("Profile migration complete")

    def _save_profile(self) -> None:
        """Save profile to disk."""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        self.profile["last_updated"] = datetime.now().isoformat()

        # Prune tool_patterns to prevent unbounded growth
        MAX_TOOLS_PER_INTENT = 10
        for intent, tools in self.profile.get("tool_patterns", {}).items():
            if len(tools) > MAX_TOOLS_PER_INTENT:
                # Keep top tools by success rate
                sorted_tools = sorted(
                    tools.items(),
                    key=lambda x: x[1]["success"] / max(x[1]["total"], 1),
                    reverse=True,
                )[:MAX_TOOLS_PER_INTENT]
                self.profile["tool_patterns"][intent] = dict(sorted_tools)

        try:
            with open(self.profile_path, "w") as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def _update_ema(
        self, current: float, observation: float, alpha: float = 0.15
    ) -> float:
        """Update value using exponential moving average.

        Args:
            current: Current EMA value (0-1)
            observation: New observation (0 or 1)
            alpha: Smoothing factor (higher = more weight to recent)

        Returns:
            Updated EMA value
        """
        return alpha * observation + (1 - alpha) * current

    def classify_intent(self, query: str) -> str:
        """Classify query intent for tool pattern tracking.

        Args:
            query: User query text

        Returns:
            Intent classification string
        """
        query_lower = query.lower()

        # Check patterns in priority order
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, query_lower):
                return intent

        return "general"

    def record_tool_outcome(
        self,
        intent: str,
        tools_used: List[Dict[str, Any]],
        success: bool = True,
    ) -> None:
        """Update tool success patterns from task outcomes.

        Args:
            intent: Classified intent (debug, implement, etc.)
            tools_used: List of tools from extract_tools_from_messages()
            success: Whether task was successful
        """
        if not tools_used:
            return

        if "tool_patterns" not in self.profile:
            self.profile["tool_patterns"] = {}

        if intent not in self.profile["tool_patterns"]:
            self.profile["tool_patterns"][intent] = {}

        for tool in tools_used:
            tool_name = tool.get("name", "unknown")
            if tool_name == "unknown":
                continue

            if tool_name not in self.profile["tool_patterns"][intent]:
                self.profile["tool_patterns"][intent][tool_name] = {
                    "success": 0,
                    "total": 0,
                }

            stats = self.profile["tool_patterns"][intent][tool_name]
            stats["total"] += 1
            if success:
                stats["success"] += 1

        self._save_profile()
        logger.debug(
            f"Recorded tool outcome: intent={intent}, success={success}, tools={[t.get('name') for t in tools_used]}"
        )

    def analyze_conversation(self, messages: List[Dict]) -> None:
        """Analyze conversation and update profile with EMA.

        Args:
            messages: List of conversation messages
        """
        if not messages:
            return

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return

        # Update interaction count
        self.profile["interaction_count"] += len(user_messages)

        # Analyze each user message
        for msg in user_messages:
            content = self._extract_text_content(msg)
            if content:
                self._analyze_message(content)

        self._save_profile()

    def _extract_text_content(self, message: Dict) -> str:
        """Extract text content from message.

        Args:
            message: Message dictionary

        Returns:
            Extracted text content
        """
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return " ".join(texts)
        return ""

    def _analyze_message(self, content: str) -> None:
        """Analyze message and update EMA metrics.

        Args:
            content: Message content text
        """
        content_lower = content.lower()
        words = re.findall(r"\b\w+\b", content_lower)

        # === VERBOSITY ===
        # Short messages (< 50 chars) = concise, long (> 200) = detailed
        verbosity_signal = 0.0
        if len(content) < 50:
            verbosity_signal = 0.0
        elif len(content) > 200:
            verbosity_signal = 1.0
        else:
            verbosity_signal = (len(content) - 50) / 150  # Linear interpolation

        # Explicit markers
        concise_markers = ["brief", "short", "quick", "just", "simply", "only"]
        detailed_markers = [
            "detailed",
            "comprehensive",
            "thorough",
            "explain fully",
            "in depth",
        ]

        if any(m in content_lower for m in concise_markers):
            verbosity_signal = 0.0
        elif any(m in content_lower for m in detailed_markers):
            verbosity_signal = 1.0

        self.profile["verbosity"] = self._update_ema(
            self.profile.get("verbosity", 0.5), verbosity_signal
        )

        # === DIRECTNESS ===
        # Imperatives = direct, questions/politeness = conversational
        directness_signal = 0.5

        imperative_starters = [
            "show",
            "give",
            "create",
            "make",
            "write",
            "fix",
            "add",
            "remove",
            "delete",
        ]
        polite_markers = [
            "please",
            "could you",
            "would you",
            "can you",
            "thanks",
            "thank you",
        ]
        question_markers = ["?", "how do", "what is", "why does", "can i"]

        has_imperative = any(
            content_lower.startswith(imp) for imp in imperative_starters
        )
        has_polite = any(m in content_lower for m in polite_markers)
        has_question = any(m in content_lower for m in question_markers)

        if has_imperative and not has_polite:
            directness_signal = 1.0
        elif has_polite or has_question:
            directness_signal = 0.2

        self.profile["directness"] = self._update_ema(
            self.profile.get("directness", 0.5), directness_signal
        )

        # === TECHNICAL LEVEL ===
        tech_count = sum(1 for term in self.TECHNICAL_TERMS if term in content_lower)
        tech_signal = min(1.0, tech_count / 3)  # 3+ technical terms = expert level

        self.profile["technical_level"] = self._update_ema(
            self.profile.get("technical_level", 0.5), tech_signal
        )

        # === ABSTRACTION ===
        concrete_markers = [
            "example",
            "show me",
            "sample",
            "demo",
            "code for",
            "how to",
        ]
        abstract_markers = [
            "explain",
            "why",
            "concept",
            "theory",
            "principle",
            "understand",
        ]

        wants_concrete = any(m in content_lower for m in concrete_markers)
        wants_abstract = any(m in content_lower for m in abstract_markers)

        if wants_concrete and not wants_abstract:
            abstraction_signal = 0.0
        elif wants_abstract and not wants_concrete:
            abstraction_signal = 1.0
        else:
            abstraction_signal = 0.5

        self.profile["abstraction"] = self._update_ema(
            self.profile.get("abstraction", 0.5), abstraction_signal
        )

        # === DOMAINS ===
        detected_domains = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                detected_domains[domain] = matches

        if detected_domains:
            self._update_domains(detected_domains)

    def _update_domains(self, detected: Dict[str, int]) -> None:
        """Update top domains with decay.

        Args:
            detected: Dict of domain -> match count
        """
        current_domains = {d["name"]: d for d in self.profile.get("top_domains", [])}

        # Decay existing scores
        for domain in current_domains.values():
            domain["score"] = domain["score"] * 0.95  # 5% decay per interaction

        # Add/update detected domains
        for name, count in detected.items():
            boost = min(0.3, count * 0.1)  # Up to 0.3 boost per detection
            if name in current_domains:
                current_domains[name]["score"] = min(
                    1.0, current_domains[name]["score"] + boost
                )
                current_domains[name]["last_seen"] = datetime.now().isoformat()
            else:
                current_domains[name] = {
                    "name": name,
                    "score": boost,
                    "last_seen": datetime.now().isoformat(),
                }

        # Keep top 5 domains
        sorted_domains = sorted(
            current_domains.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:5]

        # Filter out very low scores
        self.profile["top_domains"] = [d for d in sorted_domains if d["score"] > 0.05]

    def get_profile_summary(self) -> str:
        """Generate compact profile summary for system prompt.

        Returns:
            Formatted profile summary (~100 tokens)
        """
        if self.profile.get("interaction_count", 0) < 5:
            return ""  # Not enough data

        # Determine style labels from EMA values
        verbosity = self.profile.get("verbosity", 0.5)
        directness = self.profile.get("directness", 0.5)
        tech_level = self.profile.get("technical_level", 0.5)
        abstraction = self.profile.get("abstraction", 0.5)

        verbosity_label = (
            "concise"
            if verbosity < 0.4
            else "detailed" if verbosity > 0.6 else "balanced"
        )
        style_label = (
            "direct"
            if directness > 0.6
            else "conversational" if directness < 0.4 else "balanced"
        )
        tech_label = (
            "expert"
            if tech_level > 0.7
            else "intermediate" if tech_level > 0.4 else "beginner"
        )
        abstraction_label = (
            "examples"
            if abstraction < 0.4
            else "concepts" if abstraction > 0.6 else "mixed"
        )

        # Top domains
        top_domains = self.profile.get("top_domains", [])[:3]
        domains_str = (
            ", ".join(d["name"] for d in top_domains) if top_domains else "general"
        )

        # Best tools per intent (top 3 intents with best tools)
        tool_hints = []
        tool_patterns = self.profile.get("tool_patterns", {})

        # Sort intents by total usage
        sorted_intents = sorted(
            tool_patterns.items(),
            key=lambda x: sum(t["total"] for t in x[1].values()),
            reverse=True,
        )[:3]

        for intent, tools in sorted_intents:
            if not tools:
                continue
            # Get best tool by success rate (min 3 uses)
            qualified_tools = [
                (name, stats) for name, stats in tools.items() if stats["total"] >= 3
            ]
            if qualified_tools:
                best_tool = max(
                    qualified_tools, key=lambda x: x[1]["success"] / x[1]["total"]
                )
                tool_hints.append(f"{intent}: {best_tool[0]}")

        tools_str = "; ".join(tool_hints) if tool_hints else "learning"

        return textwrap.dedent(
            f"""
                <profile>
                Style: {verbosity_label}, {style_label}, {tech_label}-level
                Domains: {domains_str}
                Prefers: {abstraction_label}
                Tools: {tools_str}
                </profile>
            """
        ).strip()
