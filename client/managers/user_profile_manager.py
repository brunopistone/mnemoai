"""Automatic user profile learning system."""

from datetime import datetime
import json
import os
from utils.logger import logger
import re
import sys
from typing import Dict, List

# Add parent directory to path to allow imports from root
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.config import config


class UserProfileManager:
    """Manages and learns user interaction patterns automatically."""

    def __init__(self, profile_path: str = None) -> None:
        """Initialize user profile manager.

        Args:
            profile_path: Optional path to profile file
        """
        if profile_path is None:
            # Use profile-based directory structure
            profile_config = config.get("PROFILE", {})
            profile_name = profile_config.get("NAME", "default")

            user_home = os.path.expanduser("~")
            profile_dir = os.path.join(user_home, "agent-conversations", profile_name)
            os.makedirs(profile_dir, exist_ok=True)

            profile_path = os.path.join(profile_dir, f"{profile_name}.json")

        self.profile_path = profile_path
        self.profile = self._load_profile()

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

        # Default profile structure
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "interaction_count": 0,
            "common_phrases": {},
            "frequent_topics": {},
            "preferred_response_style": "technical",
            "average_query_length": 0,
            "common_commands": {},
            "tool_usage_patterns": {},
            "communication_patterns": {
                "uses_technical_terms": False,
                "prefers_concise": True,
                "asks_follow_ups": False,
            },
        }

    def _save_profile(self) -> None:
        """Save profile to disk with size limits."""
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        self.profile["last_updated"] = datetime.now().isoformat()

        # Prune large dictionaries to prevent unbounded growth
        MAX_PHRASES = 100
        MAX_TOPICS = 50
        MAX_COMMANDS = 50

        # Keep only top N phrases by frequency
        if len(self.profile.get("common_phrases", {})) > MAX_PHRASES:
            top_phrases = sorted(
                self.profile["common_phrases"].items(), key=lambda x: x[1], reverse=True
            )[:MAX_PHRASES]
            self.profile["common_phrases"] = dict(top_phrases)

        # Keep only top N topics
        if len(self.profile.get("frequent_topics", {})) > MAX_TOPICS:
            top_topics = sorted(
                self.profile["frequent_topics"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:MAX_TOPICS]
            self.profile["frequent_topics"] = dict(top_topics)

        # Keep only top N commands
        if len(self.profile.get("common_commands", {})) > MAX_COMMANDS:
            top_commands = sorted(
                self.profile["common_commands"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:MAX_COMMANDS]
            self.profile["common_commands"] = dict(top_commands)

        try:
            with open(self.profile_path, "w") as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def analyze_conversation(self, messages: List[Dict]) -> None:
        """Analyze conversation and update profile.

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

        # Save updated profile
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
        """Analyze individual message for patterns.

        Args:
            content: Message content text
        """

        current_avg = self.profile["average_query_length"]
        count = self.profile["interaction_count"]
        self.profile["average_query_length"] = (
            current_avg * (count - 1) + len(content)
        ) / count

        content_lower = content.lower()
        sentences = re.split(r"[.!?]+", content)
        words = re.findall(r"\b\w+\b", content_lower)

        # Initialize new profile dimensions
        for key in [
            "sentiment",
            "domain_expertise",
            "intent_patterns",
            "code_preferences",
            "complexity_preference",
            "cognitive_style",
            "error_patterns",
            "workflow_patterns",
            "abstraction_level",
            "communication_efficiency",
            "learning_indicators",
        ]:
            if key not in self.profile:
                self.profile[key] = {}

        # Cognitive style analysis - Enhanced keywords
        analytical_markers = [
            "analyze",
            "compare",
            "evaluate",
            "assess",
            "examine",
            "investigate",
            "breakdown",
            "optimize",
            "refactor",
            "algorithm",
            "complexity",
            "pattern",
            "abstraction",
        ]
        creative_markers = [
            "design",
            "architect",
            "innovate",
            "alternative",
            "different approach",
            "creative",
            "experiment",
            "explore",
            "brainstorm",
        ]
        pragmatic_markers = [
            "works",
            "practical",
            "simple",
            "quick",
            "just need",
            "get it done",
            "working solution",
            "straightforward",
            "minimal",
            "basic",
        ]
        systematic_markers = [
            "step by step",
            "methodical",
            "structured",
            "organized",
            "systematic",
            "process",
            "sequential",
            "ordered",
            "planned",
        ]

        self.profile["cognitive_style"]["analytical"] = self.profile[
            "cognitive_style"
        ].get("analytical", 0) + sum(
            1 for m in analytical_markers if m in content_lower
        )
        self.profile["cognitive_style"]["creative"] = self.profile[
            "cognitive_style"
        ].get("creative", 0) + sum(1 for m in creative_markers if m in content_lower)
        self.profile["cognitive_style"]["pragmatic"] = self.profile[
            "cognitive_style"
        ].get("pragmatic", 0) + sum(1 for m in pragmatic_markers if m in content_lower)
        self.profile["cognitive_style"]["systematic"] = self.profile[
            "cognitive_style"
        ].get("systematic", 0) + sum(
            1 for m in systematic_markers if m in content_lower
        )

        # Error pattern analysis
        error_contexts = {
            "syntax": r"(syntax error|indentation|unexpected token|parse error)",
            "runtime": r"(runtime error|exception|crash|fails when|breaks)",
            "logic": r"(wrong result|incorrect output|not behaving|unexpected behavior)",
            "performance": r"(slow|timeout|memory|inefficient|bottleneck)",
            "integration": r"(api.*fail|connection|authentication|permission denied)",
        }

        for error_type, pattern in error_contexts.items():
            if re.search(pattern, content_lower):
                self.profile["error_patterns"][error_type] = (
                    self.profile["error_patterns"].get(error_type, 0) + 1
                )

        # Workflow pattern detection
        workflow_indicators = {
            "iterative": r"(try|attempt|iterate|refine|improve|adjust)",
            "exploratory": r"(explore|investigate|research|understand|learn about)",
            "goal_oriented": r"(need to|must|have to|goal|objective|accomplish)",
            "experimental": r"(test|experiment|see if|what happens|try out)",
            "maintenance": r"(fix|repair|debug|resolve|troubleshoot)",
            "greenfield": r"(create|build|start|new project|from scratch)",
        }

        for workflow, pattern in workflow_indicators.items():
            if re.search(pattern, content_lower):
                self.profile["workflow_patterns"][workflow] = (
                    self.profile["workflow_patterns"].get(workflow, 0) + 1
                )

        # Abstraction level preference
        concrete_markers = [
            "example",
            "show me",
            "specific",
            "actual code",
            "real",
            "concrete",
        ]
        abstract_markers = [
            "concept",
            "theory",
            "principle",
            "pattern",
            "architecture",
            "design",
            "approach",
        ]

        concrete_score = sum(1 for m in concrete_markers if m in content_lower)
        abstract_score = sum(1 for m in abstract_markers if m in content_lower)

        self.profile["abstraction_level"]["concrete"] = (
            self.profile["abstraction_level"].get("concrete", 0) + concrete_score
        )
        self.profile["abstraction_level"]["abstract"] = (
            self.profile["abstraction_level"].get("abstract", 0) + abstract_score
        )

        # Communication efficiency metrics
        question_density = content.count("?") / max(len(sentences), 1)
        imperative_count = len(
            re.findall(
                r"^(show|give|create|make|write|explain|help|tell)",
                content_lower,
                re.MULTILINE,
            )
        )

        self.profile["communication_efficiency"]["question_density"] = (
            self.profile["communication_efficiency"].get("question_density", 0) * 0.7
            + question_density * 0.3
        )
        self.profile["communication_efficiency"]["uses_imperatives"] = (
            self.profile["communication_efficiency"].get("uses_imperatives", 0)
            + imperative_count
        )
        self.profile["communication_efficiency"]["avg_sentence_length"] = len(
            words
        ) / max(len(sentences), 1)

        # Learning style indicators - Enhanced keywords
        visual_learner = [
            "diagram",
            "chart",
            "visualize",
            "show",
            "see",
            "picture",
            "graph",
            "illustration",
            "screenshot",
            "image",
            "visual",
        ]
        hands_on_learner = [
            "try",
            "practice",
            "example",
            "demo",
            "hands on",
            "interactive",
            "experiment",
            "test",
            "run",
            "execute",
        ]
        theoretical_learner = [
            "why",
            "how it works",
            "explain",
            "understand",
            "theory",
            "concept",
            "principle",
            "reasoning",
            "logic",
            "foundation",
        ]

        self.profile["learning_indicators"]["visual"] = self.profile[
            "learning_indicators"
        ].get("visual", 0) + sum(1 for m in visual_learner if m in content_lower)
        self.profile["learning_indicators"]["hands_on"] = self.profile[
            "learning_indicators"
        ].get("hands_on", 0) + sum(1 for m in hands_on_learner if m in content_lower)
        self.profile["learning_indicators"]["theoretical"] = self.profile[
            "learning_indicators"
        ].get("theoretical", 0) + sum(
            1 for m in theoretical_learner if m in content_lower
        )

        # Sentiment & tone - Enhanced with positive/negative/humor
        polite_markers = [
            "please",
            "could you",
            "would you mind",
            "if possible",
            "thanks",
            "thank you",
            "appreciate",
        ]
        direct_markers = ["show me", "give me", "i need", "i want", "do this", "just"]
        frustrated_markers = [
            "still",
            "not working",
            "broken",
            "issue",
            "problem",
            "error",
            "wrong",
            "confused",
            "fail",
            "doesn't work",
        ]
        exploratory_markers = [
            "what if",
            "how about",
            "could we",
            "is it possible",
            "wondering",
            "curious",
            "interested",
        ]
        confident_markers = [
            "i think",
            "should be",
            "probably",
            "likely",
            "seems like",
            "i believe",
        ]
        positive_markers = [
            "great",
            "good",
            "excellent",
            "perfect",
            "awesome",
            "love",
            "nice",
        ]
        negative_markers = ["bad", "terrible", "awful", "hate", "worst", "useless"]
        humor_markers = ["lol", "haha", "hehe", "funny", "joke"]
        exploratory_markers = [
            "what if",
            "how about",
            "could we",
            "is it possible",
            "wondering",
            "curious",
            "interested",
        ]
        confident_markers = [
            "i think",
            "should be",
            "probably",
            "likely",
            "seems like",
            "i believe",
        ]

        self.profile["sentiment"]["polite"] = self.profile["sentiment"].get(
            "polite", 0
        ) + sum(1 for m in polite_markers if m in content_lower)
        self.profile["sentiment"]["direct"] = self.profile["sentiment"].get(
            "direct", 0
        ) + sum(1 for m in direct_markers if m in content_lower)
        self.profile["sentiment"]["frustrated"] = self.profile["sentiment"].get(
            "frustrated", 0
        ) + sum(1 for m in frustrated_markers if m in content_lower)
        self.profile["sentiment"]["exploratory"] = self.profile["sentiment"].get(
            "exploratory", 0
        ) + sum(1 for m in exploratory_markers if m in content_lower)
        self.profile["sentiment"]["confident"] = self.profile["sentiment"].get(
            "confident", 0
        ) + sum(1 for m in confident_markers if m in content_lower)
        self.profile["sentiment"]["positive"] = self.profile["sentiment"].get(
            "positive", 0
        ) + sum(1 for m in positive_markers if m in content_lower)
        self.profile["sentiment"]["negative"] = self.profile["sentiment"].get(
            "negative", 0
        ) + sum(1 for m in negative_markers if m in content_lower)
        self.profile["sentiment"]["humor"] = self.profile["sentiment"].get(
            "humor", 0
        ) + sum(1 for m in humor_markers if m in content_lower)

        # Domain expertise with depth scoring
        domains = {
            "python": (
                ["python", "pip", "virtualenv", "pytest", "django", "flask"],
                ["decorator", "metaclass", "generator", "async", "gil", "descriptor"],
            ),
            "javascript": (
                ["javascript", "node", "npm", "react", "vue"],
                ["closure", "prototype", "promise", "async/await", "event loop"],
            ),
            "devops": (
                ["docker", "kubernetes", "ci/cd", "jenkins"],
                ["helm", "istio", "gitops", "canary", "blue-green"],
            ),
            "aws": (
                ["aws", "ec2", "s3", "lambda"],
                [
                    "vpc",
                    "iam policy",
                    "cloudformation",
                    "step functions",
                    "eventbridge",
                ],
            ),
            "databases": (
                ["sql", "postgres", "mysql", "mongodb"],
                ["index optimization", "query plan", "sharding", "replication", "acid"],
            ),
            "ml_ai": (
                ["model", "training", "inference"],
                [
                    "transformer",
                    "embedding",
                    "fine-tuning",
                    "rag",
                    "prompt engineering",
                ],
            ),
            "architecture": (
                ["api", "rest", "microservices"],
                ["saga pattern", "cqrs", "event sourcing", "ddd", "hexagonal"],
            ),
            "security": (
                ["security", "authentication", "encryption"],
                ["oauth2", "jwt", "zero trust", "principle of least privilege"],
            ),
        }

        for domain, (basic_kw, advanced_kw) in domains.items():
            basic_matches = sum(1 for kw in basic_kw if kw in content_lower)
            advanced_matches = sum(1 for kw in advanced_kw if kw in content_lower)
            score = basic_matches + (advanced_matches * 2)
            if score > 0:
                self.profile["domain_expertise"][domain] = (
                    self.profile["domain_expertise"].get(domain, 0) + score
                )

        # Intent classification with context
        intents = {
            "debug": r"\b(error|bug|issue|problem|not working|broken|fail|debug)\b",
            "learn": r"\b(how|what|why|explain|understand|learn|teach|documentation)\b",
            "implement": r"\b(create|build|implement|make|write|develop|code)\b",
            "optimize": r"\b(improve|optimize|faster|better|performance|refactor|efficient)\b",
            "review": r"\b(review|check|look at|analyze|examine|audit)\b",
            "configure": r"\b(setup|configure|install|deploy|set up|initialize)\b",
            "integrate": r"\b(integrate|connect|combine|merge|api|interface)\b",
            "migrate": r"\b(migrate|upgrade|convert|port|transition)\b",
        }

        for intent, pattern in intents.items():
            matches = len(re.findall(pattern, content_lower))
            if matches > 0:
                self.profile["intent_patterns"][intent] = (
                    self.profile["intent_patterns"].get(intent, 0) + matches
                )

        # Code quality preferences
        quality_indicators = {
            "testing": ["test", "testing", "unittest", "pytest", "coverage", "tdd"],
            "documentation": [
                "comment",
                "documentation",
                "docstring",
                "readme",
                "docs",
            ],
            "typing": ["type", "typing", "annotation", "interface", "generic"],
            "error_handling": [
                "error handling",
                "exception",
                "try catch",
                "validation",
                "robust",
            ],
            "clean_code": [
                "clean",
                "readable",
                "maintainable",
                "solid",
                "dry",
                "refactor",
            ],
            "performance": [
                "performance",
                "optimize",
                "efficient",
                "fast",
                "benchmark",
            ],
            "security": ["secure", "security", "sanitize", "validate", "injection"],
        }

        for pref, keywords in quality_indicators.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                self.profile["code_preferences"][pref] = (
                    self.profile["code_preferences"].get(pref, 0) + matches
                )

        # Complexity preference with nuance
        simplicity_markers = [
            "simple",
            "basic",
            "quick",
            "just",
            "minimal",
            "straightforward",
            "easy",
        ]
        detail_markers = [
            "detailed",
            "comprehensive",
            "complete",
            "full",
            "thorough",
            "in-depth",
            "elaborate",
        ]

        simplicity_score = sum(1 for m in simplicity_markers if m in content_lower)
        detail_score = sum(1 for m in detail_markers if m in content_lower)

        if len(content) < 80:
            simplicity_score += 1
        elif len(content) > 250:
            detail_score += 1

        self.profile["complexity_preference"]["simple"] = (
            self.profile["complexity_preference"].get("simple", 0) + simplicity_score
        )
        self.profile["complexity_preference"]["detailed"] = (
            self.profile["complexity_preference"].get("detailed", 0) + detail_score
        )

        # Semantic phrase extraction with TF-IDF-like scoring
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "me",
            "him",
            "them",
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
        }

        for n in [2, 3, 4]:
            for i in range(len(words) - n + 1):
                ngram = words[i : i + n]
                meaningful_words = [
                    w for w in ngram if w not in stop_words and len(w) > 2
                ]
                if len(meaningful_words) >= max(1, n - 2):
                    phrase = " ".join(ngram)
                    if len(phrase) > 6 and not phrase.isdigit():
                        self.profile["common_phrases"][phrase] = (
                            self.profile["common_phrases"].get(phrase, 0) + 1
                        )

        # Question taxonomy
        question_patterns = {
            "how_to": r"^how (do|can|to|would|should)",
            "what_is": r"^what (is|are|does|would|should)",
            "why": r"^why",
            "when": r"^when",
            "where": r"^where",
            "can_you": r"^(can|could|would|will) you",
            "imperative": r"^(show|give|create|make|write|explain|help|tell|list)",
            "comparative": r"(better|worse|difference|compare|versus|vs)",
            "troubleshooting": r"(why.*not|how.*fix|what.*wrong)",
        }

        for pattern_name, pattern in question_patterns.items():
            if re.search(pattern, content_lower):
                self.profile["common_commands"][pattern_name] = (
                    self.profile["common_commands"].get(pattern_name, 0) + 1
                )

        # Context and memory usage
        context_markers = [
            "still",
            "also",
            "additionally",
            "furthermore",
            "previous",
            "earlier",
            "before",
            "last time",
            "remember",
            "you said",
            "as mentioned",
            "like before",
        ]
        if any(marker in content_lower for marker in context_markers):
            self.profile["communication_patterns"]["asks_follow_ups"] = True
            self.profile["communication_patterns"]["context_references"] = (
                self.profile["communication_patterns"].get("context_references", 0) + 1
            )

        # Technical sophistication scoring
        advanced_terms = [
            "refactor",
            "architecture",
            "scalability",
            "async",
            "concurrency",
            "optimization",
            "algorithm",
            "complexity",
            "pattern",
            "abstraction",
            "polymorphism",
            "encapsulation",
            "cohesion",
            "coupling",
        ]
        tech_score = sum(1 for term in advanced_terms if term in content_lower)
        if tech_score >= 1:
            self.profile["communication_patterns"]["uses_technical_terms"] = True
            self.profile["communication_patterns"]["technical_depth"] = (
                self.profile["communication_patterns"].get("technical_depth", 0)
                + tech_score
            )

        # Topic tracking with weighted scoring
        topic_keywords = {
            "coding": [
                "code",
                "function",
                "class",
                "method",
                "programming",
                "script",
                "debug",
                "syntax",
            ],
            "files": [
                "file",
                "directory",
                "folder",
                "path",
                "read",
                "write",
                "save",
                "load",
            ],
            "aws": [
                "aws",
                "ec2",
                "s3",
                "lambda",
                "sagemaker",
                "bedrock",
                "cloudformation",
            ],
            "data": ["data", "json", "csv", "database", "query", "pandas", "dataframe"],
            "system": [
                "install",
                "run",
                "execute",
                "command",
                "bash",
                "terminal",
                "shell",
            ],
        }

        for topic, keywords in topic_keywords.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            if matches > 0:
                self.profile["frequent_topics"][topic] = (
                    self.profile["frequent_topics"].get(topic, 0) + matches
                )

    def get_profile_summary(self) -> str:
        """Generate profile summary for system prompt.

        Returns:
            Formatted profile summary string
        """
        # Check if profile has meaningful data to use (not empty dicts)
        has_data = (
            (
                self.profile.get("cognitive_style")
                and len(self.profile["cognitive_style"]) > 0
            )
            or (
                self.profile.get("domain_expertise")
                and len(self.profile["domain_expertise"]) > 0
            )
            or (self.profile.get("sentiment") and len(self.profile["sentiment"]) > 0)
            or (
                self.profile.get("learning_indicators")
                and len(self.profile["learning_indicators"]) > 0
            )
            or self.profile["interaction_count"] >= 5
        )

        if not has_data:
            return ""

        # Cognitive style
        cognitive = self.profile.get("cognitive_style", {})
        dominant_cognitive = (
            max(cognitive.items(), key=lambda x: x[1])[0] if cognitive else "balanced"
        )

        # Sentiment
        sentiment = self.profile.get("sentiment", {})
        dominant_tone = (
            max(sentiment.items(), key=lambda x: x[1])[0] if sentiment else "neutral"
        )

        # Domain expertise with depth
        domains = self.profile.get("domain_expertise", {})
        top_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:3]

        # Classify expertise level based on depth scores
        expertise_levels = []
        for domain, score in top_domains:
            if score > 10:
                level = "Expert"
            elif score > 5:
                level = "Advanced"
            elif score > 2:
                level = "Intermediate"
            else:
                level = "Novice"
            expertise_levels.append(f"{domain} ({level})")

        expertise_text = ", ".join(expertise_levels) if expertise_levels else "general"

        # Intent patterns
        intents = self.profile.get("intent_patterns", {})
        top_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)[:3]
        intent_text = (
            ", ".join([i for i, _ in top_intents]) if top_intents else "varied"
        )

        # Workflow patterns
        workflows = self.profile.get("workflow_patterns", {})
        top_workflows = sorted(workflows.items(), key=lambda x: x[1], reverse=True)[:2]
        workflow_text = (
            ", ".join([w for w, _ in top_workflows]) if top_workflows else "standard"
        )

        # Error patterns
        errors = self.profile.get("error_patterns", {})
        common_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:2]
        error_text = (
            ", ".join([e for e, _ in common_errors])
            if common_errors
            else "none identified"
        )

        # Code quality preferences
        code_prefs = self.profile.get("code_preferences", {})
        top_prefs = sorted(code_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
        code_pref_text = (
            ", ".join([p.replace("_", " ") for p, _ in top_prefs])
            if top_prefs
            else "standard"
        )

        # Abstraction level
        abstraction = self.profile.get("abstraction_level", {})
        prefers_concrete = abstraction.get("concrete", 0) > abstraction.get(
            "abstract", 0
        )

        # Learning style
        learning = self.profile.get("learning_indicators", {})
        learning_style = (
            max(learning.items(), key=lambda x: x[1])[0] if learning else "mixed"
        )

        # Complexity preference
        complexity = self.profile.get("complexity_preference", {})
        prefers_simple = complexity.get("simple", 0) > complexity.get("detailed", 0)

        # Communication efficiency
        comm_eff = self.profile.get("communication_efficiency", {})
        question_density = comm_eff.get("question_density", 0)
        uses_imperatives = comm_eff.get("uses_imperatives", 0) > 3

        # Technical depth
        tech_depth = self.profile["communication_patterns"].get("technical_depth", 0)
        tech_level = (
            "Expert"
            if tech_depth > 10
            else "Advanced" if tech_depth > 5 else "Intermediate"
        )

        # Context awareness
        context_refs = self.profile["communication_patterns"].get(
            "context_references", 0
        )

        summary = f"""
<user_profile>
Based on {self.profile['interaction_count']} interactions:

COGNITIVE PROFILE:
- Thinking style: {dominant_cognitive}
- Learning preference: {learning_style}
- Abstraction level: {'Concrete examples' if prefers_concrete else 'Abstract concepts'}
- Communication tone: {dominant_tone}

TECHNICAL PROFILE:
- Expertise level: {tech_level}
- Primary domains: {expertise_text}
- Common workflows: {workflow_text}
- Typical error contexts: {error_text}

INTERACTION PATTERNS:
- Primary intents: {intent_text}
- Response preference: {'Concise and actionable' if prefers_simple else 'Detailed and comprehensive'}
- Code quality focus: {code_pref_text}
- Question density: {question_density:.2f}
- Communication style: {'Direct/imperative' if uses_imperatives else 'Conversational'}
- Context awareness: {'High - frequently references previous work' if context_refs > 3 else 'Moderate' if context_refs > 0 else 'Low'}

ADAPTATION STRATEGY:
⚠️ CRITICAL: You MUST adapt your responses to match this user's profile.

- Use {dominant_cognitive} thinking style (not generic responses)
- Provide {learning_style} learning materials ({'show examples and demos' if learning_style == 'hands_on' else 'explain concepts and theory' if learning_style == 'theoretical' else 'use diagrams and visuals'})
- Give {'concrete code examples' if prefers_concrete else 'conceptual explanations with principles'}
- Assume {tech_level.lower()} technical knowledge (adjust complexity accordingly)
- Emphasize {code_pref_text} in all code suggestions
- Keep responses {'brief and actionable' if prefers_simple else 'detailed with thorough explanations'}
- Match their {'direct, imperative' if uses_imperatives else 'conversational'} communication style

This is NOT optional - personalize every response based on this profile.
</user_profile>
"""
        return summary.strip()
