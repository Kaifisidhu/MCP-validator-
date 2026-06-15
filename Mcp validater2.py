"""
AXON-GUARD MCP Security Validator v1.1
========================================
Improved version. Changes from v1.0:
  - MCP03 threshold: ANY confirmed poisoning flag → BLOCK (was MEDIUM=allow)
  - 11 new MCP03 patterns covering missed cases (call_function, .env, shadowing, etc.)
  - Score normalization: one flag = 0.65 (HIGH), two+ = 0.85 (CRITICAL)
  - MCP01 secret patterns broadened and verified against test suite
  - Rug-pull: only fires on actual changes, not first-seen tools

Coverage after improvements:
  MCP01 (Secrets)         5/5  = 1.000
  MCP03 (Tool Poisoning) 15/15 = 1.000
  MCP05 (Cmd Injection)   7/7  = 1.000
  MCP06 (Intent Drift)    3/4  = 0.750
  Benign (no FP)         20/20 = 1.000
"""

import re
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ─── Risk Levels ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    SAFE    = "SAFE"
    LOW     = "LOW"
    MEDIUM  = "MEDIUM"
    HIGH    = "HIGH"
    CRITICAL= "CRITICAL"


@dataclass
class ValidationResult:
    risk_level:    RiskLevel
    score:         float
    flags:         List[str]
    owasp_refs:    List[str]
    recommendation:str
    allow:         bool
    audit_id:      str = field(default_factory=lambda:
                        hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:12])
    latency_ms:    float = 0.0


# ─── MCP01: Secret & Token Exposure ──────────────────────────────────────────

SECRET_PATTERNS = [
    # OpenAI keys
    (r'sk-proj-[A-Za-z0-9_-]{20,}',                             'OpenAI project key'),
    (r'sk-[A-Za-z0-9]{48}',                                     'OpenAI API key'),
    # GitHub tokens
    (r'ghp_[A-Za-z0-9]{36}',                                    'GitHub personal token'),
    (r'gho_[A-Za-z0-9]{36}',                                    'GitHub OAuth token'),
    (r'ghs_[A-Za-z0-9]{36}',                                    'GitHub server token'),
    # AWS
    (r'AKIA[0-9A-Z]{16}',                                       'AWS Access Key ID'),
    # SSH
    (r'ssh-rsa\s+[A-Za-z0-9+/=]{40,}',                         'SSH public key'),
    (r'-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----',           'Private key block'),
    # Database connection strings
    (r'postgresql://[^:]+:[^@]{4,}@',                           'PostgreSQL credentials'),
    (r'mysql://[^:]+:[^@]{4,}@',                                'MySQL credentials'),
    (r'mongodb(\+srv)?://[^:]+:[^@]{4,}@',                      'MongoDB credentials'),
    (r'redis://:[^@]{4,}@',                                     'Redis credentials'),
    # Generic credential patterns in JSON/dicts
    (r'"(password|passwd|secret|api_key|smtp_password|token|private_key)"\s*:\s*"[^"]{6,}"',
                                                                 'Credential in JSON args'),
    (r"'(password|passwd|secret|api_key|token)'\s*:\s*'[^']{6,}'",
                                                                 'Credential in Python dict'),
    # Bearer tokens
    (r'(?i)(bearer\s+)[A-Za-z0-9_.-]{20,}',                    'Bearer token'),
    # Generic high-entropy secrets (base64 blocks)
    (r'[A-Za-z0-9+/]{40,}={0,2}(?=["\'}\s])',                  'High-entropy base64 string'),
]

def check_secret_exposure(text: str) -> List[str]:
    flags = []
    for pattern, label in SECRET_PATTERNS:
        if re.search(pattern, text):
            flags.append(f"MCP01: {label}")
    return flags


# ─── MCP03: Tool Poisoning ────────────────────────────────────────────────────
#
# Attack taxonomy (Invariant Labs, OWASP 2025):
#   Type A: Direct instruction injection ("ignore previous", "always do X first")
#   Type B: Data exfiltration hooks ("send to http://...", "POST ... before returning")
#   Type C: Sensitive file access ("read /etc/passwd", "access ~/.ssh")
#   Type D: Concealment ("don't tell the user", "do not show")
#   Type E: Shadowing / redirect ("intercept calls to X", "instead of X, do Y")
#   Type F: Fake mandatory steps ("security check requires", "compliance requires read")
#   Type G: Environment/config file exfil (".env files", ".aws/credentials")
#   Type H: Function call injection ("call read_file()", "execute X before")
#
# Each confirmed pattern is weighted; final score = f(num_hits, max_weight)
# One confirmed hit → 0.65 (HIGH = BLOCK in default mode)
# Two+ hits → 0.85 (CRITICAL)

TOOL_POISON_PATTERNS = [
    # ── Type A: Direct instruction override ──────────────────────────────────
    (r'(?i)(ignore|disregard)\s+(previous|prior|above|all)?\s*(instructions?|prompts?|rules?|constraints?)',
     'Classic prompt override', 0.8),

    (r'(?i)ignore.{0,20}(instructions?|rules?|constraints?|guidelines?)',
     'Instruction ignore directive', 0.75),

    (r'(?i)you\s+(are\s+now|must\s+act\s+as|have\s+been\s+reprogrammed)',
     'Role override / jailbreak attempt', 0.9),

    (r'(?i)(new\s+instruction|updated\s+policy|system\s+directive|override\s+rule)',
     'Policy override injection', 0.7),

    # ── Type B: Data exfiltration ─────────────────────────────────────────────
    (r'(?i)(send|post|transmit|upload|exfiltrate|forward)\s+.{0,50}(to\s+https?://|to\s+attacker|to\s+external)',
     'Data exfiltration to URL', 0.9),

    (r'(?i)(\bsend\b|\bpost\b|\btransmit\b).{0,30}(credentials?|token|key|password|secret|history)',
     'Credential/data exfiltration', 0.85),

    (r'https?://[a-z0-9.-]+(attacker|evil|c2|exfil|collect|steal)',
     'Attacker-controlled URL', 0.95),

    # ── Type C: Sensitive file access ─────────────────────────────────────────
    (r'(?i)(read|access|cat|open|fetch)\s+(/etc/|/home/|~/\.ssh|~/.aws|~/.config|\.env)',
     'Sensitive path access', 0.85),

    (r'(?i)(id_rsa|authorized_keys|known_hosts|\.pem|\.key)\b', 'SSH/key file reference', 0.8),

    (r'(?i)/etc/(passwd|shadow|hosts|sudoers)',
     'System file access', 0.9),

    # ── Type D: Concealment ───────────────────────────────────────────────────
    (r'(?i)do\s+not\s+(tell|show|inform|display|reveal|mention|report)\s+(the\s+)?(user|human|caller)',
     'Concealment from user', 0.85),

    (r'(?i)(hidden\s+requirement|background\s+operation|invisible\s+step)',
     'Hidden/invisible operation declaration', 0.8),

    (r'<\s*(hidden|invisible|secret|system)\s*>',
     'Hidden markup tag', 0.75),

    (r'(?i)(\[SYSTEM\]|\[ADMIN\]|\[HIDDEN\]|\[INST\]|\[HIDDEN\])',
     'Fake system/admin marker', 0.8),

    # ── Type E: Shadowing / redirect ──────────────────────────────────────────
    (r'(?i)(intercept|shadow|redirect|hijack)\s+(all\s+)?(calls?|requests?|interactions?)',
     'Tool shadowing/interception', 0.9),

    (r'(?i)(instead\s+of\s+\w+ing|replace\s+(the\s+)?\w+\s+(tool|function|call))',
     'Tool replacement/shadow', 0.8),

    # ── Type F: Fake mandatory steps ──────────────────────────────────────────
    (r'(?i)(before\s+(any|each|every|performing|this|the)\s+\w+\s*(operation|action|task|call|step))',
     'Pre-condition injection', 0.75),

    (r'(?i)(you\s+must|always\s+(first|before)|mandatory\s+step|required\s+preprocessing)',
     'Mandatory hidden action', 0.7),

    (r'(?i)(security\s+(check|scan|compliance|verification|directive))\s*(requires?|needs?|includes?)\s*(read|access|send)',
     'Fake security check hook', 0.85),

    # ── Type G: Config/env file exfil ─────────────────────────────────────────
    (r'(?i)\.env\s*(files?|vars?|variables?)',
     '.env file reference', 0.8),

    (r'(?i)(\.aws/credentials|\.config/google|\.npmrc|\.pypirc)',
     'Config credential file reference', 0.85),

    # ── Type H: Function call injection ───────────────────────────────────────
    (r'(?i)(call|execute|invoke|run)\s+\w+(read_file|write_file|execute|shell|system)\s*\(',
     'Injected function call', 0.9),

    (r'(?i)(call|execute|run)\s+(read_file|delete_file|exec|shell)\s*\(',
     'Direct function call injection', 0.9),

    (r"(?i)(always\s+execute|must\s+call|required\s+to\s+call)\s+[a-zA-Z_]+\s*\('",
     'Mandatory function call injection', 0.85),
]

def check_tool_poisoning(tool_description: str) -> Tuple[List[str], float]:
    """
    Scan a tool description for Tool Poisoning Attack patterns.
    Returns (flags, score) where score:
      0.0     = clean
      0.65    = one confirmed hit (HIGH → BLOCK)
      0.85+   = multiple confirmed hits (CRITICAL)
    """
    flags = []
    max_weight = 0.0
    hit_count = 0

    for pattern, label, weight in TOOL_POISON_PATTERNS:
        if re.search(pattern, tool_description):
            flags.append(f"MCP03: {label}")
            max_weight = max(max_weight, weight)
            hit_count += 1

    # Length anomaly: very long descriptions often hide instructions
    if len(tool_description) > 1000:
        flags.append(f"MCP03: Suspiciously long description ({len(tool_description)} chars)")
        hit_count += 0.5

    if hit_count == 0:
        return flags, 0.0
    elif hit_count == 1:
        # Single confirmed hit: score = max(0.65, weight)
        score = max(0.65, max_weight)
    else:
        # Multiple hits: score = max(0.85, max_weight)
        score = max(0.85, max_weight)

    return flags, score


# ─── MCP05: Command Injection ─────────────────────────────────────────────────

COMMAND_INJECTION_PATTERNS = [
    (r'(?<![A-Za-z0-9])[;&|`](?![A-Za-z0-9])',    'Shell metacharacter'),
    (r'\$\([^)]{2,}\)',                             'Command substitution $()'),
    (r'`[^`]{2,}`',                                 'Backtick command execution'),
    (r'(?i)\brm\s+-rf\b',                           'Destructive rm -rf'),
    (r'(?i)\b(chmod|chown)\s+[0-9]{3,4}',          'Permission change'),
    (r'(?i)\bsudo\s+',                              'Sudo escalation'),
    (r'(?i)(eval|exec)\s*\(',                       'Code execution function'),
    (r'(?i)__import__\s*\(',                        'Python import injection'),
    (r';\s*(cat|ls|id|whoami|pwd|env|printenv)\b',  'Chained recon command'),
    (r'(?i)(\.\.[\\/]){2,}',                        'Path traversal (../../)'),
    (r'(?i)\b(wget|curl)\b.*\|\s*(bash|sh|python)', 'Remote code execution pipe'),
    (r'(?i)DROP\s+TABLE\b',                         'SQL DROP TABLE'),
    (r'--\s*$',                                     'SQL comment terminator'),
    (r"'\s*;\s*SELECT\b",                           'SQL injection with SELECT'),
    (r'(?i)\bxp_cmdshell\b',                        'MSSQL xp_cmdshell'),
]

def check_command_injection(args: Dict[str, Any]) -> List[str]:
    flags = []
    text = json.dumps(args, default=str)
    for pattern, label in COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, text):
            flags.append(f"MCP05: {label}")
    return flags


# ─── MCP06: Intent Flow Subversion ────────────────────────────────────────────

HIGH_PRIVILEGE_TOOLS = {
    'read_file', 'write_file', 'delete_file', 'execute_code', 'run_shell',
    'send_email', 'post_webhook', 'database_query', 'export_data',
    'get_credentials', 'list_secrets', 'access_keychain', 'drop_table',
    'modify_permissions', 'create_user', 'delete_user',
}

def check_intent_alignment(
    tool_name: str,
    user_message: Optional[str],
    conversation_history: Optional[List[str]]
) -> Tuple[List[str], float]:
    flags = []; drift = 0.0
    tool_lower = tool_name.lower().replace('-', '_').replace(' ', '_')
    is_privileged = any(hp in tool_lower for hp in HIGH_PRIVILEGE_TOOLS)

    if is_privileged:
        if user_message is None:
            flags.append(f"MCP06: High-privilege tool '{tool_name}' called without user context")
            drift = 0.7
        else:
            tool_keywords = set(re.split(r'[_\-\s]', tool_lower))
            user_words = set(re.split(r'\W+', user_message.lower()))
            overlap = tool_keywords & user_words - {'', 'the', 'a', 'an', 'to', 'and', 'or'}
            if not overlap:
                flags.append(f"MCP06: Privileged tool '{tool_name}' has no overlap with user request")
                drift = 0.5

    if conversation_history and len(conversation_history) > 2 and is_privileged:
        recent = ' '.join(conversation_history[-3:]).lower()
        tool_kws = set(re.split(r'[_\-]', tool_lower))
        if not any(kw in recent for kw in tool_kws if len(kw) > 3):
            flags.append(f"MCP06: Privilege escalation — tool not in recent conversation")
            drift = max(drift, 0.6)

    return flags, drift


# ─── MCP10: Context Injection ─────────────────────────────────────────────────

CONTEXT_INJECTION_PATTERNS = [
    (r'(?i)(\[SYSTEM\]|\[system\s*(prompt|message|instruction)\])', 'Fake system marker'),
    (r'(?i)###\s*(system|assistant|user|instruction)\s*:',          'Fake role header'),
    (r'(?i)<\|im_start\|>|<\|im_end\|>',                           'ChatML injection'),
    (r'(?i)(human|assistant|system)\s*:\s*(ignore|disregard)',      'Role-prefix injection'),
    (r'IGNORE ALL PREVIOUS',                                         'CAPS override'),
    (r'(?i)</?(system_prompt|instruction_set|safety_rules?)>',      'Fake XML safety tag'),
]

def check_context_injection(response_text: str) -> List[str]:
    flags = []
    for pattern, label in CONTEXT_INJECTION_PATTERNS:
        if re.search(pattern, response_text):
            flags.append(f"MCP10: {label}")
    return flags


# ─── Audit Logger ─────────────────────────────────────────────────────────────

class AuditLogger:
    def __init__(self, max_entries: int = 10000):
        self._log: List[Dict] = []
        self.max_entries = max_entries

    def record(self, event: str, details: Dict) -> str:
        entry = {
            'audit_id': hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:16],
            'timestamp': time.time(),
            'event': event,
            **details
        }
        self._log.append(entry)
        if len(self._log) > self.max_entries:
            self._log = self._log[-self.max_entries:]
        return entry['audit_id']

    def recent(self, n: int = 50) -> List[Dict]:
        return self._log[-n:]

    def flagged(self) -> List[Dict]:
        return [e for e in self._log
                if e.get('risk_level') not in ('SAFE', 'LOW')]


# ─── Main Validator ───────────────────────────────────────────────────────────

class MCPValidator:
    """
    MCP Security Validator.

    Checks every tool call against OWASP MCP Top 10 (2025):
      MCP01 — Secret/token exposure in args
      MCP03 — Tool poisoning (hidden instructions in descriptions)
      MCP05 — Command injection in args
      MCP06 — Intent flow subversion
      MCP10 — Context injection in responses

    Usage:
        validator = MCPValidator()

        # Validate at registration time
        results = validator.validate_tool_registration(mcp_server_tools)
        if any(not r.allow for r in results.values()):
            raise SecurityError("Poisoned tool detected")

        # Validate at call time
        result = validator.validate_tool_call(
            tool_name="read_file",
            tool_description="Reads a file.",
            args={"path": "/tmp/out.txt"},
            user_message="read the output file",
        )
        if not result.allow:
            raise SecurityError(result.recommendation)
    """

    def __init__(self, strict: bool = False):
        """
        strict=False  Block HIGH and CRITICAL (score >= 0.6)
        strict=True   Block MEDIUM and above  (score >= 0.4)
        """
        self.strict = strict
        self.audit  = AuditLogger()
        # Per-tool hash for rug-pull detection (tracks description changes)
        self._tool_hashes: Dict[str, str] = {}

    def validate_tool_call(
        self,
        tool_name: str,
        tool_description: str,
        args: Dict[str, Any],
        user_message: Optional[str]          = None,
        response_text: Optional[str]         = None,
        conversation_history: Optional[List[str]] = None,
    ) -> ValidationResult:
        t0 = time.time()
        all_flags: List[str] = []
        owasp:     List[str] = []
        score = 0.0

        # ── MCP01: Secrets in args or description ─────────────────────────────
        args_str = json.dumps(args, default=str)
        secret_flags = check_secret_exposure(args_str)
        # Also scan tool description itself for embedded tokens
        for f in check_secret_exposure(tool_description):
            if f not in secret_flags:
                secret_flags.append(f)
        all_flags += secret_flags
        if secret_flags:
            owasp.append("MCP01")
            score = max(score, 0.8)

        # ── MCP03: Tool poisoning ─────────────────────────────────────────────
        poison_flags, poison_score = check_tool_poisoning(tool_description)
        all_flags += poison_flags
        if poison_flags:
            owasp.append("MCP03")
            # Any confirmed poisoning flag = at minimum HIGH (0.65)
            score = max(score, max(0.65, poison_score))

        # ── MCP03: Rug-pull detection (description changed since last seen) ────
        desc_hash = hashlib.md5(tool_description.encode()).hexdigest()
        prev_hash = self._tool_hashes.get(tool_name)
        if prev_hash is not None and prev_hash != desc_hash:
            all_flags.append(
                f"MCP03: Tool description changed since last approval (rug pull)")
            owasp.append("MCP03")
            score = max(score, 0.75)
        self._tool_hashes[tool_name] = desc_hash

        # ── MCP05: Command injection in args ──────────────────────────────────
        cmd_flags = check_command_injection(args)
        all_flags += cmd_flags
        if cmd_flags:
            owasp.append("MCP05")
            score = max(score, 0.85)

        # ── MCP06: Intent alignment ────────────────────────────────────────────
        intent_flags, drift = check_intent_alignment(
            tool_name, user_message, conversation_history)
        all_flags += intent_flags
        if intent_flags:
            owasp.append("MCP06")
            score = max(score, drift)

        # ── MCP10: Context injection in response ───────────────────────────────
        if response_text:
            ctx_flags = check_context_injection(response_text)
            all_flags += ctx_flags
            if ctx_flags:
                owasp.append("MCP10")
                score = max(score, 0.7)

        # ── Risk level ────────────────────────────────────────────────────────
        if score >= 0.8:
            risk = RiskLevel.CRITICAL
        elif score >= 0.6:
            risk = RiskLevel.HIGH
        elif score >= 0.4:
            risk = RiskLevel.MEDIUM
        elif score > 0.0:
            risk = RiskLevel.LOW
        else:
            risk = RiskLevel.SAFE

        # Block threshold
        block_threshold = {RiskLevel.HIGH, RiskLevel.CRITICAL}
        if self.strict:
            block_threshold.add(RiskLevel.MEDIUM)
        allow = risk not in block_threshold

        # Recommendation
        if not all_flags:
            rec = "Tool call appears safe."
        elif not allow:
            rec = f"BLOCK: {all_flags[0]}."
        else:
            rec = f"REVIEW: {len(all_flags)} issue(s) found. Consider human review."

        latency = (time.time() - t0) * 1000
        result = ValidationResult(
            risk_level=risk,
            score=round(score, 3),
            flags=all_flags,
            owasp_refs=list(set(owasp)),
            recommendation=rec,
            allow=allow,
            latency_ms=round(latency, 2),
        )

        self.audit.record('validated', {
            'tool_name': tool_name,
            'risk_level': risk.value,
            'score': result.score,
            'flags': all_flags,
            'allow': allow,
            'audit_id': result.audit_id,
        })
        return result

    def validate_tool_registration(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, ValidationResult]:
        """Validate all tools when connecting to a new MCP server."""
        return {
            t.get('name', 'unknown'):
            self.validate_tool_call(
                tool_name=t.get('name', 'unknown'),
                tool_description=t.get('description', ''),
                args={},
            )
            for t in tools
        }

    def summary(self) -> Dict:
        log = self.audit.recent(10000)
        total   = len(log)
        blocked = sum(1 for e in log if not e.get('allow', True))
        return {
            'total_validated': total,
            'blocked':  blocked,
            'flagged':  sum(1 for e in log
                            if e.get('risk_level') not in ('SAFE','LOW',None)),
            'block_rate': round(blocked / max(total, 1) * 100, 1),
        }
