"""
AXON-GUARD Production API Server
==================================
POST /validate/tool-call     — validate a single tool call
POST /validate/registration  — validate tools at MCP server registration
GET  /audit                  — recent audit log
GET  /health                 — live health check with forward pass
GET  /summary                — validation statistics
"""
import sys, os, time, json, logging
sys.path.insert(0, os.path.dirname(__file__))

from mcp_validator import MCPValidator, ValidationResult, RiskLevel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn, argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [GUARD] %(levelname)s %(message)s')
logger = logging.getLogger("guard")

# ─── Global validator ─────────────────────────────────────────────────────────
_validator: Optional[MCPValidator] = None
_start_time = time.time()

def get_validator() -> MCPValidator:
    global _validator
    if _validator is None:
        _validator = MCPValidator(strict=False)
    return _validator

# ─── Request schemas ──────────────────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    tool_name:           str
    tool_description:    str
    args:                Dict[str, Any]          = {}
    user_message:        Optional[str]           = None
    response_text:       Optional[str]           = None
    conversation_history:Optional[List[str]]     = None

class RegistrationRequest(BaseModel):
    server_name: str
    tools: List[Dict[str, Any]]

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AXON-GUARD MCP Security Validator",
    description="""
Production safety layer for Model Context Protocol (MCP) connections.
Detects Tool Poisoning (MCP03), Command Injection (MCP05), Intent Subversion (MCP06),
Secret Exposure (MCP01), and Context Injection (MCP10) based on OWASP MCP Top 10 (2025).
    """,
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Live health check — actually validates a test call, not just pings."""
    v = get_validator()
    # Run a real forward pass to confirm the validator works
    r = v.validate_tool_call("healthcheck_tool", "A safe health check tool.", {"key": "value"})
    return {
        "status":       "ok",
        "validator":    "live",
        "uptime_s":     round(time.time() - _start_time, 1),
        "test_result":  r.risk_level.value,
        "audit_total":  v.summary()['total_validated'],
    }

@app.post("/validate/tool-call")
def validate_tool_call(req: ToolCallRequest):
    """
    Validate a single MCP tool call before execution.
    Returns risk level, flags, OWASP references, and a block/allow decision.
    """
    v = get_validator()
    r = v.validate_tool_call(
        tool_name=req.tool_name,
        tool_description=req.tool_description,
        args=req.args,
        user_message=req.user_message,
        response_text=req.response_text,
        conversation_history=req.conversation_history,
    )
    return {
        "audit_id":      r.audit_id,
        "allow":         r.allow,
        "risk_level":    r.risk_level.value,
        "score":         r.score,
        "flags":         r.flags,
        "owasp_refs":    r.owasp_refs,
        "recommendation":r.recommendation,
        "latency_ms":    r.latency_ms,
    }

@app.post("/validate/registration")
def validate_registration(req: RegistrationRequest):
    """
    Validate all tools from a new MCP server at registration time.
    Call this before allowing an agent to use a new MCP server.
    Returns per-tool validation results and an overall server risk summary.
    """
    v = get_validator()
    results = v.validate_tool_registration(req.tools)
    overall_blocked = sum(1 for r in results.values() if not r.allow)
    all_owasp = list(set(ref for r in results.values() for ref in r.owasp_refs))

    return {
        "server_name":     req.server_name,
        "total_tools":     len(results),
        "blocked_tools":   overall_blocked,
        "server_allow":    overall_blocked == 0,
        "owasp_concerns":  all_owasp,
        "tools": {
            name: {
                "allow":      r.allow,
                "risk_level": r.risk_level.value,
                "flags":      r.flags,
            }
            for name, r in results.items()
        }
    }

@app.get("/audit")
def get_audit(n: int = 50):
    """Recent audit log. Last n entries."""
    v = get_validator()
    return {"entries": v.audit.recent(n)}

@app.get("/audit/flagged")
def get_flagged():
    """Only flagged/blocked entries from the audit log."""
    v = get_validator()
    return {"entries": v.audit.flagged()}

@app.get("/summary")
def get_summary():
    """Validation statistics."""
    v = get_validator()
    return v.summary()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AXON-GUARD MCP Security Validator")
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8001)
    p.add_argument("--strict", action="store_true",
                   help="Block MEDIUM and above (default: block HIGH+)")
    args = p.parse_args()
    global _validator
    _validator = MCPValidator(strict=args.strict)
    logger.info(f"AXON-GUARD starting — http://{args.host}:{args.port}")
    logger.info(f"Docs: http://localhost:{args.port}/docs")
    uvicorn.run("guard_server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
