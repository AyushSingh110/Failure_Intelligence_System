"""
AC-6, AC-9: the boundary is checked in the source, so it cannot regress quietly.

These tests read the code, not its behaviour. They fail when a route reaches
around the scoped store, takes a tenant from a request, calls a tenant-sensitive
helper without a scope, or when a new piece of shared mutable state appears
without having been classified.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROUTE_MODULES = sorted((REPO / "app" / "routes").glob("*.py")) + [REPO / "app" / "auth_routes.py"]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _functions(tree: ast.Module):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[-1])
    return out


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


# ── Routes do not reach around the scoped store ───────────────────────────────

FORBIDDEN_IN_ROUTES = {
    "_db": "raw database handle",
    "_fallback_mode": "raw database state",
    "_fallback_records": "raw in-memory store",
    "_get_collection": "raw collection accessor",
    "get_signal_logs_collection": "raw collection accessor",
    "archetype_registry": "process-global cluster registry",
    "evolution_tracker": "process-global trend tracker",
    "add_confirmed_detection": "growth of the shared attack-pattern index",
    "add_pattern": "growth of the shared attack-pattern index",
    "maybe_recalibrate": "threshold change from a request path",
    "recalibrate": "threshold change from a request path",
    "maybe_trigger_retrain": "retraining from a request path",
    "resolve_user": "optional authentication",
    "require_user": "superseded authentication helper",
    "require_admin": "admin check that trusted the token",
    "verify_session_token": "credential parsing outside the dependency",
    "get_user_by_api_key": "credential parsing outside the dependency",
}


def test_route_modules_do_not_touch_raw_or_global_state():
    offenders = []
    for path in ROUTE_MODULES:
        used = _names(_tree(path))
        offenders += [f"{path.name}: {name} ({why})" for name, why in FORBIDDEN_IN_ROUTES.items() if name in used]
    assert offenders == []


def test_routes_never_take_the_tenant_from_a_request():
    """`.tenant_id` may be read from the principal or the scope, and nowhere else."""
    allowed_owners = {"principal", "scope"}
    # Functions that read `tenant_id` from something other than the principal, and why that is not
    # "taking the tenant from a request":
    exempt = {
        ("inference.py", "_enforce_body_tenant"),   # compares the body value with the principal's, then refuses
        ("auth_routes.py", "google_callback"),      # returns the account it has just created or loaded
        ("auth_routes.py", "get_me"),               # returns the caller's own account, checked against the principal
    }
    offenders = []
    for path in ROUTE_MODULES:
        for func in _functions(_tree(path)):
            if (path.name, func.name) in exempt:
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == "tenant_id":
                    owner = node.value.id if isinstance(node.value, ast.Name) else ast.dump(node.value)[:40]
                    if owner not in allowed_owners:
                        offenders.append(f"{path.name}:{node.lineno} {owner}.tenant_id in {func.name}()")
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                        and node.slice.value == "tenant_id":
                    offenders.append(f"{path.name}:{node.lineno} [...]['tenant_id'] in {func.name}()")
    assert offenders == []


SCOPE_KEYWORD = {
    "lookup_cache": "scope",
    "save_to_cache": "scope",
    "run_ground_truth_pipeline": "scope",
    "fan_out": "cache_scope",
    "fan_out_with_confidence": "cache_scope",
    "_call_single_model": "cache_scope",
    "get_context": "tenant_id",
    "store_turn": "tenant_id",
    "check_multi_turn_escalation": "tenant_id",
    "check_model_extraction": "tenant_id",
    "add_to_buffer": "tenant_id",
    "run_diagnostic": "registry",
    "run_full": "registry",
}


def test_tenant_sensitive_helpers_are_always_called_with_a_scope():
    offenders = []
    for path in ROUTE_MODULES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and _call_name(node) in SCOPE_KEYWORD:
                wanted = SCOPE_KEYWORD[_call_name(node)]
                if wanted not in {k.arg for k in node.keywords}:
                    offenders.append(f"{path.name}:{node.lineno} {_call_name(node)}() without {wanted}=")
    assert offenders == []


def test_cross_tenant_functions_are_reached_only_through_the_admin_check():
    offenders = []
    for path in ROUTE_MODULES:
        for func in _functions(_tree(path)):
            calls = {_call_name(n) for n in ast.walk(func) if isinstance(n, ast.Call)}
            if not any(name.endswith("_all_tenants") for name in calls):
                continue
            guarded = "require_platform_admin" in _names(func.args) or "authorize_cross_tenant_read" in calls
            if not guarded:
                offenders.append(f"{path.name}: {func.name}()")
    assert offenders == []


def test_unscoped_storage_functions_are_named_as_such():
    """Every public storage function either takes a tenant or says `all_tenants` in its name."""
    offenders = []
    for path, exempt in ((REPO / "storage" / "database.py", {"initialize_vault", "get_db", "flush_vault",
                                                              "request_indexes"}),
                         (REPO / "storage" / "signal_logger.py", set())):
        for node in _tree(path).body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_") or node.name in exempt:
                continue
            params = {a.arg for a in node.args.args + node.args.kwonlyargs}
            takes_tenant = "tenant_id" in params or "data" in params or "feedback_doc" in params
            if not takes_tenant and not node.name.endswith("_all_tenants"):
                offenders.append(f"{path.name}: {node.name}({', '.join(sorted(params))})")
    assert offenders == []


def test_scoped_store_is_the_only_importer_of_tenant_storage_in_routes():
    offenders = []
    for path in ROUTE_MODULES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module in ("storage.database", "storage.signal_logger"):
                offenders.append(f"{path.name}:{node.lineno} imports {node.module}")
    assert offenders == []


# ── No unclassified shared mutable state ──────────────────────────────────────

# Module-level mutable containers in app/, engine/ and storage/, with their class
# from PLAN_002 §18. A new one fails the test until someone classifies it here.
CLASSIFIED_STATE = {
    "app/notifications.py:_spike_last_sent":            "TENANT SCOPED (keyed by tenant reference)",
    "storage/database.py:_fallback_records":            "TENANT SCOPED (key starts with the tenant)",
    "engine/session_store.py:_fallback":                "TENANT SCOPED (key is (tenant, session))",
    "engine/session_store.py:_fallback_summaries":      "TENANT SCOPED (key is (tenant, session))",
    "engine/groq_service.py:_response_cache":           "TENANT SCOPED (scope is part of every key; unused without one)",
    "engine/model_extraction_tracker.py:_memory_store": "TENANT SCOPED (keyed by tenant)",
    "engine/fie_config.py:_thresholds":                 "GLOBAL ADMIN CONTROLLED",
    "engine/fie_config.py:_attack_thresholds":          "GLOBAL ADMIN CONTROLLED",
    "engine/canary_tracker.py:_canary_store":           "MUST NOT EXIST (never written; left in place)",
    "engine/demo_feedback.py:_seen_hashes":             "GLOBAL BY DESIGN (public reports, no tenant data)",
}

_CONTAINER_CALLS = {"dict", "list", "set", "defaultdict", "OrderedDict", "deque", "Counter"}


def _module_level_containers() -> set[str]:
    found = set()
    for top in ("app", "engine", "storage"):
        for path in sorted((REPO / top).rglob("*.py")):
            for node in _tree(path).body:
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets, value = [node.target], node.value
                else:
                    continue
                mutable = isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)) \
                    or (isinstance(value, ast.Call) and _call_name(value) in _CONTAINER_CALLS)
                if not mutable:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and not target.id.isupper() and target.id != "__all__":
                        found.add(f"{path.relative_to(REPO).as_posix()}:{target.id}")
    return found


def test_every_shared_mutable_container_is_classified():
    found = _module_level_containers()
    assert found - set(CLASSIFIED_STATE) == set(), "unclassified module-level mutable state"
    assert set(CLASSIFIED_STATE) - found == set(), "classified state that no longer exists"


def test_per_tenant_state_is_not_a_module_singleton_in_routes():
    """Clusters and trend are reached through the per-tenant holder, never through the old singletons."""
    for path in ROUTE_MODULES:
        source = path.read_text(encoding="utf-8")
        assert "engine.archetypes.clustering" not in source, path.name
        assert "engine.evolution.tracker" not in source, path.name


def test_no_hidden_tenant_context():
    """Scope travels as an argument. No context variable or thread-local carries a tenant."""
    offenders = []
    for top in ("app", "engine", "storage"):
        for path in sorted((REPO / top).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "threading.local" in source:
                offenders.append(f"{path.name}: threading.local")
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Call) and _call_name(node) == "ContextVar":
                    label = ast.unparse(node)
                    if "tenant" in label.lower() or "principal" in label.lower() or "scope" in label.lower():
                        offenders.append(f"{path.name}: {label}")
    assert offenders == []


def test_no_switch_reopens_anonymous_access():
    for path in ROUTE_MODULES + [REPO / "app" / "auth_guard.py", REPO / "app" / "main.py"]:
        source = path.read_text(encoding="utf-8").upper()
        for needle in ("ALLOW_ANONYMOUS", "ANONYMOUS_MONITOR", "DISABLE_AUTH", "AUTH_DISABLED", "SKIP_AUTH"):
            assert needle not in source, f"{path.name}: {needle}"
    assert '"anonymous"' not in (REPO / "app" / "routes" / "monitor.py").read_text(encoding="utf-8")
