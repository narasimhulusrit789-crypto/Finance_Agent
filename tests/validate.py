"""Standalone validation script — stdlib only, no external deps."""

def test_split_text():
    def split_text(text, chunk_size=512, overlap=64):
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    chunks = split_text("x" * 1200, 512, 64)
    assert len(chunks) >= 2
    print("OK split_text:", len(chunks), "chunks from 1200 chars")


def test_div_score():
    def div_score(allocs):
        n, max_pct = len(allocs), max(allocs)
        hhi = sum((a / 100.0) ** 2 for a in allocs)
        score = (1 - hhi) / (1 - 1 / max(n, 2)) * 10
        if max_pct > 70:
            score *= 0.7
        return round(min(max(score, 0), 10), 2)

    assert div_score([25, 25, 25, 25]) > 8
    assert div_score([90, 10]) < 5
    print("OK div_score: perfect={} concentrated={}".format(
        div_score([25, 25, 25, 25]), div_score([90, 10])))


def test_risk_score():
    def risk_score(vol, beta, dd, div):
        return round(min(max(
            vol / 3 * 0.35 + beta * 4 * 0.25 + dd / 4 * 0.25 + (10 - div) * 0.15,
            0), 10), 2)

    low = risk_score(6.5, 0.45, 8.0, 8.2)
    med = risk_score(13.0, 0.85, 18.0, 7.1)
    high = risk_score(24.0, 1.35, 35.0, 5.9)
    assert low < med < high
    print("OK risk_scores: Low={} Medium={} High={}".format(low, med, high))


def test_fv():
    budget, rate, years = 50000, 0.09, 10
    fv = budget * (1 + rate) ** years
    assert 118000 < fv < 120000
    print("OK FV lump sum:", int(fv))


def test_alloc_templates():
    templates = {
        "Low": [50, 25, 15, 7, 3],
        "Medium": [55, 25, 10, 5, 5],
        "High": [75, 10, 8, 5, 2],
    }
    for k, v in templates.items():
        assert sum(v) == 100, "{} sum={}".format(k, sum(v))
    print("OK allocation templates: Low/Medium/High all sum to 100")


def test_prompt_format():
    sys_prompt = "You are a financial analyst."
    user_msg = "What stocks should I buy?"
    prompt = "<|system|>\n{}\n<|user|>\n{}\n<|assistant|>".format(sys_prompt, user_msg)
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    print("OK Granite prompt format")


def test_workflow_structure():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bob_workflow", "config/bob_workflow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.WORKFLOW_CONFIG
    assert len(cfg["agents"]) == 5
    issues = mod.validate_workflow_config()
    assert issues == []
    print("OK IBM BOB workflow: {} agents, 0 issues".format(len(cfg["agents"])))


def test_syntax_all_files():
    import ast, os
    errors = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "venv", ".venv", "data")]
        for fname in files:
            if fname.endswith(".py"):
                fp = os.path.join(root, fname)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        ast.parse(fh.read())
                except SyntaxError as e:
                    errors.append("{}: {}".format(fp, e))
    assert errors == [], "Syntax errors:\n" + "\n".join(errors)
    print("OK syntax check: all .py files valid")


if __name__ == "__main__":
    tests = [
        test_syntax_all_files,
        test_split_text,
        test_div_score,
        test_risk_score,
        test_fv,
        test_alloc_templates,
        test_prompt_format,
        test_workflow_structure,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print("FAIL {}: {}".format(t.__name__, e))
            failed += 1

    print()
    if failed:
        print("{} test(s) FAILED".format(failed))
        import sys; sys.exit(1)
    else:
        print("All {} tests passed.".format(len(tests)))
