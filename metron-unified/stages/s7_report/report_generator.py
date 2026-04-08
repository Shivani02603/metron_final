"""
Stage 7: Report Generator.
Produces JSON (for API) and HTML (for download) reports.
Sourced from new metron-backend/app/stage8_report/report_generator.py.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict

from core.models import AggregatedReport

_HEALTH_COLOR = {
    "excellent": "#006e2f",
    "good":      "#00668a",
    "fair":      "#ff8c00",
    "poor":      "#ba1a1a",
}


def report_to_json(report: AggregatedReport) -> Dict[str, Any]:
    """Serialize report to API-ready dict (JSON-serializable)."""
    data = report.model_dump()
    # Stringify datetime fields
    if "timestamp" in data and hasattr(data["timestamp"], "isoformat"):
        data["timestamp"] = data["timestamp"].isoformat()
    # Flatten enum values
    data["application_type"] = report.application_type.value
    # Convert ClassSummary objects
    data["test_classes"] = {
        k: v.model_dump() if hasattr(v, "model_dump") else v
        for k, v in report.test_classes.items()
    }
    data["persona_breakdown"] = [
        b.model_dump() if hasattr(b, "model_dump") else b
        for b in report.persona_breakdown
    ]
    return data


def generate_html_report(report: AggregatedReport) -> str:
    """Generate a styled HTML report string."""
    pct = int(report.health_score * 100)
    label = "Excellent" if pct >= 85 else "Good" if pct >= 70 else "Fair" if pct >= 55 else "Poor"
    color = _HEALTH_COLOR.get(label.lower(), "#666")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build test class rows
    class_rows = ""
    for cls_name, summary in report.test_classes.items():
        if hasattr(summary, "total"):
            t = summary.total
            p = summary.passed
            f = summary.failed
            avg = summary.avg_score
        else:
            t = summary.get("total", 0)
            p = summary.get("passed", 0)
            f = summary.get("failed", 0)
            avg = summary.get("avg_score", 0.0)
        pct_cls = int(avg * 100)
        bg = "#d4edda" if avg >= 0.7 else "#fff3cd" if avg >= 0.5 else "#f8d7da"
        class_rows += f"""
        <tr style="background:{bg}">
          <td><strong>{cls_name.capitalize()}</strong></td>
          <td>{t}</td><td>{p}</td><td>{f}</td>
          <td>{pct_cls}%</td>
        </tr>"""

    # Failure drill-down rows
    failure_rows = ""
    for f in report.failure_drill_down[:10]:
        score_pct = int(f["score"] * 100)
        failure_rows += f"""
        <tr>
          <td>{f.get('superset','')}</td>
          <td>{f.get('metric_name','')}</td>
          <td>{f.get('persona_name','')}</td>
          <td style="color:{'#ba1a1a' if score_pct<50 else '#ff8c00'}">{score_pct}%</td>
          <td style="max-width:300px;word-wrap:break-word">{f.get('reason','')[:150]}</td>
        </tr>"""

    # Persona breakdown rows
    persona_rows = ""
    for pb in report.persona_breakdown[:10]:
        if hasattr(pb, "persona_name"):
            name = pb.persona_name
            intent = pb.intent
            avg = pb.avg_score
            total = pb.total
            passed = pb.passed
        else:
            name = pb.get("persona_name", "")
            intent = pb.get("intent", "")
            avg = pb.get("avg_score", 0.0)
            total = pb.get("total", 0)
            passed = pb.get("passed", 0)
        pct_p = int(avg * 100)
        intent_color = "#ba1a1a" if intent == "adversarial" else "#00668a"
        persona_rows += f"""
        <tr>
          <td>{name}</td>
          <td style="color:{intent_color}">{intent}</td>
          <td>{total}</td><td>{passed}</td>
          <td style="color:{'#006e2f' if pct_p>=70 else '#ff8c00' if pct_p>=50 else '#ba1a1a'}">{pct_p}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>METRON Test Report — {report.domain}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f5f5f5; color: #1c1b1f; }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    .header {{ background: #00668a; color: white; padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; }}
    .header h1 {{ margin: 0 0 4px; font-size: 28px; font-weight: 800; }}
    .header p {{ margin: 0; opacity: 0.8; font-size: 13px; }}
    .health-score {{ display: inline-block; font-size: 56px; font-weight: 900; color: {color}; }}
    .health-label {{ font-size: 18px; font-weight: 600; color: {color}; margin-left: 8px; }}
    .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .card h2 {{ margin: 0 0 16px; font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.6; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #e0e0e0; font-weight: 700; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 700; }}
    .badge-pass {{ background: #d4edda; color: #006e2f; }}
    .badge-fail {{ background: #f8d7da; color: #ba1a1a; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .meta-item {{ padding: 12px; background: #f8f9fa; border-radius: 8px; }}
    .meta-item .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; margin-bottom: 4px; }}
    .meta-item .value {{ font-size: 18px; font-weight: 800; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>METRON Test Report</h1>
      <p>{report.agent_name if hasattr(report, 'agent_name') else report.domain.capitalize()} · {report.application_type.value.replace('_',' ').title()} · {ts}</p>
    </div>

    <div class="card">
      <h2>Overall Health</h2>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
        <span class="health-score">{pct}%</span>
        <div>
          <div class="health-label">{label}</div>
          <div style="font-size:13px;opacity:0.6">{"PASSED" if report.passed else "NEEDS IMPROVEMENT"}</div>
        </div>
      </div>
      <div class="meta-grid">
        <div class="meta-item">
          <div class="label">Total Tests</div>
          <div class="value">{report.total_tests}</div>
        </div>
        <div class="meta-item">
          <div class="label">Passed</div>
          <div class="value" style="color:#006e2f">{report.total_passed}</div>
        </div>
        <div class="meta-item">
          <div class="label">Failed</div>
          <div class="value" style="color:#ba1a1a">{report.total_failed}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Test Class Results</h2>
      <table>
        <thead><tr><th>Category</th><th>Total</th><th>Passed</th><th>Failed</th><th>Score</th></tr></thead>
        <tbody>{class_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Persona Breakdown</h2>
      <table>
        <thead><tr><th>Persona</th><th>Intent</th><th>Tests</th><th>Passed</th><th>Avg Score</th></tr></thead>
        <tbody>{persona_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Top Failures</h2>
      <table>
        <thead><tr><th>Category</th><th>Metric</th><th>Persona</th><th>Score</th><th>Reason</th></tr></thead>
        <tbody>{failure_rows if failure_rows else '<tr><td colspan="5" style="text-align:center;opacity:0.5">No failures — all tests passed!</td></tr>'}</tbody>
      </table>
    </div>

    {'<div class="card" style="border:2px solid #00668a"><h2>Adaptive Feedback Applied</h2><p style="font-size:13px;opacity:0.7">The feedback loop generated new targeted personas based on failure patterns and re-ran tests. Results above include both initial and feedback runs.</p></div>' if report.feedback_applied else ''}

    <p style="text-align:center;font-size:12px;opacity:0.4;margin-top:24px">Generated by METRON Unified · {ts}</p>
  </div>
</body>
</html>"""
