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
        tax_id    = f.get("failure_taxonomy_id", "")
        tax_label = f.get("failure_taxonomy_label", "")
        fail_rsn  = f.get("failure_reason", "")
        # taxonomy badge + reason cell
        if tax_id and tax_label:
            rsn_html = (
                "<br><span style=\"font-size:11px;color:#555\">" + fail_rsn[:200] + "</span>"
                if fail_rsn else ""
            )
            reason_cell = (
                '<span style="display:inline-block;background:#ede7f6;color:#6200ee;'
                'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-bottom:4px">'
                + tax_id + " · " + tax_label + "</span>" + rsn_html
            )
        else:
            reason_cell = f.get("reason", "")[:150]
        failure_rows += f"""
        <tr>
          <td>{f.get('superset','')}</td>
          <td>{f.get('metric_name','')}</td>
          <td>{f.get('persona_name','')}</td>
          <td style="color:{'#ba1a1a' if score_pct<50 else '#ff8c00'}">{score_pct}%</td>
          <td style="max-width:320px;word-wrap:break-word">{reason_cell}</td>
        </tr>"""

    # RCA section
    rca_section = ""
    if report.rca and report.rca.top_causes:
        rca = report.rca

        # Architecture summary pills
        arch = rca.architecture_summary
        arch_pills = ""
        pill_items = [
            ("App Type",       arch.get("application_type", "—").replace("_", " ").title()),
            ("Deployment",     arch.get("deployment_type", "unknown").title()),
            ("Vector DB",      arch.get("vector_db", "none").title() if arch.get("vector_db") else "None"),
            ("Session DB",     arch.get("session_db", "none").title() if arch.get("session_db") else "None"),
            ("Rate Limiting",  "Yes" if arch.get("has_rate_limiting") else "No"),
            ("Retry Logic",    "Yes" if arch.get("has_retry_logic") else "No"),
            ("Circuit Breaker","Yes" if arch.get("has_circuit_breaker") else "No"),
        ]
        for pill_label, pill_val in pill_items:
            ok = pill_val not in ("No", "None", "Unknown")
            pill_color = "#d4edda" if ok else "#f8d7da"
            txt_color  = "#006e2f" if ok else "#ba1a1a"
            arch_pills += f'<span style="display:inline-block;background:{pill_color};color:{txt_color};border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700;margin:2px 4px 2px 0">{pill_label}: {pill_val}</span>'

        # Signal summary badges
        signals_html = ""
        for sig_key, count in sorted(rca.signal_summary.items(), key=lambda x: -x[1])[:8]:
            signals_html += f'<span style="display:inline-block;background:#fff3cd;color:#856404;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:600;margin:2px 4px 2px 0">{sig_key.replace("_", " ")} ({count})</span>'

        # Root cause cards
        cause_cards = ""
        for cause in rca.top_causes:
            prob_pct = int(cause.probability * 100)
            if prob_pct >= 70:
                prob_color = "#ba1a1a"
                prob_bg    = "#f8d7da"
                prob_label = "HIGH"
            elif prob_pct >= 45:
                prob_color = "#ff8c00"
                prob_bg    = "#fff3cd"
                prob_label = "MEDIUM"
            else:
                prob_color = "#00668a"
                prob_bg    = "#d1ecf1"
                prob_label = "LOW"

            # Category badge color
            cat_colors = {
                "C1": ("#6200ee", "#ede7f6"),
                "C2": ("#00668a", "#d1ecf1"),
                "C3": ("#e65100", "#fff3e0"),
                "C4": ("#ba1a1a", "#f8d7da"),
                "C5": ("#1b5e20", "#d4edda"),
                "C6": ("#f57f17", "#fff9c4"),
                "C7": ("#4a148c", "#f3e5f5"),
                "C8": ("#37474f", "#eceff1"),
            }
            cat_fg, cat_bg = cat_colors.get(cause.category_id, ("#333", "#eee"))

            evidence_items = "".join(
                f'<li style="margin:2px 0;font-size:12px;color:#555">{e}</li>'
                for e in cause.evidence[:3]
            )

            bar_width = max(4, prob_pct)

            cause_cards += f"""
        <div style="border:1px solid #e0e0e0;border-radius:10px;padding:14px 16px;margin-bottom:12px;background:#fafafa">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap">
                <span style="font-size:12px;font-weight:800;color:#888">#{cause.rank}</span>
                <span style="background:{cat_bg};color:{cat_fg};border-radius:4px;padding:1px 7px;font-size:11px;font-weight:700">{cause.category}</span>
                <span style="font-size:11px;color:#888;font-family:monospace">{cause.id}</span>
              </div>
              <div style="font-size:14px;font-weight:700;color:#1c1b1f;margin-bottom:6px">{cause.label}</div>
              <ul style="margin:0 0 6px 14px;padding:0">{evidence_items}</ul>
              {f'<div style="font-size:12px;color:#444;line-height:1.6;margin-top:6px;padding:6px 10px;background:#f5f5f5;border-radius:6px;border-left:3px solid #bbb">{cause.reason}</div>' if cause.reason else ''}
              <div style="font-size:12px;background:#e8f5e9;color:#1b5e20;border-radius:6px;padding:5px 9px;margin-top:6px">
                <strong>Fix:</strong> {cause.remediation}
              </div>
            </div>
            <div style="text-align:center;min-width:72px">
              <div style="background:{prob_bg};color:{prob_color};border-radius:8px;padding:6px 10px;font-size:22px;font-weight:900;line-height:1">{prob_pct}%</div>
              <div style="font-size:10px;font-weight:800;color:{prob_color};letter-spacing:0.06em;margin-top:2px">{prob_label}</div>
              <div style="margin-top:6px;background:#e0e0e0;border-radius:4px;height:6px;width:72px">
                <div style="background:{prob_color};height:6px;border-radius:4px;width:{bar_width}%"></div>
              </div>
            </div>
          </div>
        </div>"""

        rca_section = f"""
    <div class="card" style="border-left:4px solid #6200ee">
      <h2 style="color:#6200ee">Root Cause Analysis (Stage 8)</h2>
      <p style="font-size:13px;opacity:0.7;margin:0 0 14px">
        Analysed <strong>{rca.total_analyzed}</strong> evaluation results ·
        <strong>{rca.total_failed}</strong> failures mapped ·
        <strong>{rca.relevant_points}</strong> / {rca.relevant_points + rca.filtered_points} failure points relevant after architecture filter
      </p>

      <div style="margin-bottom:16px">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;opacity:0.55;margin-bottom:6px">Architecture Profile</div>
        {arch_pills}
      </div>

      {'<div style="margin-bottom:16px"><div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;opacity:0.55;margin-bottom:6px">Observed Failure Signals</div>' + signals_html + '</div>' if signals_html else ''}

      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;opacity:0.55;margin-bottom:10px">Top Probable Root Causes</div>
      {cause_cards}
    </div>"""

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
      <p>{report.agent_name or report.domain.capitalize()} · {report.application_type.value.replace('_',' ').title()} · {ts}</p>
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

    {rca_section}

    {'<div class="card" style="border:2px solid #00668a"><h2>Adaptive Feedback Applied</h2><p style="font-size:13px;opacity:0.7">The feedback loop generated new targeted personas based on failure patterns and re-ran tests. Results above include both initial and feedback runs.</p></div>' if report.feedback_applied else ''}

    <p style="text-align:center;font-size:12px;opacity:0.4;margin-top:24px">Generated by METRON Unified · {ts}</p>
  </div>
</body>
</html>"""
