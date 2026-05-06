from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("fie.notifications")

_SENDGRID_API_KEY    = os.getenv("SENDGRID_API_KEY", "")
_NOTIFICATION_EMAIL  = os.getenv("NOTIFICATION_EMAIL", "")   # where to send alerts
_FROM_EMAIL          = os.getenv("FIE_FROM_EMAIL", "noreply@failure-intelligence.io")
_ENABLED             = bool(_SENDGRID_API_KEY and _NOTIFICATION_EMAIL)


def _send(subject: str, html: str, to: Optional[str] = None) -> None:
    """Fire-and-forget — never raises, never blocks the pipeline."""
    if not _ENABLED:
        return

    recipient = to or _NOTIFICATION_EMAIL

    def _post():
        try:
            import sendgrid  # type: ignore
            from sendgrid.helpers.mail import Mail  # type: ignore

            sg   = sendgrid.SendGridAPIClient(api_key=_SENDGRID_API_KEY)
            mail = Mail(
                from_email    = _FROM_EMAIL,
                to_emails     = recipient,
                subject       = subject,
                html_content  = html,
            )
            resp = sg.send(mail)
            logger.info(
                "[FIE] Email sent | subject=%r | to=%s | status=%s",
                subject, recipient, resp.status_code,
            )
        except ImportError:
            logger.debug("[FIE] sendgrid package not installed — skipping email")
        except Exception as exc:
            logger.warning("[FIE] Email send failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


# Email templates 
def _base(title: str, body_html: str, badge_color: str = "#ff4466") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body      {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0a0f1a; color: #c9d6e3; margin: 0; padding: 32px 16px; }}
    .card     {{ max-width: 560px; margin: 0 auto; background: #111820;
                border: 1px solid #1e2a38; border-radius: 12px; overflow: hidden; }}
    .header   {{ background: #0d1520; padding: 20px 28px;
                border-bottom: 1px solid #1e2a38;
                display: flex; align-items: center; gap: 12px; }}
    .badge    {{ background: {badge_color}22; color: {badge_color};
                border: 1px solid {badge_color}44; border-radius: 4px;
                font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
                padding: 3px 8px; font-family: monospace; }}
    .logo     {{ font-weight: 800; font-size: 15px; color: #00d4ff;
                font-family: monospace; letter-spacing: 0.06em; }}
    .body     {{ padding: 24px 28px; }}
    h2        {{ margin: 0 0 16px; font-size: 18px; font-weight: 700; color: #eaf0f8; }}
    .row      {{ display: flex; gap: 8px; margin-bottom: 10px;
                align-items: baseline; }}
    .key      {{ font-family: monospace; font-size: 11px; color: #5c7a99;
                min-width: 140px; letter-spacing: 0.05em; }}
    .val      {{ font-family: monospace; font-size: 13px; color: #c9d6e3; }}
    .excerpt  {{ background: #0d1520; border-left: 3px solid {badge_color};
                border-radius: 4px; padding: 12px 16px; margin: 16px 0;
                font-family: monospace; font-size: 12px; color: #8ba4bc;
                word-break: break-word; line-height: 1.6; }}
    .footer   {{ padding: 16px 28px; border-top: 1px solid #1e2a38;
                font-size: 11px; color: #3d5166; }}
    a         {{ color: #00d4ff; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="logo">FIE</span>
      <span class="badge">{title}</span>
    </div>
    <div class="body">
      {body_html}
    </div>
    <div class="footer">
      Failure Intelligence Engine · {ts} ·
      <a href="https://failure-intelligence-system.pages.dev">Dashboard</a>
    </div>
  </div>
</body>
</html>
"""


def notify_attack_detected(
    *,
    tenant_id:   str,
    attack_type: str,
    confidence:  float,
    prompt:      str,
    model_name:  str,
    request_id:  str,
    to:          Optional[str] = None,
) -> None:
    """
    Send an alert when an adversarial / jailbreak attack is caught.
    Called from routes.py after the jury verdict confirms is_adversarial=True.
    """
    excerpt  = prompt[:300] + ("…" if len(prompt) > 300 else "")
    conf_pct = f"{confidence * 100:.1f}%"

    body = f"""
    <h2>Jailbreak attack detected</h2>
    <div class="row"><span class="key">TENANT</span>     <span class="val">{tenant_id}</span></div>
    <div class="row"><span class="key">REQUEST ID</span> <span class="val">{request_id}</span></div>
    <div class="row"><span class="key">MODEL</span>      <span class="val">{model_name}</span></div>
    <div class="row"><span class="key">ATTACK TYPE</span><span class="val">{attack_type}</span></div>
    <div class="row"><span class="key">CONFIDENCE</span> <span class="val">{conf_pct}</span></div>
    <div class="excerpt">{excerpt}</div>
    <p style="font-size:13px;color:#5c7a99;margin:0">
      FIE blocked this request. No output was returned to the caller.
    </p>
    """

    _send(
        subject = f"[FIE] Attack detected — {attack_type} ({conf_pct} confidence)",
        html    = _base("ATTACK DETECTED", body, badge_color="#ff4466"),
        to      = to,
    )


def notify_human_review(
    *,
    tenant_id:         str,
    request_id:        str,
    escalation_reason: str,
    prompt:            str,
    model_name:        str,
    to:                Optional[str] = None,
) -> None:
    """
    Send an alert when FIE cannot verify ground truth and needs a human decision.
    Called from routes.py when requires_human_review becomes True.
    """
    excerpt = prompt[:280] + ("…" if len(prompt) > 280 else "")
    reason  = escalation_reason[:300] + ("…" if len(escalation_reason) > 300 else "")

    body = f"""
    <h2>Human review needed</h2>
    <p style="font-size:13px;color:#8ba4bc;margin:0 0 16px">
      FIE could not verify a reliable answer for this inference.
      The original model output was returned unchanged.
    </p>
    <div class="row"><span class="key">TENANT</span>    <span class="val">{tenant_id}</span></div>
    <div class="row"><span class="key">REQUEST ID</span><span class="val">{request_id}</span></div>
    <div class="row"><span class="key">MODEL</span>     <span class="val">{model_name}</span></div>
    <div class="row"><span class="key">REASON</span>    <span class="val">{reason}</span></div>
    <div class="excerpt">{excerpt}</div>
    <p style="font-size:13px;color:#5c7a99;margin:0">
      Use <code>fie.submit_feedback(request_id, is_correct=…)</code> to label this
      inference and improve future predictions.
    </p>
    """

    _send(
        subject = "[FIE] Human review needed — FIE couldn't verify this answer",
        html    = _base("HUMAN REVIEW NEEDED", body, badge_color="#ffaa00"),
        to      = to,
    )


def notify_weekly_digest(
    *,
    tenant_id:      str,
    total:          int,
    high_risk:      int,
    attacks:        int,
    fix_applied:    int,
    escalations:    int,
    top_archetype:  str,
    period_days:    int = 7,
    to:             Optional[str] = None,
) -> None:
    """
    Send a weekly usage digest to the tenant.
    Triggered by the /api/v1/notifications/digest endpoint or a cron job.
    """
    risk_pct  = round(high_risk / total * 100) if total else 0
    fix_pct   = round(fix_applied / total * 100) if total else 0

    def _row(key, val):
        return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'

    body = f"""
    <h2>Weekly digest — last {period_days} days</h2>
    {_row("TOTAL INFERENCES", f"{total:,}")}
    {_row("HIGH RISK",         f"{high_risk:,} ({risk_pct}%)")}
    {_row("ATTACKS CAUGHT",    str(attacks))}
    {_row("AUTO-FIXES APPLIED",f"{fix_applied:,} ({fix_pct}%)")}
    {_row("ESCALATIONS",       str(escalations))}
    {_row("TOP FAILURE TYPE",  top_archetype.replace("_", " "))}
    <p style="font-size:13px;color:#5c7a99;margin:16px 0 0">
      <a href="https://failure-intelligence-system.pages.dev">View full dashboard →</a>
    </p>
    """

    _send(
        subject = f"[FIE] Weekly digest — {total:,} inferences, {attacks} attacks",
        html    = _base("WEEKLY DIGEST", body, badge_color="#00d4ff"),
        to      = to,
    )
