from datetime import datetime


# Colour palette
_C = {
    "bg_card":    "#0d1117",
    "bg_surface": "#161b22",
    "bg_page":    "#090b10",
    "border":     "#21262d",
    "border_sub": "#1b1f27",
    "text_dim":   "#6e7681",
    "text_muted": "#484f58",
    "text_main":  "#e6edf3",
    "text_body":  "#c9d1d9",
    "blue":       "#58a6ff",
    "green":      "#3fb950",
    "red":        "#f85149",
    "yellow":     "#e3b341",
    "purple":     "#bc8cff",
    "orange":     "#f0883e",
}

_ACCENT = {
    "ok":    _C["green"],
    "risk":  _C["red"],
    "warn":  _C["yellow"],
    "novel": _C["purple"],
    "info":  _C["blue"],
}


# ── Page header ────────────────────────────────────────────────────────────

def render_page_header(subtitle: str = "Operational Monitoring · Phase 3 Active") -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""
    <div style="display:flex;align-items:center;gap:16px;
                border-bottom:1px solid {_C['border']};
                padding-bottom:20px;margin-bottom:28px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:38px;height:38px;border-radius:10px;
                    background:linear-gradient(135deg, rgba(88,166,255,0.15) 0%, rgba(188,140,255,0.10) 100%);
                    border:1px solid rgba(88,166,255,0.2);
                    display:flex;align-items:center;justify-content:center;
                    font-size:18px;">⬡</div>
        <div>
          <div style="font-family:'Inter',sans-serif;font-size:18px;
                      font-weight:700;color:{_C['text_main']};letter-spacing:-0.5px;">
            Failure Intelligence Engine
          </div>
          <div style="font-size:11px;color:{_C['text_dim']};
                      text-transform:uppercase;letter-spacing:1.5px;margin-top:2px;
                      font-weight:500;">
            {subtitle}
          </div>
        </div>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                     background:{_C['green']};box-shadow:0 0 6px {_C['green']};"></span>
        <span style="font-family:'JetBrains Mono',monospace;
                     font-size:11px;color:{_C['text_dim']};">
          {ts}
        </span>
      </div>
    </div>
    """


# ── Section label ──────────────────────────────────────────────────────────

def render_section_label(text: str, margin_top: bool = False) -> str:
    mt = "margin-top:28px;" if margin_top else ""
    return f"""
    <div style="{mt}display:flex;align-items:center;gap:10px;
                margin-bottom:16px;padding-bottom:10px;
                border-bottom:1px solid {_C['border']};">
      <div style="width:3px;height:14px;border-radius:2px;
                  background:linear-gradient(180deg, {_C['blue']} 0%, {_C['purple']} 100%);"></div>
      <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                   text-transform:uppercase;letter-spacing:1.5px;color:{_C['text_dim']};">
        {text}
      </span>
    </div>
    """


# ── Status pill ────────────────────────────────────────────────────────────

def render_status_pill(label: str, variant: str = "ok") -> str:
    colour = _ACCENT.get(variant, _C["blue"])
    return (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:600;'
        f'background:rgba({_hex_to_rgb(colour)},0.12);color:{colour};'
        f'border:1px solid rgba({_hex_to_rgb(colour)},0.25);'
        f'letter-spacing:0.3px;">'
        f'{label}</span>'
    )


# ── KPI card ───────────────────────────────────────────────────────────────

def render_kpi_card_html(
    label: str,
    value: str,
    delta: str = "",
    delta_dir: str = "",
    variant: str = "info",
) -> str:
    accent  = _ACCENT.get(variant, _C["blue"])
    d_color = _C["red"] if delta_dir == "up" else (_C["green"] if delta_dir == "down" else _C["text_dim"])

    return f"""
    <div style="background:{_C['bg_card']};
                border:1px solid {_C['border']};
                border-radius:12px;padding:18px 22px;position:relative;
                min-height:100px;
                border-left:3px solid {accent};
                transition:border-color 0.2s ease;">
      <div style="font-family:'Inter',sans-serif;font-size:10px;
                  text-transform:uppercase;letter-spacing:1.5px;font-weight:600;
                  color:{_C['text_dim']};margin-bottom:10px;">
        {label}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;
                  font-weight:700;color:{_C['text_main']};line-height:1;
                  letter-spacing:-0.5px;">
        {value}
      </div>
      <div style="font-family:'Inter',sans-serif;font-size:11px;
                  margin-top:10px;color:{d_color};font-weight:500;">
        {delta}
      </div>
    </div>
    """


# ── Inference row ──────────────────────────────────────────────────────────

def render_inference_row(
    request_id: str,
    model_name: str,
    timestamp: str,
    entropy_val: float | None,
) -> str:
    ent_str  = f"{entropy_val:.3f}" if entropy_val is not None else "—"
    is_risk  = (entropy_val or 0) > 0.75
    colour   = _C["red"] if is_risk else _C["green"]
    short_id = (request_id[:26] + "…") if len(request_id) > 26 else request_id
    ts_clean = timestamp[:19].replace("T", " ") if timestamp else "—"

    indicator = (
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{colour};box-shadow:0 0 6px rgba({_hex_to_rgb(colour)},0.4);'
        f'margin-right:8px;vertical-align:middle;"></span>'
    )

    pill = (
        f'<span style="display:inline-block;padding:4px 12px;border-radius:6px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:600;'
        f'background:rgba({_hex_to_rgb(colour)},0.1);color:{colour};'
        f'letter-spacing:0.3px;">H: {ent_str}</span>'
    )

    return f"""
    <div style="background:{_C['bg_card']};border:1px solid {_C['border']};
                border-radius:8px;padding:12px 16px;margin-bottom:6px;
                display:flex;justify-content:space-between;align-items:center;
                transition:border-color 0.15s ease;">
      <div style="display:flex;align-items:center;">
        {indicator}
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                      color:{_C['blue']};font-weight:500;">{short_id}</div>
          <div style="font-family:'Inter',sans-serif;font-size:11px;
                      color:{_C['text_dim']};margin-top:3px;">
            {model_name}&ensp;·&ensp;{ts_clean}
          </div>
        </div>
      </div>
      <div>{pill}</div>
    </div>
    """


# ── Empty state ────────────────────────────────────────────────────────────

def render_empty_state(message: str) -> str:
    return f"""
    <div style="border:1px dashed {_C['border']};border-radius:12px;
                padding:48px 32px;text-align:center;color:{_C['text_dim']};
                font-family:'Inter',sans-serif;font-size:13px;line-height:2;
                background:rgba(13,17,23,0.4);">
      <div style="font-size:28px;margin-bottom:12px;opacity:0.4;">⬡</div>
      {message}
    </div>
    """


# ── Connection badge ───────────────────────────────────────────────────────

def render_connection_badge(connected: bool) -> str:
    return render_status_pill("● CONNECTED", "ok") if connected else render_status_pill("● OFFLINE", "risk")


# ── Field label (for forms) ────────────────────────────────────────────────

def render_field_label(text: str, hint: str = "") -> str:
    hint_html = (
        f'<span style="color:{_C["text_muted"]};font-weight:400;'
        f'margin-left:6px;font-size:10px;">({hint})</span>'
        if hint else ""
    )
    return (
        f'<div style="font-family:\'Inter\',sans-serif;font-size:12px;'
        f'font-weight:600;color:{_C["text_dim"]};letter-spacing:0.5px;'
        f'text-transform:uppercase;margin-bottom:6px;margin-top:4px;">'
        f'{text}{hint_html}</div>'
    )


# ── Info callout ───────────────────────────────────────────────────────────

def render_callout(text: str, variant: str = "info") -> str:
    colour = _ACCENT.get(variant, _C["blue"])
    return (
        f'<div style="background:rgba({_hex_to_rgb(colour)},0.05);'
        f'border:1px solid rgba({_hex_to_rgb(colour)},0.15);'
        f'border-left:3px solid {colour};border-radius:8px;'
        f'padding:14px 18px;margin-bottom:16px;'
        f'font-family:\'Inter\',sans-serif;font-size:13px;'
        f'color:{_C["text_body"]};line-height:1.6;">'
        f'{text}</div>'
    )


# ── Helper ─────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    """Converts #RRGGBB to 'R,G,B' string for use in rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"