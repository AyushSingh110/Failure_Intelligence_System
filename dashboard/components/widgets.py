from datetime import datetime


# Colour palette 
_C = {
    "bg_card":    "#161b22",
    "bg_page":    "#0a0c10",
    "border":     "#21262d",
    "text_dim":   "#6e7681",
    "text_main":  "#e6edf3",
    "text_body":  "#c9d1d9",
    "blue":       "#58a6ff",
    "green":      "#3fb950",
    "red":        "#f85149",
    "yellow":     "#e3b341",
    "purple":     "#bc8cff",
}

_ACCENT = {
    "ok":    _C["green"],
    "risk":  _C["red"],
    "warn":  _C["yellow"],
    "novel": _C["purple"],
    "info":  _C["blue"],
}


# Page header 

def render_page_header(subtitle: str = "Operational Monitoring · Phase 1") -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""
    <div style="display:flex;align-items:center;gap:14px;
                border-bottom:1px solid {_C['border']};
                padding-bottom:18px;margin-bottom:24px;">
      <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;
                    font-weight:600;color:{_C['blue']};letter-spacing:-1px;">
          ⬡ Failure Intelligence Engine
        </div>
        <div style="font-size:11px;color:{_C['text_dim']};
                    text-transform:uppercase;letter-spacing:2px;margin-top:4px;">
          {subtitle}
        </div>
      </div>
      <div style="margin-left:auto;font-family:'IBM Plex Mono',monospace;
                  font-size:11px;color:{_C['text_dim']};">
        {ts}
      </div>
    </div>
    """


# Section label 

def render_section_label(text: str, margin_top: bool = False) -> str:
    mt = "margin-top:24px;" if margin_top else ""
    return f"""
    <div style="{mt}font-family:'IBM Plex Mono',monospace;font-size:11px;
                text-transform:uppercase;letter-spacing:2px;color:{_C['text_dim']};
                border-bottom:1px solid {_C['border']};padding-bottom:8px;margin-bottom:16px;">
      {text}
    </div>
    """


# Status pill 

def render_status_pill(label: str, variant: str = "ok") -> str:
    colour = _ACCENT.get(variant, _C["blue"])
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:600;'
        f'background:rgba({_hex_to_rgb(colour)},0.15);color:{colour};'
        f'border:1px solid rgba({_hex_to_rgb(colour)},0.35);">'
        f'{label}</span>'
    )

def render_kpi_card_html(
    label: str,
    value: str,
    delta: str = "",
    delta_dir: str = "",   
    variant: str = "info",
) -> str:
    """
    Returns a single KPI card as a self-contained HTML block with
    100% inline styles. Safe to render inside any st.markdown() call.
    """
    accent  = _ACCENT.get(variant, _C["blue"])
    d_color = _C["red"] if delta_dir == "up" else (_C["green"] if delta_dir == "down" else _C["text_dim"])

    return f"""
    <div style="background:{_C['bg_card']};border:1px solid {_C['border']};
                border-radius:8px;padding:16px 20px;position:relative;
                border-top:2px solid {accent};min-height:90px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                  color:{_C['text_dim']};margin-bottom:6px;">
        {label}
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;
                  font-weight:600;color:{_C['text_main']};line-height:1;">
        {value}
      </div>
      <div style="font-size:11px;margin-top:6px;color:{d_color};">
        {delta}
      </div>
    </div>
    """


# Inference row 

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

    pill = (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:600;'
        f'background:rgba({_hex_to_rgb(colour)},0.15);color:{colour};'
        f'border:1px solid rgba({_hex_to_rgb(colour)},0.3);">H: {ent_str}</span>'
    )

    return f"""
    <div style="background:{_C['bg_card']};border:1px solid {_C['border']};
                border-radius:6px;padding:10px 14px;margin-bottom:6px;
                display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    color:{_C['blue']};">{short_id}</div>
        <div style="font-size:11px;color:{_C['text_dim']};margin-top:2px;">
          {model_name} · {ts_clean}
        </div>
      </div>
      <div>{pill}</div>
    </div>
    """


# Empty state 
def render_empty_state(message: str) -> str:
    return f"""
    <div style="border:1px dashed {_C['border']};border-radius:8px;
                padding:40px 24px;text-align:center;color:{_C['text_dim']};
                font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:2;">
      {message}
    </div>
    """


# Connection badge 
def render_connection_badge(connected: bool) -> str:
    return render_status_pill("● CONNECTED", "ok") if connected else render_status_pill("● OFFLINE", "risk")


# Helper 

def _hex_to_rgb(hex_color: str) -> str:
    """Converts #RRGGBB to 'R,G,B' string for use in rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"