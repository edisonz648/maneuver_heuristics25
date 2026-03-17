import cv2
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
video_ID = "0EXT"

VIDEO_PATH = f"clips/Parking-clip{video_ID}.mp4"
CSV_PATH   = f"clip_trajectory_csvs/Parking-clip{video_ID}.csv"

FRAME_COL  = "frame"
CX_COL     = "cx"
CY_COL     = "cy"

PREDICTIONS_CSV_PATH = "consolidated_predictions.csv"

TINT_WINDOW          = 3    # frames on each side for color flash
BASE_INTERVAL_MS     = 80   # ~12 FPS normal playback
CRITICAL_WINDOW      = 15   # frames on each side to slow down near key frames
CRITICAL_INTERVAL_MS = 200  # ~5 FPS near critical frames
# ─────────────────────────────────────────────────────────────────────────────

# ── Load CSVs ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.sort_values(FRAME_COL).reset_index(drop=True)

predictions_df = pd.read_csv(PREDICTIONS_CSV_PATH)
prediction = predictions_df[predictions_df["ID"] == video_ID]
front_parking_frame = prediction["front_parking"].values[0]
rear_parking_frame  = prediction["rear_parking"].values[0]
peak_frame          = prediction["peak"].values[0]
zone_based_frame    = prediction["zone_based"].values[0]

CRITICAL_FRAMES = [front_parking_frame, rear_parking_frame, peak_frame, zone_based_frame]

# ── Helper: nearest CSV row ───────────────────────────────────────────────────
def get_nearest_row(frame_idx):
    if df.empty:
        return None
    idx = (df[FRAME_COL] - frame_idx).abs().idxmin()
    return df.loc[idx]

# ── Resolve static marker coords ─────────────────────────────────────────────
def coord_for(frame_num):
    row = get_nearest_row(frame_num)
    return row if row is not None else None

front_parking_coord = coord_for(front_parking_frame)
rear_parking_coord  = coord_for(rear_parking_frame)
peak_coord          = coord_for(peak_frame)
zone_based_coord    = coord_for(zone_based_frame)

# label, coord, frame_num, color, symbol
STATIC_MARKERS = [
    ("Front Parking", front_parking_coord, front_parking_frame, "green",  "star"),
    ("Rear Parking",  rear_parking_coord,  rear_parking_frame,  "orange", "diamond"),
    ("Peak",          peak_coord,          peak_frame,          "purple", "triangle-up"),
    ("Zone Based",    zone_based_coord,    zone_based_frame,    "brown",  "square"),
]

# ── Tint definitions ──────────────────────────────────────────────────────────
if video_ID.endswith("EXT"):
    parking_tint = (0, 200,   0)   # green  (BGR)
    event_tint   = (0,   0, 220)   # red    (BGR)
elif video_ID.endswith("ENT"):
    parking_tint = (0,   0, 220)   # red    (BGR)
    event_tint   = (0, 200,   0)   # green  (BGR)
else:
    parking_tint = (0, 200,   0)   # default green
    event_tint   = (0,   0, 220)   # default red

TINTS = [
    (front_parking_frame, parking_tint),
    (rear_parking_frame,  parking_tint),
    (peak_frame,          event_tint),
    (zone_based_frame,    event_tint),
]

# ── Preload all frames at startup ─────────────────────────────────────────────
print("Loading video frames into memory... (one-time cost)")
cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS          = cap.get(cv2.CAP_PROP_FPS)

FRAMES_B64 = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    FRAMES_B64.append(base64.b64encode(buffer).decode("utf-8"))

cap.release()
TOTAL_FRAMES = len(FRAMES_B64)
print(f"Loaded {TOTAL_FRAMES} frames at {FPS:.1f} FPS")


# ── Frame helpers ─────────────────────────────────────────────────────────────
def get_frame_b64(frame_idx: int) -> str:
    if 0 <= frame_idx < len(FRAMES_B64):
        return FRAMES_B64[frame_idx]
    return ""


def apply_tint(frame_b64: str, color_bgr: tuple, alpha: float = 0.35) -> str:
    data    = base64.b64decode(frame_b64)
    arr     = np.frombuffer(data, dtype=np.uint8)
    img     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    overlay = np.full_like(img, color_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    _, buf  = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode("utf-8")


def get_active_tint(frame_idx: int):
    for key_frame, color in TINTS:
        if abs(frame_idx - key_frame) <= TINT_WINDOW:
            return color
    return None


def is_near_critical(frame_idx: int) -> bool:
    return any(abs(frame_idx - cf) <= CRITICAL_WINDOW for cf in CRITICAL_FRAMES)


# ── Trajectory figure ─────────────────────────────────────────────────────────
def build_trajectory_fig(highlight_idx: int | None = None):
    traces = [
        go.Scatter(
            x=df[CX_COL], y=df[CY_COL],
            mode="lines",
            line=dict(color="royalblue", width=1.5),
            name="Trajectory",
            hovertemplate="cx: %{x}<br>cy: %{y}<extra></extra>",
        )
    ]

    # ── Static event markers ──────────────────────────────────────────────────
    for label, coord, frame_num, color, symbol in STATIC_MARKERS:
        if coord is not None:
            traces.append(go.Scatter(
                x=[coord[CX_COL]], y=[coord[CY_COL]],
                mode="markers+text",
                marker=dict(color=color, size=14, symbol=symbol,
                            line=dict(color="white", width=1.5)),
                text=[f"{label}<br>(f{frame_num})"],
                textposition="top center",
                textfont=dict(size=10),
                name=label,
            ))

    # ── Live tracking dot ─────────────────────────────────────────────────────
    if highlight_idx is not None:
        row = get_nearest_row(highlight_idx)
        if row is not None:
            traces.append(go.Scatter(
                x=[row[CX_COL]], y=[row[CY_COL]],
                mode="markers",
                marker=dict(color="crimson", size=12, symbol="circle",
                            line=dict(color="white", width=2)),
                name="Current",
            ))

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title="Trajectory",
            xaxis=dict(title="cx"),
            yaxis=dict(title="cy", autorange="reversed"),
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h"),
            uirevision="trajectory",
        ),
    )


# ── DASH APP ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Video + Trajectory Viewer", style={"textAlign": "center", "fontFamily": "sans-serif"}),

    html.Div([
        # ── Left: video + controls ────────────────────────────────────────────
        html.Div([
            html.Img(id="video-frame",
                     style={"width": "100%", "borderRadius": "6px", "background": "#000"}),

            html.Div([
                html.Button("▶ Play",  id="play-btn",  n_clicks=0,
                            style={"marginRight": "8px"}),
                html.Button("⏸ Pause", id="pause-btn", n_clicks=0,
                            style={"marginRight": "8px"}),
                html.Button("↺ Reset", id="reset-btn", n_clicks=0),
            ], style={"margin": "10px 0", "display": "flex", "alignItems": "center"}),

            dcc.Slider(
                id="frame-slider",
                min=0, max=TOTAL_FRAMES - 1, step=1, value=0,
                marks={0: "0", TOTAL_FRAMES - 1: str(TOTAL_FRAMES - 1)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(id="frame-counter",
                     style={"fontFamily": "monospace", "marginTop": "4px",
                            "fontSize": "12px", "color": "#555"}),
        ], style={"flex": "1", "padding": "10px"}),

        # ── Right: trajectory plot ────────────────────────────────────────────
        html.Div([
            dcc.Graph(id="trajectory-plot", figure=build_trajectory_fig(),
                      style={"height": "480px"}),
        ], style={"flex": "1", "padding": "10px"}),
    ], style={"display": "flex", "flexWrap": "wrap"}),

    # ── Hidden state ──────────────────────────────────────────────────────────
    dcc.Store(id="is-playing", data=False),
    dcc.Store(id="current-frame", data=0),
    dcc.Interval(id="playback-interval",
                 interval=BASE_INTERVAL_MS,
                 n_intervals=0,
                 disabled=True),
])


# ── 1. Play / Pause / Reset ───────────────────────────────────────────────────
@app.callback(
    Output("is-playing",        "data"),
    Output("playback-interval", "disabled"),
    Output("current-frame",     "data", allow_duplicate=True),
    Input("play-btn",  "n_clicks"),
    Input("pause-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def control_playback(play, pause, reset):
    triggered = dash.callback_context.triggered[0]["prop_id"]
    if "reset" in triggered:
        return False, True, 0
    if "play" in triggered:
        return True, False, dash.no_update
    return False, True, dash.no_update


# ── 2. Interval tick → advance frame ─────────────────────────────────────────
@app.callback(
    Output("current-frame", "data"),
    Input("playback-interval", "n_intervals"),
    dash.State("current-frame", "data"),
    dash.State("is-playing",    "data"),
    prevent_initial_call=True,
)
def advance_frame(_, current, playing):
    if not playing:
        return current
    nxt = current + 1
    return 0 if nxt >= TOTAL_FRAMES else nxt


# ── 2b. Dynamically slow down near critical frames ────────────────────────────
@app.callback(
    Output("playback-interval", "interval"),
    Input("current-frame", "data"),
)
def adjust_interval(frame_idx):
    if is_near_critical(frame_idx):
        return CRITICAL_INTERVAL_MS
    return BASE_INTERVAL_MS


# ── 3. Slider drag → update stored frame ─────────────────────────────────────
@app.callback(
    Output("current-frame", "data", allow_duplicate=True),
    Input("frame-slider", "value"),
    prevent_initial_call=True,
)
def slider_to_frame(value):
    return value


# ── 4. current-frame → render everything ─────────────────────────────────────
@app.callback(
    Output("video-frame",     "src"),
    Output("trajectory-plot", "figure"),
    Output("frame-slider",    "value"),
    Output("frame-counter",   "children"),
    Input("current-frame",    "data"),
)
def render(frame_idx):
    img_b64 = get_frame_b64(frame_idx)
    tint_color = get_active_tint(frame_idx)
    if tint_color and img_b64:
        img_b64 = apply_tint(img_b64, tint_color)
    src   = f"data:image/jpeg;base64,{img_b64}" if img_b64 else ""
    fig   = build_trajectory_fig(highlight_idx=frame_idx)
    label = f"Frame {frame_idx} / {TOTAL_FRAMES - 1}  ·  {frame_idx / FPS:.2f}s"
    return src, fig, frame_idx, label


if __name__ == "__main__":
    app.run(debug=True)
