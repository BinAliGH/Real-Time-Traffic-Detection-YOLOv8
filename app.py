import cv2
import datetime
import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        h1 {margin-top: 0rem;}
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
        }
        div[data-testid="stMetricLabel"] {font-size: 14px; color: #aaa;}
        div[data-testid="stMetricValue"] {font-size: 24px; color: #fff;}
    </style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES (OPTIMIZATION) ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# --- SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = [] 
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

# --- MAIN UI ---
st.title("🚦 Advanced Traffic Command Center")

# Layout
upload_col, settings_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader("📂 Upload Traffic Video", type=["mp4", "avi", "mov"])

# File Handling (Optimized for Large Files)
input_video_path = None
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    # Read/Write in chunks to prevent RAM crash
    while True:
        chunk = uploaded_file.read(10 * 1024 * 1024) # 10MB chunks
        if not chunk:
            break
        tfile.write(chunk)
    input_video_path = tfile.name

with settings_col:
    line_position = st.slider("Line Position (Y-Axis)", 100, 1000, 400)

# Controls
if input_video_path:
    if st.button("▶️ Start / Stop Analysis", use_container_width=True):
        st.session_state.run_detection = not st.session_state.run_detection
else:
    st.info("Please upload a video file to begin analysis.")

# --- DASHBOARD LAYOUT ---
st.markdown("---")
video_placeholder = st.empty()

st.markdown("### 📊 Live Analytics")

# Metrics Row
kpi1, kpi2, kpi3 = st.columns(3)
metric1 = kpi1.empty()
metric2 = kpi2.empty()
metric3 = kpi3.empty()

# Trend Chart Row
trend_chart_placeholder = st.empty()

# Charts Row
bar_chart_placeholder = st.empty()

# Logs Row
log_placeholder = st.empty()

# --- PROCESSING LOOP ---
if st.session_state.run_detection and input_video_path:
    # Load Model (Cached)
    model = load_model()
    cap = cv2.VideoCapture(input_video_path)
    
    # Init Tracking Variables
    counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
    track_history = {}
    already_counted = set()
    frame_count = 0
    
    while cap.isOpened() and st.session_state.run_detection:
        success, frame = cap.read()
        if not success:
            st.warning("Video finished.")
            st.session_state.run_detection = False
            break
            
        frame_count += 1
        
        # 1. YOLO TRACKING
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Draw Counting Line
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                class_name = results[0].names[cls]
                center_y = (y1 + y2) // 2

                # Store movement history
                if track_id not in track_history: track_history[track_id] = []
                track_history[track_id].append(center_y)
                if len(track_history[track_id]) > 30: track_history[track_id].pop(0)

                # Check line crossing
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2]
                    curr_y = track_history[track_id][-1]

                    if prev_y < line_position and curr_y >= line_position:
                        if track_id not in already_counted and class_name in counts:
                            counts[class_name] += 1
                            already_counted.add(track_id)
                            
                            # Store Data
                            st.session_state.traffic_data.append({
                                "Time": datetime.datetime.now(),
                                "Type": class_name
                            })

                # Draw Bounding Box & Labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name.capitalize()} {track_id}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. UPDATE VIDEO FEED
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # 3. UPDATE DASHBOARD (Throttled to every 5 frames)
        if frame_count % 5 == 0 and len(st.session_state.traffic_data) > 0:
            df = pd.DataFrame(st.session_state.traffic_data)
            
            # Time Windows
            now = pd.Timestamp.now()
            one_min_ago = now - pd.Timedelta(minutes=1)
            ten_mins_ago = now - pd.Timedelta(minutes=10)
            
            # KPI Metrics
            total_vehicles = len(df)
            vehicles_last_1min = len(df[df['Time'] >= one_min_ago])
            common_type = df['Type'].mode()[0].capitalize() if not df.empty else "N/A"

            metric1.metric("Total Traffic", total_vehicles)
            metric2.metric("Flow Rate (1 min)", f"{vehicles_last_1min} v/min")
            metric3.metric("Dominant Type", common_type)

            # Trend Chart
            df['Interval'] = df['Time'].dt.floor('5s')
            start_index = ten_mins_ago.floor('5s')
            end_index = now.floor('5s')
            full_index = pd.date_range(start=start_index, end=end_index, freq='5s')
            
            df_recent = df[df['Time'] >= ten_mins_ago]
            trend_data = df_recent.groupby('Interval').size().reindex(full_index, fill_value=0).reset_index()
            trend_data.columns = ['Time', 'Count']
            
            fig_trend = px.area(trend_data, x='Time', y='Count', 
                                title="🌊 Real-Time Traffic Volume (Last 10 Min)",
                                template="plotly_dark")
            fig_trend.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20),
                                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333'))
            fig_trend.update_traces(line_color='#00CC96', fillcolor='rgba(0, 204, 150, 0.3)')
            
            trend_chart_placeholder.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{frame_count}")

            # Vehicle Classification Bar Chart
            type_counts = df['Type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig_bar = px.bar(type_counts, x='Type', y='Count', color='Type',
                             title="🚗 Vehicle Classification", text='Count', template="plotly_dark")
            fig_bar.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            
            bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{frame_count}")

            # Logs
            with log_placeholder.container():
                with st.expander("📝 View Detailed Logs (Live)", expanded=False):
                    st.dataframe(df.tail(20).sort_values(by='Time', ascending=False), use_container_width=True)

    cap.release()


# to run the code 'streamlit run app.py --server.maxUploadSize=2000'