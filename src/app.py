import streamlit as st
import cv2
import numpy as np
from main import OpticalFlowDetector
import plotly.graph_objects as go
import time

custom_css = """
<style>
    body {
        color: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #4CAF50;
        font-weight: 700;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 25px;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .streamlit-expanderHeader {
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50;
        border-radius: 10px;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
    }
    .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(5px);
    }
</style>
"""

st.set_page_config(page_title="Capturing Optical Flow", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

if 'detector' not in st.session_state:
    st.session_state.detector = OpticalFlowDetector()
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def sidebar():
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.session_state.detector.trajectory_len = st.slider("🧵 Trajectory Length", 10, 100, 40)
        st.session_state.detector.detect_interval = st.slider("⏱️ Detect Interval", 1, 20, 5)
        st.session_state.detector.feature_params['maxCorners'] = st.slider("🔢 Max Corners", 10, 100, 20)
        st.session_state.detector.feature_params['qualityLevel'] = st.slider("🎯 Quality Level", 0.1, 0.5, 0.3, 0.01)

def main_content():
    st.title("🌊 Capturing Optical Flow")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run = st.button("▶️ Start/Stop")
    
    with col2:
        st.write("⏱️ FPS: ")
        fps_text = st.empty()
    
    with col3:
        st.write("🔢 Track Count: ")
        track_count = st.empty()

    video_feed = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = st.session_state.detector.process_frame(frame)
            
            cv2.polylines(processed_frame, [np.int32(trajectory) for trajectory in st.session_state.detector.trajectories], False, (0, 255, 0))

            video_feed.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Update FPS
            st.session_state.frame_count += 1
            elapsed_time = time.time() - st.session_state.start_time
            fps = st.session_state.frame_count / elapsed_time
            fps_text.write(f"⏱️ FPS: {fps:.2f}")
            
            # Update track count
            track_count.write(f"🔢 Track Count: {len(st.session_state.detector.trajectories)}")

            if not run:
                break

        cap.release()

    # Trajectory visualization
    st.markdown("---")
    st.subheader("📊 Trajectory Visualization")
    if st.session_state.detector.trajectories:
        fig = go.Figure()
        for trajectory in st.session_state.detector.trajectories:
            x, y = zip(*trajectory)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color='#4CAF50', width=2), marker=dict(size=4)))
        
        fig.update_layout(
            title="Optical Flow Trajectories",
            xaxis_title="X position",
            yaxis_title="Y position",
            font=dict(color="white", family="Roboto"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trajectories detected yet. Start the video feed to see visualization.")

if __name__ == "__main__":
    sidebar()
    main_content()
