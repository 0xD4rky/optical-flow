import streamlit as st
import cv2
import numpy as np
from main import OpticalFlowDetector
import plotly.graph_objects as go
import time

# Custom CSS with more advanced styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        color: #E0E0E0;
        background-color: #121212;
        font-family: 'Roboto', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #4CAF50;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
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
    .sidebar .sidebar-content {
        background-color: rgba(18, 18, 18, 0.9);
        backdrop-filter: blur(10px);
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
    }
    .stPlotlyChart {
        background-color: rgba(40, 40, 40, 0.7);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stats-container {
        background-color: rgba(40, 40, 40, 0.7);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .custom-info-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .custom-info-box h4 {
        color: #4CAF50;
        margin-top: 0;
    }
</style>
"""

# JavaScript for improving button responsiveness
js_code = """
<script>
function toggleRunning() {
    const button = document.querySelector('.stButton button');
    if (button.innerText === '▶️ Start') {
        button.innerText = '⏹️ Stop';
    } else {
        button.innerText = '▶️ Start';
    }
}
</script>
"""

st.set_page_config(page_title="Advanced Optical Flow Tracker", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(js_code, unsafe_allow_html=True)

# Initialize session state variables
if 'detector' not in st.session_state:
    st.session_state.detector = OpticalFlowDetector()
if 'running' not in st.session_state:
    st.session_state.running = False
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

def sidebar():
    with st.sidebar:
        st.title("📊 Statistics")
        st.session_state.stats_container = st.empty()

def update_stats():
    with st.session_state.stats_container.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FPS", f"{st.session_state.fps:.2f}")
        with col2:
            st.metric("Track Count", len(st.session_state.detector.trajectories))

def calculate_fps():
    current_time = time.time()
    fps = 1 / (current_time - st.session_state.last_update_time)
    st.session_state.last_update_time = current_time
    return fps

def main_content():
    st.title("🌊 Advanced Optical Flow Tracker")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_feed = st.empty()
    
    with col2:
        st.markdown("### 🎮 Controls")
        start_stop = st.button("▶️ Start" if not st.session_state.running else "⏹️ Stop", on_click=lambda: st.markdown("<script>toggleRunning()</script>", unsafe_allow_html=True))
        if start_stop:
            st.session_state.running = not st.session_state.running
        
        st.markdown("### 📈 Trajectory Visualization")
        plot_container = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        if st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = st.session_state.detector.process_frame(frame)
            
            # Draw trajectories
            for trajectory in st.session_state.detector.trajectories:
                cv2.polylines(processed_frame, [np.int32(trajectory)], False, (0, 255, 0), 2)
                if len(trajectory) > 1:
                    cv2.circle(processed_frame, tuple(map(int, trajectory[-1])), 4, (0, 0, 255), -1)

            video_feed.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Update FPS
            st.session_state.fps = calculate_fps()
            
            # Update statistics
            update_stats()
            
            # Update trajectory plot
            update_trajectory_plot(plot_container)
        else:
            with video_feed:
                st.info("Press the Start button to begin optical flow tracking.")
            time.sleep(0.1)  # Prevent high CPU usage when not running

        # Check if the Streamlit script should rerun
        if st.session_state.running != st.session_state.get('last_running_state', None):
            st.session_state.last_running_state = st.session_state.running
            st.rerun()

    cap.release()

def update_trajectory_plot(container):
    if st.session_state.detector.trajectories:
        fig = go.Figure()
        for trajectory in st.session_state.detector.trajectories:
            x, y = zip(*trajectory)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(color='#4CAF50', width=2), marker=dict(size=4)))
        
        fig.update_layout(
            title="Optical Flow Trajectories",
            xaxis_title="X position",
            yaxis_title="Y position",
            font=dict(color="#E0E0E0"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        container.plotly_chart(fig, use_container_width=True)
    else:
        container.info("No trajectories detected yet.")

if __name__ == "__main__":
    sidebar()
    main_content()