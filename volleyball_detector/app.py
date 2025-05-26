from controllers.detection_controller import process_video
from views.ui_components import render_uploader, render_results
from models.player_model import Player
import streamlit as st
import os

def main():
    st.set_page_config(
        page_title="Volleyball Performance Analyzer",
        layout="wide",
        page_icon="🏐"
    )
    
    st.title("🏐 Volleyball Jersey Number Tracker")
    st.markdown("Upload a short clip (12-30s) to label player jersey numbers")
    
    # File upload and settings
    video_path = render_uploader()
    
    if video_path:
        if st.button("Process Video", type="primary"):
            with st.spinner("Detecting players and reading jersey numbers..."):
                # Process video through controller
                player_data = process_video(video_path)
                
                # Perform OCR on detected players
                
                # Store in session state
                st.session_state.player_data = player_data
                st.success(f"Detected {len(player_data)} player instances")
        
        # Display results if available
        if 'player_data' in st.session_state:
            render_results(st.session_state.player_data)

if __name__ == "__main__":
    main()