import streamlit as st
import socket
import struct
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
import time

from physics_scaler import PhysicsScaler
from electrical_analyzer import ElectricalAnalyzer


# Features 9 

def extract_features(signal_window):
    return {
        'max': np.max(signal_window),
        'min': np.min(signal_window),
        'mean': np.mean(signal_window),
        'sd': np.std(signal_window),
        'rms': np.sqrt(np.mean(signal_window**2)),
        'skewness': skew(signal_window),
        'kurtosis': kurtosis(signal_window),
        'crest': np.max(signal_window) / np.sqrt(np.mean(signal_window**2)) if np.sqrt(np.mean(signal_window**2)) > 0 else 0,
        'form': np.sqrt(np.mean(signal_window**2)) / np.mean(np.abs(signal_window)) if np.mean(np.abs(signal_window)) > 0 else 0
    }


# Dashboard UI 
st.set_page_config(page_title="Real Time fault Diagnosis System", page_icon="", layout="wide")

st.title("Real Time fault Diagnosis System")
st.markdown("### Real-Time Fault Diagnosis System from MATLAB Simulink")
st.markdown("---")

# Sidebar
st.sidebar.header(" Motor Nameplate Data")
rated_hp = st.sidebar.number_input("Rated HP ", value=0, format="%.2f")
rated_rpm = st.sidebar.number_input("Rated RPM ", value=0, format="%.2f")
rated_current = st.sidebar.number_input("Rated Current (A)", value=0, format="%.2f")

# AI Model Load
@st.cache_resource
def load_model():
    try:
        return joblib.load('rf_bearing_model.pkl')
    except Exception as e:
        st.error(f"Model Loading error: {e}")
        return None

rf_model = load_model()


# button  Layout

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    start_button = st.button(" Start Monitoring", use_container_width=True)
with col_btn2:
    stop_button = st.button(" Stop Monitoring", use_container_width=True)

st.markdown("---")

#  Live Metrics 
st.markdown("####  Live Parameters")
met1, met2, met3, met4 = st.columns(4)
rpm_metric = met1.empty()
ia_metric = met2.empty()
ib_metric = met3.empty()
ic_metric = met4.empty()

st.markdown("---")

#  Live Waveforms
st.markdown("####  Live Signals (Waveforms)")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**3-Phase Current Trend (A)**")
    current_chart = st.empty()
with chart_col2:
    st.markdown("**Vibration Signal (g)**")
    vibration_chart = st.empty()

st.markdown("---")

#  Diagnostic Status
st.markdown("####  Diagnostics Status")
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.subheader(" Electrical Status")
    elec_status_placeholder = st.empty()
with status_col2:
    st.subheader(" Mechanical Status")
    mech_status_placeholder = st.empty()
with status_col3:
    st.subheader(" Bearing fault Prediction")
    ai_status_placeholder = st.empty()


# Main Loop

MAX_HISTORY = 100 
current_history = {'Ia': [], 'Ib': [], 'Ic': []}

if start_button and rf_model:
    st.success("Listening to MATLAB on 172.20.10.2:5005...")
    
  
    UDP_IP = "172.20.10.2"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(2.0) 
    
    # Classes Initialize 
    scaler = PhysicsScaler(base_hp=2.0, base_rpm=1797.0)
    analyzer = ElectricalAnalyzer(rated_current=rated_current)
    
    buffer_size = 2048
    vibration_buffer = []
    
    try:
        while not stop_button:
            try:
               
                data, addr = sock.recvfrom(1024)
                if len(data) == 40:
                    unpacked_data = struct.unpack('<5d', data)
                    i_a, i_b, i_c, rpm_rad, vib = unpacked_data
                    
                    
                    current_rpm = rpm_rad * (30 / np.pi)
                    
                    #  Buffer
                    vibration_buffer.append(vib)

                    current_history['Ia'].append(i_a)
                    current_history['Ib'].append(i_b)
                    current_history['Ic'].append(i_c)

                    # update chart
                    if len(current_history['Ia']) > MAX_HISTORY:
                        current_history['Ia'].pop(0)
                        current_history['Ib'].pop(0)
                        current_history['Ic'].pop(0)
                    
                    # Buffer 2048  UI Update 
                    if len(vibration_buffer) >= buffer_size:
                        # 1. Electrical Analysis
                        elec_results = analyzer.analyze_currents(i_a, i_b, i_c)
                        
                        # 2. Mechanical/AI Analysis
                        features = extract_features(np.array(vibration_buffer))
                        scaled_features = scaler.scale_down_to_baseline(features, rated_hp, rated_rpm)
                        

                        features_df = pd.DataFrame([scaled_features])
                        prediction = rf_model.predict(features_df)[0]
                        
                        #  Dashboard 
                        
                        # 1. updte Metrics 
                        rpm_metric.metric(label="Rotor Speed", value=f"{current_rpm:.0f} RPM")
                        ia_metric.metric(label="Phase A Current", value=f"{i_a:.2f} A")
                        ib_metric.metric(label="Phase B Current", value=f"{i_b:.2f} A")
                        ic_metric.metric(label="Phase C Current", value=f"{i_c:.2f} A")

                        # 2. update Waveforms
                       
                        df_currents = pd.DataFrame(current_history)
                        current_chart.line_chart(df_currents, color=["#FF4B4B", "#26A69A", "#4C78A8"]) 
                        
                        
                        df_vib = pd.DataFrame(vibration_buffer, columns=['Vibration'])
                        vibration_chart.line_chart(df_vib, color=["#FFA500"])
                        
                        
                        if elec_results['overload_fault']:
                            elec_status_placeholder.error(f" OVERLOAD FAULT! (Avg: {elec_results['i_avg']}A)")
                        elif elec_results['unbalance_fault']:
                            elec_status_placeholder.warning(f" Phase Unbalance! ({elec_results['unbalance_percent']:.1f}%)")
                        else:
                            elec_status_placeholder.success(f" Electrical Normal (Avg: {elec_results['i_avg']}A)")
                            
                        
                        mech_status_placeholder.info(f"Vibration RMS: {features['rms']:.4f}g")
                        
                        if prediction == 'Normal':
                            ai_status_placeholder.success(" Bearing: Normal")
                        elif 'Inner' in prediction or 'IR' in prediction:
                            ai_status_placeholder.error(" Bearing: Inner Race Fault")
                        elif 'Outer' in prediction or 'OR' in prediction:
                            ai_status_placeholder.error(" Bearing: Outer Race Fault")
                        else:
                            ai_status_placeholder.warning(f" Bearing: {prediction}")
                            
                        # emty Buffer for next loop 
                        vibration_buffer = []
                        
            except socket.timeout:
                st.warning("Stoped Receiving Data Frome MATLAB.Please Run The Simulink.")
                break
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        sock.close()