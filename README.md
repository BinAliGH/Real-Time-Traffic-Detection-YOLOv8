# 🚦 Advanced Traffic Command Center

### 📌 Project Overview
This project is an end-to-end **Real-Time Traffic Analysis System** built using **YOLOv8** for computer vision and **Streamlit** for the interactive dashboard. 

Unlike simple detection scripts, this application functions as a complete "Command Center." It allows users to upload raw traffic footage and instantly receive actionable insights, including vehicle counting, flow rates, and classification trends.

### 🎥 Demo / Screenshots
*(Add a screenshot of your dashboard here later)*

### 🛠️ Tech Stack
* **Core Logic:** Python
* **Computer Vision:** YOLOv8 (Ultralytics) + OpenCV
* **Dashboard & UI:** Streamlit
* **Data Visualization:** Plotly Express (Interactive Charts)
* **Data Manipulation:** Pandas & NumPy

### 🚀 Key Features
* **Smart Object Tracking:** Uses YOLOv8 with ByteTrack to persistently track vehicles across frames.
* **Line-Crossing Counting:** Implements vector logic to count vehicles only when they cross a user-defined position.
* **Interactive Dashboard:**
    * **Live KPIs:** Displays Total Traffic, Flow Rate (vehicles/min), and Dominant Vehicle Type.
    * **Trend Analysis:** Real-time Area Chart showing traffic volume spikes over the last 10 minutes.
    * **Classification:** Dynamic Bar Chart comparing counts of Cars vs. Trucks vs. Buses.
* **Optimized Performance:**
    * Uses `@st.cache_resource` to load the AI model only once.
    * Implements **chunk-based file reading** to handle large video files without crashing memory (RAM).
* **Customizable Settings:** Adjustable detection line position via a sidebar slider.

### ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/Real-Time-Traffic-Detection-YOLOv8.git](https://github.com/YourUsername/Real-Time-Traffic-Detection-YOLOv8.git)
cd Real-Time-Traffic-Detection-YOLOv8
