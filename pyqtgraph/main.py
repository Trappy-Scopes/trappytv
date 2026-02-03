# Simplified Video and DataFrame Synchronization Application
# Requirements: pip install pyqt6 pyqtgraph opencv-python pandas numpy

import sys
import cv2
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLineEdit, QLabel, QComboBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
from PyQt6.QtGui import QIcon
from collections import deque

class VideoDataSyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("trappytv")
        self.setGeometry(100, 100, 1600, 600)  # Made window bigger too
        self.setWindowIcon(QIcon('CR.png'))
        
        # Initialize variables
        self.dataframe = None
        self.video_path = None

        self.split_no = None
        self.gframe = None

        self.video_cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.frame_buffer = deque([], maxlen=500)
        self.fps = 25
        self.frameduration = (1/self.fps)*1000
        self.vidframesize = (760, 760)
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for automatic frame progression
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(self.frameduration))  # Update every 40ms
        
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # File input fields
        input_layout = QVBoxLayout()
        
        # Video file path
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video Path:"))
        self.video_path_edit = QLineEdit()
        self.video_path_edit.textChanged.connect(self.load_video)
        video_layout.addWidget(self.video_path_edit)
        input_layout.addLayout(video_layout)
        
        # Data file path
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data Path:"))
        self.data_path_edit = QLineEdit()
        self.data_path_edit.textChanged.connect(self.load_dataframe)
        data_layout.addWidget(self.data_path_edit)
        input_layout.addLayout(data_layout)
        
        main_layout.addLayout(input_layout)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left side - Larger square video display
        self.video_view = pg.ImageView()
        self.video_view.setFixedSize(*self.vidframesize)  # Much larger square (was 400x400)
        #self.video_view.ui.roiBtn.hide()
        self.video_view.ui.menuBtn.hide()
        self.video_view.ui.menuBtn.hide()
        self.video_view.ui.histogram.hide()

        content_layout.addWidget(self.video_view)
        
        # Right side - Data display and column selection
        right_layout = QVBoxLayout()
        
        # Column selection dropdowns
        self.column_combos = []
        for i in range(3):
            combo_layout = QHBoxLayout()
            combo_layout.addWidget(QLabel(f"Row {i+1} Column:"))
            combo = QComboBox()
            combo.currentTextChanged.connect(self.update_plots)
            self.column_combos.append(combo)
            combo_layout.addWidget(combo)
            right_layout.addLayout(combo_layout)
        
        # Three data plot rows
        self.plot_widgets = []
        for i in range(3):
            plot_widget = pg.PlotWidget()
            plot_widget.setFixedHeight(200)  # Made plots taller too
            right_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget)
        
        main_layout.addLayout(content_layout)
        
    def load_video(self):
        """Load video when path changes"""
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            return
            
        try:
            if self.video_cap:
                self.video_cap.release()
                
            self.video_cap = cv2.VideoCapture(video_path)
            if self.video_cap.isOpened():
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame = 0
                self.display_current_frame()
        except:
            pass
            
    def load_dataframe(self):
        """Load dataframe when path changes"""
        data_path = self.data_path_edit.text().strip()
        if not data_path:
            return
            
        try:
            if data_path.lower().endswith('.csv'):
                self.dataframe = pd.read_csv(data_path)
            elif data_path.lower().endswith(('.xlsx', '.xls')):
                self.dataframe = pd.read_excel(data_path)
            elif data_path.lower().endswith(('.hd5', '.hdf')):
                self.dataframe = pd.read_hdf(data_path)
            else:
                return
                
            # Populate column combo boxes
            numeric_columns = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()
            
            for combo in self.column_combos:
                combo.clear()
                combo.addItems(['None'] + numeric_columns)
                
            self.update_plots()
        except:
            pass
            
    def display_current_frame(self):
        """Display current video frame"""
        if not self.video_cap or self.total_frames == 0:
            return
            
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_cap.read()
        
        if ret:
            # Convert BGR to RGB and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_view.setImage(frame_rgb.transpose(1, 0, 2))
            
    def update_plots(self):
        """Update data plots with current frame sync"""
        if self.dataframe is None:
            return
            
        for i, plot_widget in enumerate(self.plot_widgets):
            combo = self.column_combos[i]
            column_name = combo.currentText()
            
            if column_name == 'None' or column_name not in self.dataframe.columns:
                plot_widget.clear()
                continue
                
            # Get data and sync with current frame
            data = self.dataframe[column_name].values
            x_data = np.arange(len(data))
            
            # Plot data
            plot_widget.clear()
            plot_widget.plot(x_data, data, pen='b')
            
            # Add vertical line at current frame position
            if self.current_frame < len(data):
                vline = pg.InfiniteLine(angle=90, pos=self.current_frame, pen='r')
                plot_widget.addItem(vline)
                
    def next_frame(self):
        """Move to next frame and sync data"""
        if self.total_frames == 0:
            return
            
        self.current_frame = (self.current_frame + 1) % self.total_frames
        self.display_current_frame()
        self.update_plots()
        
    def closeEvent(self, event):
        """Clean up on close"""
        if self.video_cap:
            self.video_cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoDataSyncApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()