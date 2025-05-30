#!/usr/bin/env python3
"""
RM3100 Magnetometer Data Logger and Real-time Plotter

This script reads magnetometer data from the RM3100 sensor via serial port,
saves it to CSV files, and provides real-time plotting for calibration purposes.

Usage:
    python rm3100_logger.py [--port PORT] [--baudrate BAUDRATE] [--output OUTPUT]

Features:
    - Real-time data logging to CSV
    - Live 3D magnetometer data visualization
    - Automatic calibration data collection
    - Sphere fitting for hard/soft iron correction
"""

import serial
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import argparse
import threading
import queue
from datetime import datetime
import os
import sys

class RM3100Logger:
    def __init__(self, port, baudrate=115200, output_dir="rm3100_data"):
        self.port = port
        self.baudrate = baudrate
        self.output_dir = output_dir
        self.data_queue = queue.Queue()
        self.running = False
        self.serial_conn = None
        
        # Data storage (unlimited points)
        self.x_data = []
        self.y_data = []
        self.z_data = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(output_dir, f"rm3100_data_{timestamp}.csv")
        
        print(f"Data will be saved to: {self.csv_filename}")
        
    def connect_serial(self):
        """Connect to the serial port"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            time.sleep(2)  # Allow connection to stabilize
            return True
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return False
            
    def disconnect_serial(self):
        """Disconnect from serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")
            
    def parse_data_line(self, line):
        """Parse a CSV line from the magnetometer"""
        try:
            # Expected format: Timestamp(ms),M_x(µT),M_y(µT),M_z(µT),M_total(µT)
            parts = line.strip().split(',')
            if len(parts) == 5:
                timestamp = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                magnitude = float(parts[4])
                return timestamp, x, y, z, magnitude
        except (ValueError, IndexError):
            pass
        return None
        
    def serial_reader_thread(self):
        """Thread function to read serial data"""
        csv_file = None
        csv_writer = None
        
        try:
            # Open CSV file for writing
            csv_file = open(self.csv_filename, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Timestamp_ms', 'X_uT', 'Y_uT', 'Z_uT', 'Magnitude_uT', 'System_Time'])
            
            print("Starting data collection...")
            
            while self.running:
                try:
                    if self.serial_conn and self.serial_conn.in_waiting:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore')
                        
                        # Skip header lines and error messages
                        if line.startswith('#') or line.startswith('ERROR') or line.startswith('RM3100') or 'initialized' in line:
                            print(f"Info: {line.strip()}")
                            continue
                            
                        parsed_data = self.parse_data_line(line)
                        if parsed_data:
                            timestamp, x, y, z, magnitude = parsed_data
                            system_time = datetime.now().isoformat()
                            
                            # Write to CSV
                            csv_writer.writerow([timestamp, x, y, z, magnitude, system_time])
                            csv_file.flush()  # Ensure data is written immediately
                            
                            # Add to plot queue (no size limit for unlimited points)
                            self.data_queue.put((timestamp, x, y, z, magnitude))
                            
                            print(f"Data: t={timestamp}, x={x:.2f}, y={y:.2f}, z={z:.2f}, |B|={magnitude:.2f} µT")
                            
                except serial.SerialException as e:
                    print(f"Serial error: {e}")
                    break
                    
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\nStopping data collection...")
        finally:
            if csv_file:
                csv_file.close()
                print(f"Data saved to {self.csv_filename}")
                
    def update_plot_data(self):
        """Update plot data from queue"""
        new_data = False
        while not self.data_queue.empty():
            try:
                timestamp, x, y, z, magnitude = self.data_queue.get_nowait()
                self.x_data.append(x)
                self.y_data.append(y)
                self.z_data.append(z)
                new_data = True
                    
            except queue.Empty:
                break
        return new_data
        
    def setup_plots(self):
        """Setup the matplotlib plots"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(8, 8))
        
        # Single 3D scatter plot for calibration visualization
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_title('3D Magnetometer Calibration Data')
        self.ax_3d.set_xlabel('X (µT)')
        self.ax_3d.set_ylabel('Y (µT)')
        self.ax_3d.set_zlabel('Z (µT)')
        
        plt.tight_layout()
        
    def update_plots(self):
        """Update 3D plot with new data - lightweight version"""
        if len(self.x_data) < 2:
            return
            
        # Clear and redraw 3D plot efficiently
        self.ax_3d.clear()
        
        # Use all points but with efficient plotting
        if len(self.x_data) > 0:
            # For large datasets, use simpler plotting without color gradients
            if len(self.x_data) > 5000:
                # Sample points for display to keep it responsive
                step = max(1, len(self.x_data) // 5000)
                x_plot = self.x_data[::step]
                y_plot = self.y_data[::step]
                z_plot = self.z_data[::step]
            else:
                x_plot = self.x_data
                y_plot = self.y_data
                z_plot = self.z_data
            
            # Simple scatter plot without color gradient for speed
            self.ax_3d.scatter(x_plot, y_plot, z_plot, c='cyan', s=1, alpha=0.6)
            
            self.ax_3d.set_title(f'3D Magnetometer Data ({len(self.x_data)} points)')
            self.ax_3d.set_xlabel('X (µT)')
            self.ax_3d.set_ylabel('Y (µT)')
            self.ax_3d.set_zlabel('Z (µT)')
            
        plt.tight_layout()
        
    def run_realtime_plot(self):
        """Run the real-time plotting"""
        if not self.connect_serial():
            return
            
        self.running = True
        
        # Start serial reading thread
        serial_thread = threading.Thread(target=self.serial_reader_thread)
        serial_thread.daemon = True
        serial_thread.start()
        
        # Setup plots
        self.setup_plots()
        plt.ion()
        plt.show()
        
        print("\nReal-time 3D plotting started!")
        print("For calibration: slowly rotate the sensor in all orientations")
        print("You should see a sphere-like pattern forming in 3D")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                if self.update_plot_data():
                    self.update_plots()
                    plt.pause(0.05)  # Faster refresh for better responsiveness
                else:
                    plt.pause(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            
        finally:
            self.running = False
            self.disconnect_serial()
            plt.ioff()
            plt.show()  # Keep final plot open
            
    def analyze_calibration_data(self):
        """Analyze collected data for calibration purposes"""
        if len(self.x_data) < 100:
            print("Insufficient data for calibration analysis")
            return
            
        x = np.array(self.x_data)
        y = np.array(self.y_data)
        z = np.array(self.z_data)
        
        print("\n=== Calibration Analysis ===")
        print(f"Data points collected: {len(x)}")
        print(f"X range: {x.min():.2f} to {x.max():.2f} µT (span: {x.max()-x.min():.2f} µT)")
        print(f"Y range: {y.min():.2f} to {y.max():.2f} µT (span: {y.max()-y.min():.2f} µT)")
        print(f"Z range: {z.min():.2f} to {z.max():.2f} µT (span: {z.max()-z.min():.2f} µT)")
        
        # Calculate center estimates (hard iron bias)
        center_x = (x.max() + x.min()) / 2
        center_y = (y.max() + y.min()) / 2
        center_z = (z.max() + z.min()) / 2
        
        print(f"\nEstimated hard iron bias:")
        print(f"  X offset: {center_x:.2f} µT")
        print(f"  Y offset: {center_y:.2f} µT") 
        print(f"  Z offset: {center_z:.2f} µT")
        
        # Calculate scale factors (soft iron correction estimate)
        span_x = x.max() - x.min()
        span_y = y.max() - y.min()
        span_z = z.max() - z.min()
        avg_span = (span_x + span_y + span_z) / 3
        
        scale_x = avg_span / span_x if span_x > 0 else 1.0
        scale_y = avg_span / span_y if span_y > 0 else 1.0
        scale_z = avg_span / span_z if span_z > 0 else 1.0
        
        print(f"\nEstimated soft iron scale factors:")
        print(f"  X scale: {scale_x:.3f}")
        print(f"  Y scale: {scale_y:.3f}")
        print(f"  Z scale: {scale_z:.3f}")
        
        print(f"\nCalibration matrix (simplified):")
        print(f"X_cal = {scale_x:.3f} * (X_raw - {center_x:.2f})")
        print(f"Y_cal = {scale_y:.3f} * (Y_raw - {center_y:.2f})")
        print(f"Z_cal = {scale_z:.3f} * (Z_raw - {center_z:.2f})")

def find_serial_ports():
    """Find available serial ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def main():
    parser = argparse.ArgumentParser(description='RM3100 Magnetometer Data Logger')
    parser.add_argument('--port', type=str, help='Serial port (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--output', type=str, default='rm3100_data', help='Output directory')
    
    args = parser.parse_args()
    
    # Auto-detect serial port if not specified
    if not args.port:
        available_ports = find_serial_ports()
        if not available_ports:
            print("No serial ports found!")
            sys.exit(1)
            
        print("Available serial ports:")
        for i, port in enumerate(available_ports):
            print(f"  {i}: {port}")
            
        if len(available_ports) == 1:
            args.port = available_ports[0]
            print(f"Auto-selecting: {args.port}")
        else:
            try:
                choice = int(input("Select port number: "))
                args.port = available_ports[choice]
            except (ValueError, IndexError):
                print("Invalid selection")
                sys.exit(1)
    
    # Create logger and run
    logger = RM3100Logger(args.port, args.baudrate, args.output)
    
    try:
        logger.run_realtime_plot()
        logger.analyze_calibration_data()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()