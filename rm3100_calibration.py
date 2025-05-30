#!/usr/bin/env python3
"""
RM3100 Magnetometer Calibration Tool

This script performs comprehensive magnetometer calibration using collected data.
It fits an ellipsoid to the data and computes hard iron (offset) and soft iron 
(scale/rotation) correction matrices.

Usage:
    python rm3100_calibration.py [--input CSV_FILE] [--output OUTPUT_DIR]

The calibration process:
1. Loads magnetometer data from CSV
2. Fits an ellipsoid to the 3D data points
3. Computes calibration matrices
4. Generates calibrated data and validation plots
5. Outputs calibration parameters for flight software
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import argparse
import os
from datetime import datetime
import json

class MagnetometerCalibrator:
    def __init__(self):
        self.raw_data = None
        self.calibrated_data = None
        self.hard_iron_offset = None
        self.soft_iron_matrix = None
        self.calibration_quality = None
        
    def load_data(self, csv_file):
        """Load magnetometer data from CSV file"""
        try:
            # Try different possible column names
            df = pd.read_csv(csv_file)
            
            # Check for standard column names
            if 'X_uT' in df.columns:
                self.raw_data = df[['X_uT', 'Y_uT', 'Z_uT']].values
            elif 'M_x(µT)' in df.columns:
                self.raw_data = df[['M_x(µT)', 'M_y(µT)', 'M_z(µT)']].values
            else:
                # Try to auto-detect numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 3:
                    self.raw_data = df[numeric_cols[:3]].values
                    print(f"Auto-detected columns: {list(numeric_cols[:3])}")
                else:
                    raise ValueError("Could not find 3 numeric columns for X, Y, Z data")
            
            print(f"Loaded {len(self.raw_data)} data points from {csv_file}")
            print(f"Data range: X=[{self.raw_data[:,0].min():.2f}, {self.raw_data[:,0].max():.2f}]")
            print(f"           Y=[{self.raw_data[:,1].min():.2f}, {self.raw_data[:,1].max():.2f}]")
            print(f"           Z=[{self.raw_data[:,2].min():.2f}, {self.raw_data[:,2].max():.2f}]")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def ellipsoid_fit(self, data):
        """
        Fit an ellipsoid to 3D data points using least squares
        
        Ellipsoid equation: (x-x0)^T * A * (x-x0) = 1
        Where A is a 3x3 symmetric positive definite matrix
        """
        
        def ellipsoid_residuals(params, points):
            """Residual function for ellipsoid fitting"""
            # Extract parameters
            center = params[:3]  # x0, y0, z0
            # Symmetric matrix elements: [a11, a22, a33, a12, a13, a23]
            a11, a22, a33, a12, a13, a23 = params[3:9]
            
            # Build symmetric matrix A
            A = np.array([[a11, a12, a13],
                         [a12, a22, a23],
                         [a13, a23, a33]])
            
            # Compute residuals
            residuals = []
            for point in points:
                diff = point - center
                residual = diff.T @ A @ diff - 1.0
                residuals.append(residual)
            
            return np.array(residuals)
        
        # Initial guess for ellipsoid parameters
        center_guess = np.mean(data, axis=0)
        
        # Initial guess for A matrix (identity scaled by data variance)
        data_centered = data - center_guess
        scale = 1.0 / np.var(data_centered, axis=0)
        
        initial_params = np.concatenate([
            center_guess,
            [scale[0], scale[1], scale[2], 0, 0, 0]  # diagonal matrix initially
        ])
        
        print("Fitting ellipsoid to data...")
        
        # Perform least squares optimization
        result = least_squares(ellipsoid_residuals, initial_params, args=(data,))
        
        if not result.success:
            print("Warning: Ellipsoid fitting may not have converged properly")
        
        # Extract fitted parameters
        center = result.x[:3]
        a11, a22, a33, a12, a13, a23 = result.x[3:9]
        
        A = np.array([[a11, a12, a13],
                     [a12, a22, a23],
                     [a13, a23, a33]])
        
        return center, A
    
    def compute_calibration_matrices(self):
        """Compute calibration matrices from ellipsoid fit"""
        if self.raw_data is None:
            print("Error: No data loaded")
            return False
        
        print("\nComputing calibration matrices...")
        
        # Fit ellipsoid to raw data
        center, A = self.ellipsoid_fit(self.raw_data)
        
        # Hard iron offset is the ellipsoid center
        self.hard_iron_offset = center
        
        # Soft iron correction matrix
        # We want to transform the ellipsoid to a unit sphere
        # The transformation is: x_cal = M * (x_raw - offset)
        # Where M is derived from the ellipsoid matrix A
        
        try:
            # Eigenvalue decomposition of A
            eigenvals, eigenvecs = np.linalg.eigh(A)
            
            # Ensure all eigenvalues are positive
            if np.any(eigenvals <= 0):
                print("Warning: Non-positive eigenvalues detected, using absolute values")
                eigenvals = np.abs(eigenvals)
            
            # Compute the transformation matrix
            # M = V * sqrt(D) * V^T where D is diagonal matrix of eigenvalues
            D_sqrt = np.diag(np.sqrt(eigenvals))
            self.soft_iron_matrix = eigenvecs @ D_sqrt @ eigenvecs.T
            
            print(f"Hard iron offset: [{self.hard_iron_offset[0]:.3f}, {self.hard_iron_offset[1]:.3f}, {self.hard_iron_offset[2]:.3f}]")
            print("Soft iron matrix:")
            print(f"  [{self.soft_iron_matrix[0,0]:.6f}, {self.soft_iron_matrix[0,1]:.6f}, {self.soft_iron_matrix[0,2]:.6f}]")
            print(f"  [{self.soft_iron_matrix[1,0]:.6f}, {self.soft_iron_matrix[1,1]:.6f}, {self.soft_iron_matrix[1,2]:.6f}]")
            print(f"  [{self.soft_iron_matrix[2,0]:.6f}, {self.soft_iron_matrix[2,1]:.6f}, {self.soft_iron_matrix[2,2]:.6f}]")
            
            return True
            
        except np.linalg.LinAlgError as e:
            print(f"Error in matrix computation: {e}")
            return False
    
    def apply_calibration(self):
        """Apply calibration to raw data"""
        if self.raw_data is None or self.hard_iron_offset is None or self.soft_iron_matrix is None:
            print("Error: Calibration not computed yet")
            return False
        
        print("Applying calibration to data...")
        
        # Apply calibration: x_cal = M * (x_raw - offset)
        data_corrected = self.raw_data - self.hard_iron_offset
        self.calibrated_data = (self.soft_iron_matrix @ data_corrected.T).T
        
        return True
    
    def evaluate_calibration_quality(self):
        """Evaluate the quality of the calibration"""
        if self.calibrated_data is None:
            return
        
        # Compute distance from origin for calibrated data
        distances = np.linalg.norm(self.calibrated_data, axis=1)
        
        # Ideal case: all distances should be equal (perfect sphere)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Quality metric: lower std deviation relative to mean is better
        quality_score = std_distance / mean_distance
        
        self.calibration_quality = {
            'mean_radius': mean_distance,
            'std_radius': std_distance,
            'quality_score': quality_score,
            'sphericity': 1.0 - quality_score  # Higher is better (max 1.0 for perfect sphere)
        }
        
        print(f"\nCalibration Quality Assessment:")
        print(f"  Mean radius: {mean_distance:.3f} µT")
        print(f"  Radius std dev: {std_distance:.3f} µT")
        print(f"  Quality score: {quality_score:.4f} (lower is better)")
        print(f"  Sphericity: {self.calibration_quality['sphericity']:.4f} (higher is better)")
        
        if quality_score < 0.05:
            print("  ✓ Excellent calibration!")
        elif quality_score < 0.1:
            print("  ✓ Good calibration")
        elif quality_score < 0.2:
            print("  ⚠ Fair calibration - consider collecting more data")
        else:
            print("  ⚠ Poor calibration - more data needed or check sensor mounting")
    
    def plot_calibration_results(self, output_dir):
        """Create visualization plots of calibration results"""
        if self.raw_data is None or self.calibrated_data is None:
            return
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 8))
        
        # Raw data 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(self.raw_data[:,0], self.raw_data[:,1], self.raw_data[:,2], 
                   c='red', s=1, alpha=0.6)
        ax1.set_title('Raw Magnetometer Data')
        ax1.set_xlabel('X (µT)')
        ax1.set_ylabel('Y (µT)')
        ax1.set_zlabel('Z (µT)')
        
        # Force equal aspect ratio for raw data
        max_range = np.array([self.raw_data[:,0].max()-self.raw_data[:,0].min(),
                             self.raw_data[:,1].max()-self.raw_data[:,1].min(),
                             self.raw_data[:,2].max()-self.raw_data[:,2].min()]).max() / 2.0
        mid_x = (self.raw_data[:,0].max()+self.raw_data[:,0].min()) * 0.5
        mid_y = (self.raw_data[:,1].max()+self.raw_data[:,1].min()) * 0.5
        mid_z = (self.raw_data[:,2].max()+self.raw_data[:,2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        # Force equal box aspect
        ax1.set_box_aspect([1,1,1])
        
        # Calibrated data 3D plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(self.calibrated_data[:,0], self.calibrated_data[:,1], self.calibrated_data[:,2],
                   c='cyan', s=1, alpha=0.6)
        ax2.set_title('Calibrated Magnetometer Data')
        ax2.set_xlabel('X (µT)')
        ax2.set_ylabel('Y (µT)')
        ax2.set_zlabel('Z (µT)')
        
        # Force equal aspect ratio for calibrated data
        cal_max_range = np.array([self.calibrated_data[:,0].max()-self.calibrated_data[:,0].min(),
                                 self.calibrated_data[:,1].max()-self.calibrated_data[:,1].min(),
                                 self.calibrated_data[:,2].max()-self.calibrated_data[:,2].min()]).max() / 2.0
        cal_mid_x = (self.calibrated_data[:,0].max()+self.calibrated_data[:,0].min()) * 0.5
        cal_mid_y = (self.calibrated_data[:,1].max()+self.calibrated_data[:,1].min()) * 0.5
        cal_mid_z = (self.calibrated_data[:,2].max()+self.calibrated_data[:,2].min()) * 0.5
        ax2.set_xlim(cal_mid_x - cal_max_range, cal_mid_x + cal_max_range)
        ax2.set_ylim(cal_mid_y - cal_max_range, cal_mid_y + cal_max_range)
        ax2.set_zlim(cal_mid_z - cal_max_range, cal_mid_z + cal_max_range)
        # Force equal box aspect - THIS IS THE KEY!
        ax2.set_box_aspect([1,1,1])
        
        # Add reference sphere to calibrated plot
        if self.calibration_quality:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            r = self.calibration_quality['mean_radius']
            x_sphere = r * np.outer(np.cos(u), np.sin(v)) + cal_mid_x
            y_sphere = r * np.outer(np.sin(u), np.sin(v)) + cal_mid_y
            z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + cal_mid_z
            ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='yellow')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(output_dir, 'calibration_results.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Calibration plots saved to: {plot_filename}")
        
        plt.show()
    
    def save_calibration_parameters(self, output_dir):
        """Save calibration parameters in multiple formats"""
        if self.hard_iron_offset is None or self.soft_iron_matrix is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON for easy parsing
        json_data = {
            'timestamp': timestamp,
            'hard_iron_offset': self.hard_iron_offset.tolist(),
            'soft_iron_matrix': self.soft_iron_matrix.tolist(),
            'calibration_quality': self.calibration_quality,
            'calibration_formula': "x_cal = soft_iron_matrix * (x_raw - hard_iron_offset)"
        }
        
        json_filename = os.path.join(output_dir, f'rm3100_calibration_{timestamp}.json')
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as C header file for flight software
        header_filename = os.path.join(output_dir, f'rm3100_calibration_{timestamp}.h')
        with open(header_filename, 'w') as f:
            f.write(f"// RM3100 Magnetometer Calibration Parameters\n")
            f.write(f"// Generated on {timestamp}\n\n")
            f.write(f"#ifndef RM3100_CALIBRATION_H\n")
            f.write(f"#define RM3100_CALIBRATION_H\n\n")
            
            f.write(f"// Hard iron offset (µT)\n")
            f.write(f"static const float rm3100_hard_iron_offset[3] = {{\n")
            f.write(f"    {self.hard_iron_offset[0]:.6f}f,  // X offset\n")
            f.write(f"    {self.hard_iron_offset[1]:.6f}f,  // Y offset\n")
            f.write(f"    {self.hard_iron_offset[2]:.6f}f   // Z offset\n")
            f.write(f"}};\n\n")
            
            f.write(f"// Soft iron correction matrix\n")
            f.write(f"static const float rm3100_soft_iron_matrix[3][3] = {{\n")
            for i in range(3):
                f.write(f"    {{")
                for j in range(3):
                    f.write(f"{self.soft_iron_matrix[i,j]:.6f}f")
                    if j < 2:
                        f.write(f", ")
                f.write(f"}}")
                if i < 2:
                    f.write(f",")
                f.write(f"\n")
            f.write(f"}};\n\n")
            
            f.write(f"// Calibration function\n")
            f.write(f"// Usage: apply_calibration(raw_x, raw_y, raw_z, &cal_x, &cal_y, &cal_z)\n")
            f.write(f"static inline void rm3100_apply_calibration(float raw_x, float raw_y, float raw_z,\n")
            f.write(f"                                           float* cal_x, float* cal_y, float* cal_z) {{\n")
            f.write(f"    // Apply hard iron correction\n")
            f.write(f"    float corrected[3] = {{\n")
            f.write(f"        raw_x - rm3100_hard_iron_offset[0],\n")
            f.write(f"        raw_y - rm3100_hard_iron_offset[1],\n")
            f.write(f"        raw_z - rm3100_hard_iron_offset[2]\n")
            f.write(f"    }};\n\n")
            f.write(f"    // Apply soft iron correction matrix\n")
            f.write(f"    *cal_x = rm3100_soft_iron_matrix[0][0] * corrected[0] +\n")
            f.write(f"             rm3100_soft_iron_matrix[0][1] * corrected[1] +\n")
            f.write(f"             rm3100_soft_iron_matrix[0][2] * corrected[2];\n")
            f.write(f"    *cal_y = rm3100_soft_iron_matrix[1][0] * corrected[0] +\n")
            f.write(f"             rm3100_soft_iron_matrix[1][1] * corrected[1] +\n")
            f.write(f"             rm3100_soft_iron_matrix[1][2] * corrected[2];\n")
            f.write(f"    *cal_z = rm3100_soft_iron_matrix[2][0] * corrected[0] +\n")
            f.write(f"             rm3100_soft_iron_matrix[2][1] * corrected[1] +\n")
            f.write(f"             rm3100_soft_iron_matrix[2][2] * corrected[2];\n")
            f.write(f"}}\n\n")
            f.write(f"#endif // RM3100_CALIBRATION_H\n")
        
        # Save calibrated data CSV
        if self.calibrated_data is not None:
            cal_df = pd.DataFrame(self.calibrated_data, columns=['X_cal_uT', 'Y_cal_uT', 'Z_cal_uT'])
            cal_df['Magnitude_cal_uT'] = np.linalg.norm(self.calibrated_data, axis=1)
            
            csv_filename = os.path.join(output_dir, f'rm3100_calibrated_data_{timestamp}.csv')
            cal_df.to_csv(csv_filename, index=False)
            print(f"Calibrated data saved to: {csv_filename}")
        
        print(f"Calibration parameters saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  C Header: {header_filename}")
    
    def run_full_calibration(self, csv_file, output_dir="calibration_output"):
        """Run the complete calibration process"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("RM3100 MAGNETOMETER CALIBRATION")
        print("=" * 60)
        
        # Load data
        if not self.load_data(csv_file):
            return False
        
        # Check data quality
        if len(self.raw_data) < 100:
            print("Error: Insufficient data points for calibration (need at least 100)")
            return False
        
        # Compute calibration
        if not self.compute_calibration_matrices():
            return False
        
        # Apply calibration
        if not self.apply_calibration():
            return False
        
        # Evaluate quality
        self.evaluate_calibration_quality()
        
        # Save results
        self.save_calibration_parameters(output_dir)
        
        # Create plots
        self.plot_calibration_results(output_dir)
        
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE!")
        print("=" * 60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='RM3100 Magnetometer Calibration')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with magnetometer data')
    parser.add_argument('--output', type=str, default='calibration_output', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    calibrator = MagnetometerCalibrator()
    calibrator.run_full_calibration(args.input, args.output)

if __name__ == "__main__":
    main()