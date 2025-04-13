# Required Libraries
import streamlit as st
import numpy as np
import cv2 # Use cv2 consistently
import io
import time
import random
import pandas as pd # Keep pandas for easy data handling in experiments
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # For DLT visualization
import os
import scipy.io # <--- ADDED for loading .mat files
from numpy.linalg import inv, svd, det # <--- Specific imports for DLT functions
from scipy.linalg import rq      # <--- Specific imports for DLT functions


# ==============================================================================
# Part 0: Configuration & Constants
# ==============================================================================
DEFAULT_PATTERN_SIZE = (7, 6) # (cols, rows) for OpenCV findChessboardCorners
DEFAULT_SQUARE_SIZE = 25.0

# ==============================================================================
# Part 1: Utility Functions
# ==============================================================================

def decode_image(file_uploader_content):
    """Decodes image from file uploader bytes."""
    try:
        if hasattr(file_uploader_content, 'seek'): file_uploader_content.seek(0)
        file_bytes = np.asarray(bytearray(file_uploader_content.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

def load_mat_file(uploaded_file):
    """Loads data safely from an uploaded .mat file."""
    if uploaded_file is None: return None
    try:
        uploaded_file.seek(0)
        # Use BytesIO for compatibility with file-like object from Streamlit
        data = scipy.io.loadmat(io.BytesIO(uploaded_file.read()))
        return data
    except Exception as e:
        st.error(f"Error loading MAT file '{getattr(uploaded_file, 'name', '')}': {e}")
        return None

# Helper function for rerun support
def rerun_if_possible():
    """Attempts to rerun the Streamlit app."""
    try:
        st.rerun() # Preferred method in newer Streamlit versions
    except AttributeError:
        try:
            st.experimental_rerun() # Older method
        except AttributeError:
            st.warning("Auto-rerun functionality not detected. Please refresh the page manually if needed.")


# ==============================================================================
# Part 2: Core Calibration Logic Functions (Standard & Fisheye)
# ==============================================================================
def process_images(image_files, pattern_size_cv, square_size):
    """ Processes uploaded image files to find checkerboard corners. """
    # pattern_size_cv = (cols, rows) for findChessboardCorners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size_cv[0] * pattern_size_cv[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size_cv[0], 0:pattern_size_cv[1]].T.reshape(-1, 2)
    objp *= square_size # Scale by square size in mm

    objpoints_all = [] # 3d point in real world space
    imgpoints_all = [] # 2d points in image plane.
    image_shapes = [] # Store shapes of processed images
    valid_image_indices = [] # Indices of images where corners were found

    if not image_files:
        st.warning("No image files provided for processing.")
        return None, None, None, None

    # --- Progress Reporting ---
    progress_text_area = st.empty()
    prog_bar = st.progress(0)
    total_files = len(image_files)
    processed_count = 0
    found_count = 0

    for i, uploaded_file in enumerate(image_files):
        file_name = getattr(uploaded_file, 'name', f'image_{i+1}')
        progress_text_area.text(f"Processing {i+1}/{total_files}: {file_name}...")

        try:
            img = decode_image(uploaded_file)
            if img is None:
                st.warning(f"Skipping file {i+1}: Could not decode image.")
                continue # Skip this file

            processed_count += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            current_shape = gray.shape[::-1] # Get (width, height)
            image_shapes.append(current_shape)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size_cv, None)

            # If found, add object points, image points (after refining them)
            if ret:
                found_count += 1
                valid_image_indices.append(i)
                objpoints_all.append(objp)

                # Refine corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints_all.append(corners2)
            # else:
                # Optional: Draw corners or indicate failure for specific images
                # cv2.drawChessboardCorners(img, pattern_size_cv, corners2, ret)
                # st.image(img, caption=f"Corners found in {file_name}")
                # pass # Or add a note that corners weren't found

        except Exception as e:
            st.warning(f"Error processing image {i+1} ({file_name}): {e}")

        # Update progress bar
        prog_bar.progress((i + 1) / total_files)

    # Clear progress elements
    prog_bar.empty()
    progress_text_area.empty()

    if not imgpoints_all:
        st.error("Checkerboard corners could not be detected in any of the uploaded images.")
        return None, None, None, None

    st.success(f"Found corners in {found_count} out of {processed_count} processed images.")

    # Check for consistent image dimensions among valid images
    valid_shapes = [sh for idx, sh in enumerate(image_shapes) if idx in valid_image_indices]
    first_shape = valid_shapes[0] if valid_shapes else None
    if len(set(valid_shapes)) > 1:
        st.warning("Inconsistent image dimensions detected among valid images. Calibration might be affected. Using shape of the first valid image.")

    return objpoints_all, imgpoints_all, first_shape, valid_image_indices

def run_standard_calibration(objpoints, imgpoints, gray_shape):
    """Performs standard camera calibration using cv2.calibrateCamera."""
    if not objpoints or not imgpoints or gray_shape is None or len(objpoints) != len(imgpoints):
        print("Standard Calibration Preconditions Failed: Check inputs.")
        return False, None, None, None, None
    if len(objpoints) < 2: # Need at least 2 views for standard calibration typically
        print("Standard Calibration Failed: Need at least 2 views.")
        return False, None, None, None, None

    try:
        # Ensure objpoints are float32, imgpoints are float32
        objpoints_f32 = [op.astype(np.float32) for op in objpoints]
        imgpoints_f32 = [ip.astype(np.float32) for ip in imgpoints]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_f32, imgpoints_f32, gray_shape, None, None)

        if not ret or mtx is None or dist is None:
            print("cv2.calibrateCamera returned False or None.")
            return False, None, None, None, None

        return ret, mtx, dist, rvecs, tvecs
    except cv2.error as e:
        print(f"OpenCV Error during Standard Calibration: {e}")
        # Provide more specific feedback if possible based on the error type
        if "solvePnP" in str(e) or "more points needed" in str(e):
            print(" -> Check if enough points per view or enough views are provided.")
        return False, None, None, None, None
    except Exception as e:
        print(f"Unexpected Error during Standard Calibration: {e}")
        return False, None, None, None, None

def run_fisheye_calibration(objpoints, imgpoints, gray_shape):
    """Performs fisheye camera calibration using cv2.fisheye.calibrate."""
    if not objpoints or not imgpoints or gray_shape is None or len(objpoints) != len(imgpoints):
        print("Fisheye Calibration Preconditions Failed: Check inputs.")
        return False, None, None, None, None
    if len(objpoints) < 2: # Need multiple views
         print("Fisheye Calibration Failed: Need at least 2 views.")
         return False, None, None, None, None

    N = len(objpoints)
    # Initialize K and D matrices as required by the function
    K_init = np.zeros((3, 3), dtype=np.float64)
    D_init = np.zeros((4, 1), dtype=np.float64) # Fisheye uses 4 distortion coeffs (k1, k2, k3, k4)

    # Fisheye calibration requires points in specific shapes and types
    # Reshape objpoints to (N, 1, n_points, 3) and imgpoints to (N, 1, n_points, 2)
    # Or pass lists of (n_points, 1, 3) and (n_points, 1, 2)
    try:
        objp_f64 = [op.reshape(-1, 1, 3).astype(np.float64) for op in objpoints]
        imgp_f64 = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in imgpoints]

        # Set calibration flags
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        # Termination criteria for the optimization process
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objp_f64, imgp_f64, gray_shape, K_init, D_init,
            flags=flags, criteria=criteria
        )

        if not ret or K is None or D is None:
             print("cv2.fisheye.calibrate returned False or None.")
             return False, None, None, None, None

        return ret, K, D, rvecs, tvecs
    except cv2.error as e:
        print(f"OpenCV Error during Fisheye Calibration: {e}")
        # Example: Check condition number error often means poor view variety
        if "CALIB_CHECK_COND" in str(e):
             print(" -> Check Calibration Condition Error: Ensure views have sufficient variety (angles, positions).")
        return False, None, None, None, None
    except Exception as e:
        print(f"Unexpected Error during Fisheye Calibration: {e}")
        return False, None, None, None, None


def calculate_reprojection_error(objpoints_full, imgpoints_full, mtx, dist, rvecs_calib, tvecs_calib, is_fisheye=False):
    """Calculates the Root Mean Square (RMS) reprojection error."""
    total_err_sq = 0.0
    total_pts = 0

    if not objpoints_full or not imgpoints_full or len(objpoints_full) != len(imgpoints_full):
        print("Reprojection Error Calc: Input lists invalid or mismatched lengths.")
        return np.inf

    n_views = len(objpoints_full)

    # Ensure rvecs and tvecs lists match the number of views if provided
    # If not provided (e.g., for standard evaluation where solvePnP is used internally), create lists of Nones
    rvecs = rvecs_calib if rvecs_calib is not None and len(rvecs_calib) == n_views else [None] * n_views
    tvecs = tvecs_calib if tvecs_calib is not None and len(tvecs_calib) == n_views else [None] * n_views

    for i in range(n_views):
        objp = np.asarray(objpoints_full[i], dtype=np.float32) # Ensure float32
        imgp_det = np.asarray(imgpoints_full[i], dtype=np.float32).reshape(-1, 1, 2) # Detected points

        if len(objp) == 0: continue # Skip if no points for this view

        imgp_proj = None # Initialize projected points for this view

        try:
            rvec = rvecs[i]
            tvec = tvecs[i]

            # --- Project points based on camera model ---
            if is_fisheye:
                # Fisheye projection requires rvec/tvec from calibration or re-estimation
                if rvec is not None and tvec is not None:
                    # Ensure correct types/shapes for fisheye.projectPoints
                    objp_f64 = objp.reshape(-1, 1, 3).astype(np.float64)
                    rvec_f64 = np.asarray(rvec).astype(np.float64).reshape(1, 3) # Reshape needed? check docs
                    tvec_f64 = np.asarray(tvec).astype(np.float64).reshape(1, 3) # Reshape needed? check docs
                    mtx_f64 = np.asarray(mtx).astype(np.float64)
                    dist_f64 = np.asarray(dist).astype(np.float64)

                    imgp_proj, _ = cv2.fisheye.projectPoints(objp_f64, rvec_f64, tvec_f64, mtx_f64, dist_f64)
                else:
                    # print(f"Skipping view {i} for fisheye error: Missing pose (rvec/tvec).")
                    continue # Cannot calculate error without pose

            else: # Standard camera model
                # If rvec/tvec are provided (from calibration), use them directly.
                # Otherwise (e.g., evaluating on full dataset after subset calib), estimate pose using solvePnP.
                if rvec is not None and tvec is not None:
                    imgp_proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
                else:
                    # Estimate pose if not provided
                    if len(objp) <= 3:
                        # print(f"Skipping view {i} for std error: Need > 3 points for solvePnP.")
                        continue # solvePnP requires more than 3 points
                    # Ensure points are float64 for solvePnP is often safer, check objp/imgp types
                    ret_pnp, rvec_pnp, tvec_pnp = cv2.solvePnP(objp, imgp_det, mtx, dist)
                    if ret_pnp:
                        imgp_proj, _ = cv2.projectPoints(objp, rvec_pnp, tvec_pnp, mtx, dist)
                    else:
                        # print(f"Skipping view {i} for std error: solvePnP failed.")
                        continue # Pose estimation failed

            # --- Calculate error for this view ---
            if imgp_proj is not None:
                 # Ensure imgp_proj has the same shape as imgp_det (N, 1, 2)
                imgp_proj = imgp_proj.reshape(-1, 1, 2).astype(np.float32)

                if imgp_proj.shape == imgp_det.shape:
                    # Calculate squared Euclidean distance
                    err_sq = np.sum((imgp_det - imgp_proj)**2, axis=(1,2)) # Sum over x,y coords
                    total_err_sq += np.sum(err_sq)
                    total_pts += len(objp)
                # else:
                    # print(f"Shape mismatch view {i}: Detected {imgp_det.shape}, Projected {imgp_proj.shape}")


        except cv2.error as e:
            # print(f"OpenCV Error during Reprojection (View {i}): {e}")
            pass # Continue to next image
        except Exception as e:
            # print(f"Unexpected Error during Reprojection (View {i}): {e}")
            pass # Continue to next image

    if total_pts == 0:
        print("Reprojection Error Calc: No points were successfully projected.")
        return np.inf # Avoid division by zero

    mean_error_sq = total_err_sq / total_pts
    rms_error = np.sqrt(mean_error_sq)
    return rms_error


# ==============================================================================
# Part 3: DLT Calibration Functions
# ==============================================================================
# These functions are based on the provided code structure.
# Added more robust error checking and handling.

def dlt_calibrate_intrinsics(pts_2D, cam_pts_3D):
    """Estimates intrinsic matrix K using DLT assuming known extrinsics (3D points are in camera frame)."""
    if pts_2D is None or cam_pts_3D is None:
        st.error("DLT Intrinsics Error: Input points are None.")
        return None
    if len(pts_2D) < 6 or len(pts_2D) != len(cam_pts_3D):
        st.error(f"DLT Intrinsics Error: Need at least 6 corresponding points. Got {len(pts_2D)} pairs.")
        return None
    if pts_2D.shape[1] != 2 or cam_pts_3D.shape[1] != 3:
        st.error(f"DLT Intrinsics Error: Incorrect input shapes. pts_2D should be Nx2, cam_pts_3D should be Nx3. Got {pts_2D.shape} and {cam_pts_3D.shape}.")
        return None

    try:
        pts_2DT = pts_2D.T       # Transpose: 2xN
        cam_pts_3DT = cam_pts_3D.T # Transpose: 3xN
        num_points = pts_2D.shape[0]
        one = np.ones((num_points, 1))

        # Build matrix A = [u, v, 1]' for all points (Nx3) -> Transposed AT (3xN)
        A = np.hstack((pts_2D, one))
        AT = A.T # 3xN

        # Build matrix B = [X/Z, Y/Z, 1]' for all points (Nx3) -> Transposed BT (3xN)
        x1 = np.empty((num_points, 1))
        x2 = np.empty((num_points, 1))
        for i in range(num_points):
            z_val = cam_pts_3DT[2, i] # Access Z coordinate correctly
            # Avoid division by zero or very small numbers
            safe_z = np.sign(z_val) * max(abs(z_val), 1e-8)
            x1[i] = cam_pts_3DT[0, i] / safe_z # X/Z
            x2[i] = cam_pts_3DT[1, i] / safe_z # Y/Z

        B = np.hstack((x1, x2, one))
        BT = B.T # 3xN

        # Solve K = A * pinv(B) => K' = pinv(B') * A' (using transposes)
        # K' (3x3) = Binv (3xN) @ AT' (Nx3) -- Check dimensions
        # Let's stick to K = A * pinv(B). K = A @ np.linalg.pinv(B) ? No.
        # We want to solve K * B' = A' => K = A' * pinv(B') -- this seems right.
        # K (3x3) = AT (3xN) @ pinv(BT) (Nx3)
        Binv = np.linalg.pinv(BT) # Pseudo-inverse of B transpose (Nx3)
        K = AT @ Binv # (3xN) @ (Nx3) -> 3x3

        # Normalize K so K[2, 2] = 1
        if abs(K[2, 2]) > 1e-8:
            K = K / K[2, 2]
        else:
            st.warning("DLT Intrinsics Warning: K[2, 2] is close to zero. Normalization might be unstable.")

        return K
    except np.linalg.LinAlgError as e:
        st.error(f"DLT Intrinsics Linear Algebra Error: {e}")
        return None
    except Exception as e:
        st.error(f"DLT Intrinsics Unexpected Error: {e}")
        return None

def dlt_calibrate_projection(pts_2d, pts_3d):
    """Estimates the 3x4 projection matrix P using DLT."""
    if pts_2d is None or pts_3d is None:
         st.error("DLT Projection Error: Input points are None.")
         return None
    if pts_2d.shape[1] != 2 or pts_3d.shape[1] != 3 or pts_2d.shape[0] != pts_3d.shape[0]:
        st.error(f"DLT Projection Error: Shape mismatch. pts_2d={pts_2d.shape}, pts_3d={pts_3d.shape}")
        return None
    if pts_2d.shape[0] < 6:
        st.error(f"DLT Projection Error: Need at least 6 points. Got {pts_2d.shape[0]}.")
        return None

    try:
        n = pts_2d.shape[0]
        # Convert 3D points to homogeneous coordinates (Nx4)
        X_h = np.hstack((pts_3d, np.ones((n, 1))))

        # Create the matrix M for the DLT equation (2n x 12)
        M = np.zeros((2 * n, 12))
        for i in range(n):
            Xi = X_h[i, :]  # Homogeneous 3D point (1x4)
            ui, vi = pts_2d[i, 0], pts_2d[i, 1] # 2D point coordinates

            # Fill rows 2i and 2i+1 of M
            M[2 * i, :]   = np.hstack((np.zeros(4), -Xi, vi * Xi))
            M[2 * i + 1, :] = np.hstack((Xi, np.zeros(4), -ui * Xi))

        # Solve M * P_vec = 0 using SVD
        U, S, Vh = svd(M)
        # The solution is the last row of Vh (or last column of V), reshaped to 3x4
        P = Vh[-1, :].reshape((3, 4))

        return P
    except np.linalg.LinAlgError as e:
         st.error(f"DLT Projection Linear Algebra Error: {e}")
         return None
    except Exception as e:
        st.error(f"DLT Projection Unexpected Error: {e}")
        return None

def dlt_P_to_KRt(P):
    """Decomposes the projection matrix P into K, R, t using RQ decomposition."""
    if P is None or P.shape != (3, 4):
        st.error("DLT KRt Error: Invalid projection matrix P.")
        return None, None, None
    try:
        M = P[0:3, 0:3] # Left 3x3 part of P
        K_rq, R_rq = rq(M) # RQ decomposition

        # Ensure K has positive diagonal elements
        # The sign of the diagonal elements of K_rq affects the sign of columns in R_rq
        T_sign = np.diag(np.sign(np.diag(K_rq)))
        K = K_rq @ T_sign # Adjust K
        R = T_sign @ R_rq # Adjust R accordingly

        # Normalize K so K[2, 2] = 1
        if abs(K[2, 2]) > 1e-8:
            K_norm = K / K[2, 2]
        else:
            st.warning("DLT KRt Warning: K[2, 2] near zero during normalization.")
            K_norm = K # Use unnormalized K

        # Check if R is a valid rotation matrix (det(R) should be +1)
        det_R = det(R)
        if abs(det_R - 1.0) > 1e-3:
            # If determinant is -1, it's a reflection. This can happen.
            # In some contexts, you might adjust P or points, but here we just warn.
            st.warning(f"DLT KRt Warning: det(R) = {det_R:.4f} (expected +1). May indicate reflection or issues.")
            # Optional: Force det(R) = +1 by multiplying the last column of P by -1 and redoing?
            # Or multiply R and t by -1? Check conventions. For now, just return as is.

        # Calculate translation t = -inv(K) * P[:, 3] ? No, it's inv(K) * P[:, 3]
        # P = K [R | t] => P[:, 0:3] = K*R (M) and P[:, 3] = K*t
        # So, t = inv(K) * P[:, 3]
        if abs(K[2, 2]) > 1e-8: # Use normalized K for t calculation if possible
            t = inv(K_norm) @ P[:, 3]
        else: # Use unnormalized K if normalization failed
             t = inv(K) @ P[:, 3]


        return K_norm, R, t.reshape(3, 1) # Return K (normalized), R, t (as column vector)

    except np.linalg.LinAlgError as e:
        st.error(f"DLT KRt Linear Algebra Error: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"DLT KRt Unexpected Error: {e}")
        return None, None, None

def dlt_compute_reprojection_error(P, pts_3d, pts_2d):
    """Computes RMS reprojection error given P, 3D points, and 2D points."""
    if P is None or pts_3d is None or pts_2d is None:
        st.error("DLT Reprojection Error: Input P, pts_3d, or pts_2d is None.")
        return np.inf, None
    if len(pts_3d) != len(pts_2d):
        st.error("DLT Reprojection Error: Mismatched number of 3D and 2D points.")
        return np.inf, None

    try:
        n = pts_3d.shape[0]
        # Convert 3D points to homogeneous coordinates (Nx4)
        X_h = np.hstack((pts_3d, np.ones((n, 1))))

        # Project 3D points to 2D using P: x_proj_h = P * X_h'
        x_proj_h = (P @ X_h.T).T # Result is Nx3 (homogeneous 2D coordinates)

        # Convert projected homogeneous coordinates to Euclidean 2D coordinates
        # Handle potential division by zero or near-zero w coordinate
        w = x_proj_h[:, 2]
        # Replace zero or small w with a small number to avoid errors, but keep sign
        w_safe = np.sign(w) * np.maximum(np.abs(w), 1e-8)
        x_proj = x_proj_h[:, :2] / w_safe[:, np.newaxis] # Nx2

        # Calculate Euclidean distance between detected and reprojected points
        errors = np.linalg.norm(pts_2d - x_proj, axis=1)

        # Calculate RMS error
        rms_error = np.sqrt(np.mean(errors**2))

        return rms_error, x_proj # Return RMS error and the projected points

    except Exception as e:
        st.error(f"DLT Reprojection Calculation Error: {e}")
        return np.inf, None

def dlt_visualize_calibration(img, pts_2d_original, pts_2d_reprojected):
    """Visualizes DLT calibration by plotting original and reprojected points on the image."""
    if img is None or pts_2d_original is None or pts_2d_reprojected is None:
        st.warning("DLT Visualization Warning: Missing image or point data.")
        return None
    if len(pts_2d_original) != len(pts_2d_reprojected):
         st.warning("DLT Visualization Warning: Mismatched number of original/reprojected points.")
         return None

    try:
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 8)) # Slightly larger figure
        ax.set_aspect('equal')

        # Display the image (convert BGR to RGB for Matplotlib)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Plot original points (red circles)
        ax.plot(pts_2d_original[:, 0], pts_2d_original[:, 1], 'ro', markersize=6,
                mfc='none', label='Original Detected Points') # mfc='none' for hollow circles

        # Plot reprojected points (green crosses)
        ax.plot(pts_2d_reprojected[:, 0], pts_2d_reprojected[:, 1], 'gx', markersize=8, # Larger crosses
                label='Reprojected Points')

        # Add lines connecting original to reprojected points to show error vectors
        for i in range(len(pts_2d_original)):
             ax.plot([pts_2d_original[i, 0], pts_2d_reprojected[i, 0]],
                     [pts_2d_original[i, 1], pts_2d_reprojected[i, 1]], 'y-', linewidth=0.5, alpha=0.7)


        # Customize plot
        ax.legend()
        ax.set_title("DLT Calibration: Original vs. Reprojected Points")
        h, w = img.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0) # Invert y-axis to match image coordinates
        ax.set_xlabel("X Pixel Coordinate")
        ax.set_ylabel("Y Pixel Coordinate")
        ax.axis('on') # Ensure axes are visible
        plt.tight_layout()

        return fig
    except Exception as e:
        st.error(f"DLT Visualization Error: {e}")
        return None


# ==============================================================================
# Part 4: Experiment Function (Data Efficiency - Images Only) - FIXED
# ==============================================================================
def run_data_efficiency_images_experiment(calibrate_func, image_files, pattern_size_cv, square_size, num_images_list, is_fisheye, num_runs=3):
    """
    Runs calibration (standard or fisheye) with subsets of images and evaluates
    the reprojection error on the *full* set of valid images.

    FIXED: Uses correct pose estimation for fisheye evaluation.
    """
    method_name = "Fisheye" if is_fisheye else "Standard"
    results = {'num_images': [], 'avg_rms': [], 'std_rms': []} # Store avg/std RMS

    _placeholder = st.empty()
    _placeholder.info(f"Processing all images for {method_name} baseline evaluation...")

    # Use existing processed data if available, otherwise process
    # This prevents reprocessing if only the experiment parameters change
    if ('objpoints_processed' in st.session_state and st.session_state.objpoints_processed and
        'imgpoints_processed' in st.session_state and st.session_state.imgpoints_processed and
        'img_shape_processed' in st.session_state and st.session_state.img_shape_processed):
         objpoints_full = st.session_state.objpoints_processed
         imgpoints_full = st.session_state.imgpoints_processed
         img_shape = st.session_state.img_shape_processed
         # valid_indices = st.session_state.valid_indices_processed # Not strictly needed here
         _placeholder.success(f"Using {len(objpoints_full)} pre-processed valid views.")
         time.sleep(1) # Show message briefly
    else:
        objpoints_full, imgpoints_full, img_shape, valid_indices = process_images(image_files, pattern_size_cv, square_size)
        # Store processed data in session state if successful
        if objpoints_full:
            st.session_state.objpoints_processed = objpoints_full
            st.session_state.imgpoints_processed = imgpoints_full
            st.session_state.img_shape_processed = img_shape
            st.session_state.valid_indices_processed = valid_indices

    _placeholder.empty() # Clear the status message

    # --- Pre-computation Checks ---
    if objpoints_full is None or not imgpoints_full or img_shape is None:
        st.error(f"Cannot run {method_name} experiment: Initial image processing failed or yielded no corners.")
        return pd.DataFrame(results) # Return empty dataframe

    num_valid_views = len(imgpoints_full)
    if num_valid_views < 2:
         st.error(f"Cannot run {method_name} experiment: Need at least 2 valid images, found {num_valid_views}.")
         return pd.DataFrame(results)

    st.write(f"Starting {method_name} efficiency experiment...")
    st.write(f"- Evaluating calibration based on subsets of images.")
    st.write(f"- Reprojection error calculated using **all {num_valid_views} valid views** for each subset calibration.")
    st.write(f"- Averaging results over {num_runs} random runs per subset size.")

    prog_bar = st.progress(0)
    status_text = st.empty()

    # Determine the list of image counts to test
    if not num_images_list: # If empty, generate default range
        num_images_list = list(range(2, num_valid_views + 1))
    # Filter counts to be within the valid range [2, num_valid_views]
    valid_counts_to_test = sorted([n for n in num_images_list if 2 <= n <= num_valid_views])

    if not valid_counts_to_test:
        st.warning(f"No valid image counts between 2 and {num_valid_views} were specified for the {method_name} experiment.")
        prog_bar.empty()
        return pd.DataFrame(results)

    run_errors_raw = {} # Optional: Store all raw RMS values for potential deeper analysis

    # --- Main Experiment Loop ---
    for i, num_images_subset in enumerate(valid_counts_to_test):
        status_text.text(f"Testing with {num_images_subset} images ({i+1}/{len(valid_counts_to_test)})...")
        errors_for_this_subset_size = []
        successful_calibrations_count = 0
        successful_rms_calculations_count = 0

        for run in range(num_runs):
            try:
                # 1. Sample a random subset of images
                indices_subset = random.sample(range(num_valid_views), num_images_subset)
                obj_subset = [objpoints_full[j] for j in indices_subset]
                img_subset = [imgpoints_full[j] for j in indices_subset]

                # 2. Run calibration on the SUBSET
                ret_calib, mtx_sub, dist_sub, _, _ = calibrate_func(obj_subset, img_subset, img_shape)
                # Note: We don't need rvecs/tvecs from the subset calibration itself for evaluation

                if ret_calib and mtx_sub is not None and dist_sub is not None:
                    successful_calibrations_count += 1
                    current_run_rms = np.inf # Initialize RMS for this run

                    # 3. Evaluate using the FULL dataset
                    if is_fisheye:
                        # --- Fisheye Evaluation ---
                        # Need to estimate poses for ALL views using the subset's K, D
                        rvecs_full_est = []
                        tvecs_full_est = []
                        poses_estimated_count = 0
                        view_estimation_failed = False

                        for view_idx in range(num_valid_views):
                            objp_view = objpoints_full[view_idx].astype(np.float32) # Ensure type
                            imgp_view = imgpoints_full[view_idx].astype(np.float32) # Ensure type

                            try:
                                # Undistort points using the subset's K and D
                                # P=mtx_sub maps undistorted points back to pixel scale using the subset's K
                                imgp_undistorted = cv2.fisheye.undistortPoints(imgp_view, mtx_sub, dist_sub, None, mtx_sub)

                                # Solve PnP using undistorted points and K_sub (distortion is None now)
                                # Ensure objp has correct shape (N, 1, 3) or (N, 3) - solvePnP handles both
                                if len(objp_view) > 3:
                                     ret_pnp, rvec, tvec = cv2.solvePnP(objp_view, imgp_undistorted, mtx_sub, None) # No distortion needed
                                     if ret_pnp:
                                         rvecs_full_est.append(rvec)
                                         tvecs_full_est.append(tvec)
                                         poses_estimated_count += 1
                                     else:
                                         rvecs_full_est.append(None) # Placeholder
                                         tvecs_full_est.append(None)
                                         view_estimation_failed = True
                                else:
                                     rvecs_full_est.append(None); tvecs_full_est.append(None); view_estimation_failed = True


                            except cv2.error as pose_cv_e:
                                # Specific OpenCV error during pose estimation
                                # print(f" [Run {run+1}, {num_images_subset} imgs] CV2 Pose Est Error (View {view_idx}): {pose_cv_e}")
                                rvecs_full_est.append(None); tvecs_full_est.append(None); view_estimation_failed = True
                            except Exception as pose_e:
                                # Other unexpected error
                                # print(f" [Run {run+1}, {num_images_subset} imgs] Pose Est Error (View {view_idx}): {pose_e}")
                                rvecs_full_est.append(None); tvecs_full_est.append(None); view_estimation_failed = True

                        # Calculate RMS only if ALL poses were estimated successfully
                        if not view_estimation_failed and poses_estimated_count == num_valid_views:
                             current_run_rms = calculate_reprojection_error(
                                 objpoints_full, imgpoints_full,
                                 mtx_sub, dist_sub, # Use subset's K, D
                                 rvecs_full_est, tvecs_full_est, # Use estimated poses for ALL views
                                 is_fisheye=True
                             )
                        # else:
                             # Optional: Log why RMS was skipped
                             # print(f" [Run {run+1}, {num_images_subset} imgs] Skipping RMS calc: Failed to estimate pose for all views.")


                    else:
                        # --- Standard Evaluation ---
                        # calculate_reprojection_error handles internal solvePnP for standard case
                        # Pass None for rvecs/tvecs_calib to trigger internal estimation
                         current_run_rms = calculate_reprojection_error(
                             objpoints_full, imgpoints_full,
                             mtx_sub, dist_sub, # Use subset's K, D
                             None, None, # Let function estimate poses internally
                             is_fisheye=False
                         )

                    # Store the RMS if it's valid
                    if np.isfinite(current_run_rms):
                        errors_for_this_subset_size.append(current_run_rms)
                        successful_rms_calculations_count += 1
                # else: # Calibration itself failed for this run/subset
                    # print(f" [Run {run+1}, {num_images_subset} imgs] Calibration function failed.")
                    # pass # No RMS can be calculated

            except Exception as calib_e:
                # Catch errors during the calibration function call itself
                st.warning(f"Error during calibration run (Images: {num_images_subset}, Run: {run+1}): {calib_e}")
                continue # Skip to next run

        # --- After all runs for a given subset size ---
        run_errors_raw[num_images_subset] = errors_for_this_subset_size # Store raw errors

        # Calculate and store average/std dev if we got valid RMS values
        if errors_for_this_subset_size:
            results['num_images'].append(num_images_subset)
            results['avg_rms'].append(np.mean(errors_for_this_subset_size))
            results['std_rms'].append(np.std(errors_for_this_subset_size))
        elif successful_calibrations_count > 0:
            # Warn if calibration worked sometimes but RMS calculation always failed
             if successful_rms_calculations_count == 0:
                  st.warning(f"Calibration succeeded for {num_images_subset} images, but RMS calculation failed across all {num_runs} runs (check pose estimation steps).")
        # else: # Calibration failed for all runs
             # st.warning(f"Calibration failed for all {num_runs} runs using {num_images_subset} images.")


        # Update progress bar
        prog_bar.progress((i + 1) / len(valid_counts_to_test))

    # --- Experiment End ---
    prog_bar.empty()
    status_text.empty()
    df_results = pd.DataFrame(results)

    if df_results.empty:
        st.warning(f"The {method_name} data efficiency experiment completed but yielded no valid numerical results.")
        # Optional: Display raw errors for debugging
        # st.write("Raw errors per run (for debugging):", run_errors_raw)

    return df_results


# ==============================================================================
# Part 5: Streamlit Application UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Camera Calibration Tool")

# --- Initialize Session State ---
# Use a helper function to avoid repetition
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Calibration results
init_state('std_calib_results', None)
init_state('fisheye_calib_results', None)
init_state('dlt_calib_results', None)
# Input data / parameters
init_state('calib_images', []) # Store uploaded file objects
init_state('pattern_params', {'rows': DEFAULT_PATTERN_SIZE[1], 'cols': DEFAULT_PATTERN_SIZE[0], 'size': DEFAULT_SQUARE_SIZE})
# Processed data (from checkerboard images)
init_state('objpoints_processed', None)
init_state('imgpoints_processed', None)
init_state('img_shape_processed', None)
init_state('valid_indices_processed', None)
# Experiment results
init_state('eff_img_results_std', None)
init_state('eff_img_results_fish', None)
# DLT file storage
init_state('dlt_mat_file_3d_pts', None) # Renamed for clarity
init_state('dlt_mat_file_2d_pts', None) # Renamed for clarity
init_state('dlt_mat_file_corres', None)
init_state('dlt_image_file', None)


# ========================
# --- Sidebar Setup ---
# ========================
st.sidebar.title("📷 Calibration Methods")
app_mode = st.sidebar.radio(
    "Choose Calibration Type",
    ["Standard Calibration", "Fisheye Calibration", "DLT Calibration (MAT Files)"],
    key="app_mode_radio",
    help="Select the camera model or method to use."
)
st.sidebar.divider()

# --- Input Sections within Sidebar ---

# Checkerboard Inputs (for Standard & Fisheye)
if app_mode in ["Standard Calibration", "Fisheye Calibration"]:
    st.sidebar.header("Checkerboard Input")
    st.sidebar.markdown("Define the checkerboard pattern dimensions and square size.")

    # Use pattern_params from session state for sticky values
    current_params = st.session_state.pattern_params
    rows_input = st.sidebar.number_input("Inner Corners Height (Rows)", min_value=2, value=current_params['rows'], key="pattern_rows", help="Number of inner corners vertically.")
    cols_input = st.sidebar.number_input("Inner Corners Width (Cols)", min_value=2, value=current_params['cols'], key="pattern_cols", help="Number of inner corners horizontally.")
    size_input = st.sidebar.number_input("Square Size (mm)", min_value=0.01, value=current_params['size'], step=0.1, format="%.2f", key="pattern_size", help="The actual side length of a square on the checkerboard.")

    # Check if parameters changed
    if rows_input != current_params['rows'] or cols_input != current_params['cols'] or size_input != current_params['size']:
        st.session_state.pattern_params = {'rows': rows_input, 'cols': cols_input, 'size': size_input}
        # Clear potentially stale processed data if pattern changes
        st.session_state.objpoints_processed = None
        st.session_state.imgpoints_processed = None
        st.session_state.std_calib_results = None
        st.session_state.fisheye_calib_results = None
        st.session_state.eff_img_results_std = None
        st.session_state.eff_img_results_fish = None
        rerun_if_possible() # Rerun to reflect changes immediately

    pattern_size_cv = (cols_input, rows_input) # OpenCV uses (cols, rows)

    # File Uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Checkerboard Images",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg'],
        key="calib_uploads",
        help="Upload multiple images of the checkerboard from different angles/positions."
    )

    # Check if new files were uploaded
    # Comparing file objects directly can be tricky, rely on button press or check list identity/length?
    # A simple check: if the current list is different from the state list
    if uploaded_files is not None and uploaded_files != st.session_state.get('calib_images'):
         st.session_state.calib_images = uploaded_files
         # Clear downstream results if images change
         st.session_state.objpoints_processed = None
         st.session_state.imgpoints_processed = None
         st.session_state.img_shape_processed = None
         st.session_state.valid_indices_processed = None
         st.session_state.std_calib_results = None
         st.session_state.fisheye_calib_results = None
         st.session_state.eff_img_results_std = None
         st.session_state.eff_img_results_fish = None
         rerun_if_possible() # Rerun to update file count display

    # Display info and processing button
    st.sidebar.info(f"Pattern: {pattern_size_cv[0]}x{pattern_size_cv[1]}, Size: {size_input:.2f} mm")
    if st.session_state.calib_images:
        st.sidebar.write(f"{len(st.session_state.calib_images)} image(s) ready.")
    else:
        st.sidebar.warning("Upload checkerboard images to proceed.")

    if st.sidebar.button("Process Checkerboard Images", key="process_images_button",
                         disabled=(not st.session_state.calib_images),
                         help="Detects corners in the uploaded images based on the defined pattern."):
        with st.spinner("Processing images... This may take a moment."):
            # Clear previous processed data and results before reprocessing
            st.session_state.objpoints_processed = None
            st.session_state.imgpoints_processed = None
            st.session_state.img_shape_processed = None
            st.session_state.valid_indices_processed = None
            st.session_state.std_calib_results = None
            st.session_state.fisheye_calib_results = None
            st.session_state.eff_img_results_std = None
            st.session_state.eff_img_results_fish = None

            objp_all, imgp_all, shape, valid_indices = process_images(
                st.session_state.calib_images,
                pattern_size_cv,
                size_input
            )
            # Store results in session state
            st.session_state.objpoints_processed = objp_all
            st.session_state.imgpoints_processed = imgp_all
            st.session_state.img_shape_processed = shape
            st.session_state.valid_indices_processed = valid_indices
        # No need to rerun here, state update triggers UI refresh


# DLT Inputs (for DLT Calibration)
elif app_mode == "DLT Calibration (MAT Files)":
    st.sidebar.header("DLT Input Files")
    st.sidebar.markdown("Upload specific `.mat` files and the corresponding image.")
    st.sidebar.info("""
    Requires:
    - `pt_corres.mat` (containing `pts_2D`, `cam_pts_3D`)
    - `rubik_3D_pts.mat` (containing `pts_3d`)
    - `rubik_2D_pts.mat` (containing `pts_2d`)
    - `rubik_cube.jpg` (or similar image)
    """)

    # File uploaders in sidebar with unique keys and descriptive labels
    mat_corres_f = st.sidebar.file_uploader("1. Correspondences (`pt_corres.mat`)", type=['mat'], key='dlt_mat_corres_up')
    mat_3d_rubik_f = st.sidebar.file_uploader("2. 3D Points (`rubik_3D_pts.mat`)", type=['mat'], key='dlt_mat3d_rubik_up')
    mat_2d_rubik_f = st.sidebar.file_uploader("3. 2D Points (`rubik_2D_pts.mat`)", type=['mat'], key='dlt_mat2d_rubik_up')
    dlt_img_f = st.sidebar.file_uploader("4. Image (`rubik_cube.jpg`)", type=['png', 'jpg', 'jpeg'], key='dlt_img_up')

    # Update session state if files are uploaded (simple replacement)
    # We'll check for actual content later when the button is pressed
    st.session_state.dlt_mat_file_corres = mat_corres_f
    st.session_state.dlt_mat_file_3d_pts = mat_3d_rubik_f # Updated key name
    st.session_state.dlt_mat_file_2d_pts = mat_2d_rubik_f # Updated key name
    st.session_state.dlt_image_file = dlt_img_f


# Dataset Links (Common Sidebar Section)
st.sidebar.divider()
st.sidebar.markdown("### Datasets & Info")
st.sidebar.markdown("""
*   **Checkerboard Samples:**
    *   [OpenCV `left*.jpg`](https://github.com/opencv/opencv/tree/4.x/samples/data)
    *   [Caltech Dataset](http://www.vision.caltech.edu/bouguetj/calib_doc/)
*   **DLT Files:** Typically provided with specific assignments or research datasets. Ensure the `.mat` files contain the exact variable names expected by the code (`pts_2D`, `cam_pts_3D`, `pts_3d`, `pts_2d`).
*   **Fisheye Tips:** Calibration benefits from images covering the full field of view, especially the edges and corners, with varying checkerboard orientations.
""")

# ========================
# --- Main App Area ---
# ========================

st.title("Camera Calibration Tool")

# --- Standard Calibration Mode ---
if app_mode == "Standard Calibration":
    st.header("Standard Camera Calibration (Pinhole Model)")
    st.markdown("Uses `cv2.calibrateCamera` which implements Zhang's method. Assumes a standard lens (not fisheye).")
    st.markdown("---")

    # Check if checkerboard data is processed
    if st.session_state.get('objpoints_processed') is None or st.session_state.get('imgpoints_processed') is None:
        st.info("⬅️ Please upload and process checkerboard images using the sidebar first.")
    else:
        st.success(f"✅ Checkerboard data processed for {len(st.session_state['imgpoints_processed'])} valid images.")

        # Calibration Button
        if st.button("🚀 Run Standard Calibration", key="std_calib_button", help="Perform calibration using the processed checkerboard points."):
            with st.spinner("Running Standard Calibration..."):
                objp_all = st.session_state.objpoints_processed
                imgp_all = st.session_state.imgpoints_processed
                shape = st.session_state.img_shape_processed

                if not objp_all or not imgp_all or shape is None:
                     st.error("Cannot calibrate. Processed point data or image shape is missing.")
                else:
                    # Clear previous results before running
                    st.session_state.std_calib_results = None
                    st.session_state.fisheye_calib_results = None # Ensure only one result type active

                    ret, mtx, dist, rvecs, tvecs = run_standard_calibration(objp_all, imgp_all, shape)

                    if ret:
                         # Calculate RMS error on the same data used for calibration
                         rms = calculate_reprojection_error(objp_all, imgp_all, mtx, dist, rvecs, tvecs, is_fisheye=False)
                         # Store results
                         st.session_state.std_calib_results = {
                             "K": mtx,
                             "dist": dist,
                             "rms_error": rms,
                             "img_shape": shape,
                             "rvecs": rvecs, # Store for potential later use
                             "tvecs": tvecs
                         }
                         st.success(f"Standard Calibration Successful! RMS Error: {rms:.4f} px")
                    else:
                         st.error("Standard Calibration Failed. Check console output for details (if running locally) or ensure sufficient/varied views.")
                         st.session_state.std_calib_results = None # Ensure results are cleared on failure

        # Display Standard Calibration Results
        if st.session_state.get('std_calib_results'):
            st.divider()
            st.subheader("✅ Standard Calibration Results")
            res = st.session_state.std_calib_results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMS Reprojection Error", f"{res['rms_error']:.4f} px")
                st.write("**Intrinsic Matrix (K):**")
                st.code(f"{res['K']}")
            with col2:
                st.write(f"**Image Size (WxH):** {res['img_shape'][0]}x{res['img_shape'][1]}")
                st.write("**Distortion Coefficients (k1, k2, p1, p2, k3):**")
                st.code(f"{res['dist'].flatten()}")


            # --- Data Efficiency Experiment Section ---
            st.divider()
            st.subheader("📊 Data Efficiency Experiment")
            st.markdown("Analyze how the number of calibration images affects the RMS reprojection error (evaluated on the full dataset).")
            num_views_available = len(st.session_state.get('imgpoints_processed', []))

            if num_views_available < 2:
                st.info("Need at least 2 valid views to run the data efficiency experiment.")
            else:
                # Define steps dynamically up to the number of available views
                default_steps = list(range(2, num_views_available + 1))
                # Optionally allow user input for steps? For now, use all steps.
                st.write(f"Testing with image counts: `{default_steps}`")
                num_runs_per_step = st.slider("Number of Random Runs per Step", min_value=1, max_value=10, value=3, key="eff_runs_std",
                                               help="How many times to randomly select images for each count (e.g., 3 runs using 5 images).")

                if st.button("Run Standard Efficiency Test", key="eff_run_std"):
                     with st.spinner("Running Efficiency Test... This may take some time."):
                         # Clear previous experiment results
                         st.session_state.eff_img_results_std = None
                         df_eff = run_data_efficiency_images_experiment(
                             calibrate_func=run_standard_calibration, # Pass the specific calibration function
                             image_files=st.session_state.calib_images, # Pass original files
                             pattern_size_cv=pattern_size_cv,
                             square_size=size_input,
                             num_images_list=default_steps, # Pass the steps
                             is_fisheye=False,
                             num_runs=num_runs_per_step
                         )
                         st.session_state.eff_img_results_std = df_eff # Store results in session state

                # Display Experiment Results if available
                if st.session_state.get('eff_img_results_std') is not None:
                     st.markdown("**Experiment Results:**")
                     df_display = st.session_state.eff_img_results_std
                     if not df_display.empty:
                         # Show table with formatted RMS
                         df_table = df_display[['num_images', 'avg_rms', 'std_rms']].rename(
                             columns={'num_images': '# Images', 'avg_rms': 'Mean RMS', 'std_rms': 'Std Dev RMS'}
                         )
                         df_table['Mean RMS'] = df_table['Mean RMS'].map('{:.4f}'.format)
                         df_table['Std Dev RMS'] = df_table['Std Dev RMS'].map('{:.4f}'.format)
                         st.dataframe(df_table, hide_index=True, use_container_width=True)

                         # Plotting
                         try:
                             fig_eff, ax_eff = plt.subplots(figsize=(8, 4))
                             ax_eff.plot(df_display['num_images'], df_display['avg_rms'], marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=6, label='Mean RMS Error')
                             # Add shaded region for standard deviation
                             if 'std_rms' in df_display.columns:
                                 ax_eff.fill_between(df_display['num_images'],
                                                     df_display['avg_rms'] - df_display['std_rms'],
                                                     df_display['avg_rms'] + df_display['std_rms'],
                                                     color='dodgerblue', alpha=0.2, label='±1 Std Dev')

                             # Plot customization
                             max_error_plot = df_display['avg_rms'].max()
                             min_error_plot = df_display['avg_rms'].min()
                             y_top_limit = max(0.3, max_error_plot * 1.1) if np.isfinite(max_error_plot) else 0.5 # Sensible upper limit
                             ax_eff.set_ylim(bottom=0, top=y_top_limit) # Start y-axis at 0

                             # Set x-axis ticks to match the tested image counts
                             tested_ticks = sorted(df_display['num_images'].unique())
                             ax_eff.set_xticks(tested_ticks) # Show all tested counts
                             # Optional: Reduce tick density if too many points
                             # if len(tested_ticks) > 15:
                             #      ax_eff.set_xticks(tested_ticks[::max(1, len(tested_ticks)//10)])

                             ax_eff.set_xlabel("Number of Images Used for Calibration")
                             ax_eff.set_ylabel("Mean RMS Error (px) on Full Dataset")
                             ax_eff.set_title("Standard Calibration: Data Efficiency")
                             ax_eff.grid(axis='y', linestyle='--', alpha=0.6)
                             ax_eff.legend()
                             plt.tight_layout()
                             st.pyplot(fig_eff)
                         except Exception as plot_e:
                             st.warning(f"Could not generate efficiency plot: {plot_e}")
                     else:
                         # This message comes from the experiment function if it yields no results
                         # st.warning("Efficiency experiment did not generate any results.") # Redundant?
                         pass


# --- Fisheye Calibration Mode ---
elif app_mode == "Fisheye Calibration":
    st.header("🐠 Fisheye Camera Calibration")
    st.markdown("Uses `cv2.fisheye.calibrate`. Suitable for wide-angle or fisheye lenses.")
    st.markdown("---")

    # Check if checkerboard data is processed
    if st.session_state.get('objpoints_processed') is None or st.session_state.get('imgpoints_processed') is None:
        st.info("⬅️ Please upload and process checkerboard images using the sidebar first.")
    else:
        st.success(f"✅ Checkerboard data processed for {len(st.session_state['imgpoints_processed'])} valid images.")

        # Calibration Button
        if st.button("🚀 Run Fisheye Calibration", key="fisheye_calib_button", help="Perform fisheye calibration."):
            with st.spinner("Running Fisheye Calibration..."):
                objp_all = st.session_state.objpoints_processed
                imgp_all = st.session_state.imgpoints_processed
                shape = st.session_state.img_shape_processed

                if not objp_all or not imgp_all or shape is None:
                    st.error("Cannot calibrate. Processed point data or image shape is missing.")
                else:
                    # Clear previous results
                    st.session_state.std_calib_results = None # Ensure only one result type active
                    st.session_state.fisheye_calib_results = None

                    ret, K, D, rvecs, tvecs = run_fisheye_calibration(objp_all, imgp_all, shape)

                    if ret:
                        # Calculate RMS error using the fisheye model
                        rms = calculate_reprojection_error(objp_all, imgp_all, K, D, rvecs, tvecs, is_fisheye=True)
                        # Store results
                        st.session_state.fisheye_calib_results = {
                            "K": K,
                            "dist": D, # Fisheye distortion coefficients (k1, k2, k3, k4)
                            "rms_error": rms,
                            "img_shape": shape,
                            "rvecs": rvecs,
                            "tvecs": tvecs
                        }
                        st.success(f"Fisheye Calibration Successful! RMS Error: {rms:.4f} px")
                    else:
                        st.error("Fisheye Calibration Failed. Check console for details or ensure views have enough variety, especially near edges.")
                        st.session_state.fisheye_calib_results = None # Clear results on failure

        # Display Fisheye Calibration Results
        if st.session_state.get('fisheye_calib_results'):
            st.divider()
            st.subheader("✅ Fisheye Calibration Results")
            res = st.session_state.fisheye_calib_results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMS Reprojection Error", f"{res['rms_error']:.4f} px")
                st.write("**Intrinsic Matrix (K):**")
                st.code(f"{res['K']}")
            with col2:
                 st.write(f"**Image Size (WxH):** {res['img_shape'][0]}x{res['img_shape'][1]}")
                 st.write("**Fisheye Distortion (k1, k2, k3, k4):**")
                 st.code(f"{res['dist'].flatten()}")

            # --- Fisheye Data Efficiency Experiment Section ---
            st.divider()
            st.subheader("📊 Data Efficiency Experiment (Fisheye)")
            st.markdown("Analyze how the number of calibration images affects the RMS reprojection error (evaluated on the full dataset using the fisheye model).")
            num_views_available = len(st.session_state.get('imgpoints_processed', []))

            if num_views_available < 2:
                st.info("Need at least 2 valid views to run the data efficiency experiment.")
            else:
                default_steps = list(range(2, num_views_available + 1))
                st.write(f"Testing with image counts: `{default_steps}`")
                num_runs_per_step = st.slider("Number of Random Runs per Step", min_value=1, max_value=10, value=3, key="eff_runs_fish",
                                               help="How many times to randomly select images for each count.")

                if st.button("Run Fisheye Efficiency Test", key="eff_run_fish"):
                     with st.spinner("Running Fisheye Efficiency Test... This may take some time."):
                         # Clear previous experiment results
                         st.session_state.eff_img_results_fish = None
                         df_eff_fish = run_data_efficiency_images_experiment(
                             calibrate_func=run_fisheye_calibration, # Pass the fisheye calibration function
                             image_files=st.session_state.calib_images,
                             pattern_size_cv=pattern_size_cv,
                             square_size=size_input,
                             num_images_list=default_steps,
                             is_fisheye=True, # Specify fisheye model
                             num_runs=num_runs_per_step
                         )
                         st.session_state.eff_img_results_fish = df_eff_fish

                # Display Fisheye Experiment Results if available
                if st.session_state.get('eff_img_results_fish') is not None:
                     st.markdown("**Experiment Results:**")
                     df_display_fish = st.session_state.eff_img_results_fish
                     if not df_display_fish.empty:
                         # Show table
                         df_table_fish = df_display_fish[['num_images', 'avg_rms', 'std_rms']].rename(
                             columns={'num_images': '# Images', 'avg_rms': 'Mean RMS', 'std_rms': 'Std Dev RMS'}
                         )
                         df_table_fish['Mean RMS'] = df_table_fish['Mean RMS'].map('{:.4f}'.format)
                         df_table_fish['Std Dev RMS'] = df_table_fish['Std Dev RMS'].map('{:.4f}'.format)
                         st.dataframe(df_table_fish, hide_index=True, use_container_width=True)

                         # Plotting
                         try:
                             fig_eff_fish, ax_eff_fish = plt.subplots(figsize=(8, 4))
                             ax_eff_fish.plot(df_display_fish['num_images'], df_display_fish['avg_rms'], marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=6, label='Mean RMS Error')
                             if 'std_rms' in df_display_fish.columns:
                                 ax_eff_fish.fill_between(df_display_fish['num_images'],
                                                     df_display_fish['avg_rms'] - df_display_fish['std_rms'],
                                                     df_display_fish['avg_rms'] + df_display_fish['std_rms'],
                                                     color='mediumseagreen', alpha=0.2, label='±1 Std Dev')

                             # Plot customization
                             max_err_plot_f = df_display_fish['avg_rms'].max()
                             min_err_plot_f = df_display_fish['avg_rms'].min()
                             y_top_limit_f = max(max_err_plot_f * 1.1, min_err_plot_f + 0.1) if np.isfinite(max_err_plot_f) else 1.0
                             ax_eff_fish.set_ylim(bottom=0, top=y_top_limit_f)

                             tested_ticks_f = sorted(df_display_fish['num_images'].unique())
                             ax_eff_fish.set_xticks(tested_ticks_f)

                             ax_eff_fish.set_xlabel("Number of Images Used for Calibration")
                             ax_eff_fish.set_ylabel("Mean RMS Error (px) on Full Dataset")
                             ax_eff_fish.set_title("Fisheye Calibration: Data Efficiency")
                             ax_eff_fish.grid(axis='y', linestyle='--', alpha=0.6)
                             ax_eff_fish.legend()
                             plt.tight_layout()
                             st.pyplot(fig_eff_fish)
                         except Exception as plot_e:
                             st.warning(f"Could not generate fisheye efficiency plot: {plot_e}")
                     # else: # Message handled by experiment function

# --- DLT Calibration Mode ---
elif app_mode == "DLT Calibration (MAT Files)":
    st.header("📐 Direct Linear Transformation (DLT) Calibration")
    st.markdown("""
    Uses specific `.mat` files containing 3D-2D point correspondences. Assumes a pinhole camera model but calculates parameters differently than `cv2.calibrateCamera`.
    - **Part 1:** Estimates intrinsics (K) assuming known extrinsics (i.e., 3D points provided in the *camera's* coordinate system). Uses `pt_corres.mat`.
    - **Part 2:** Estimates the full 3x4 projection matrix (P) from *world* 3D points to 2D image points, then decomposes P into K, R, t. Uses `rubik_*.mat` files and the image.
    """)
    st.warning("Ensure correct files with expected variable names (`cam_pts_3D`, `pts_2D`, `pts_3d`, `pts_2d`) are uploaded via the sidebar. The code will attempt to handle `NxD` or `DxN` formats.")
    st.markdown("---")

    # Display status of uploaded DLT files
    st.subheader("Uploaded DLT Files Status:")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    files_present = {}
    # (File status display code remains the same as before)
    with col_f1:
        f = st.session_state.get('dlt_mat_file_corres')
        st.markdown("**1. Correspondences File:**")
        if f: st.success(f"✓ `{f.name}`"); files_present['corres'] = True
        else: st.error("Missing"); files_present['corres'] = False
    with col_f2:
        f = st.session_state.get('dlt_mat_file_3d_pts')
        st.markdown("**2. 3D World Points File:**")
        if f: st.success(f"✓ `{f.name}`"); files_present['3d_pts'] = True
        else: st.error("Missing"); files_present['3d_pts'] = False
    with col_f3:
        f = st.session_state.get('dlt_mat_file_2d_pts')
        st.markdown("**3. 2D Image Points File:**")
        if f: st.success(f"✓ `{f.name}`"); files_present['2d_pts'] = True
        else: st.error("Missing"); files_present['2d_pts'] = False
    with col_f4:
        f = st.session_state.get('dlt_image_file')
        st.markdown("**4. Image File:**")
        if f: st.success(f"✓ `{f.name}`"); files_present['image'] = True
        else: st.error("Missing"); files_present['image'] = False

    all_files_ready = all(files_present.values())

    # DLT Calibration Button
    if st.button("🚀 Run DLT Calibration", key="dlt_calib_button", disabled=not all_files_ready,
                 help="Run DLT calibration using the uploaded files."):
        with st.spinner("Running DLT Calibration..."):
            # Clear previous results
            st.session_state.dlt_calib_results = None
            success = True  # Flag to track if we should continue processing

            try:
                # --- Part 1: Calibration with Known Extrinsics ---
                # Load correspondences data
                corres_data = load_mat_file(st.session_state.dlt_mat_file_corres)
                if corres_data is None:
                    st.error("Failed to load correspondences file.")
                    success = False
                
                if success:
                    # Extract points, ensuring correct shape (Nx2 for 2D, Nx3 for 3D)
                    pts_2D = corres_data.get('pts_2D')
                    cam_pts_3D = corres_data.get('cam_pts_3D')

                    if pts_2D is None or cam_pts_3D is None:
                        st.error("Could not find pts_2D or cam_pts_3D in correspondences file.")
                        success = False

                if success:
                    # Ensure points are in correct shape (Nx2 and Nx3)
                    if pts_2D.shape[0] == 2 and pts_2D.shape[1] > 2:  # If 2xN
                        pts_2D = pts_2D.T
                    if cam_pts_3D.shape[0] == 3 and cam_pts_3D.shape[1] > 3:  # If 3xN
                        cam_pts_3D = cam_pts_3D.T

                    # Run Part 1 calibration
                    K_part1 = dlt_calibrate_intrinsics(pts_2D, cam_pts_3D)

                    # --- Part 2: Full Calibration from World Points ---
                    # Load world points data
                    world_3d_data = load_mat_file(st.session_state.dlt_mat_file_3d_pts)
                    world_2d_data = load_mat_file(st.session_state.dlt_mat_file_2d_pts)
                    
                    if world_3d_data is None or world_2d_data is None:
                        st.error("Failed to load world points files.")
                        success = False

                if success:
                    # Extract points
                    pts_3d = world_3d_data.get('pts_3d')
                    pts_2d = world_2d_data.get('pts_2d')

                    if pts_3d is None or pts_2d is None:
                        st.error("Could not find pts_3d or pts_2d in world points files.")
                        success = False

                if success:
                    # Ensure points are in correct shape (Nx3 and Nx2)
                    if pts_3d.shape[0] == 3 and pts_3d.shape[1] > 3:  # If 3xN
                        pts_3d = pts_3d.T
                    if pts_2d.shape[0] == 2 and pts_2d.shape[1] > 2:  # If 2xN
                        pts_2d = pts_2d.T

                    # Run Part 2 calibration
                    P_part2 = dlt_calibrate_projection(pts_2d, pts_3d)
                    K_part2, R_part2, t_part2 = dlt_P_to_KRt(P_part2)

                    # Calculate reprojection error
                    rms_error_part2, proj_pts = dlt_compute_reprojection_error(P_part2, pts_3d, pts_2d)

                    # Load and process image for visualization
                    img = decode_image(st.session_state.dlt_image_file)
                    if img is not None:
                        # Create visualization
                        fig = dlt_visualize_calibration(img, pts_2d, proj_pts)
                    else:
                        fig = None
                        st.warning("Could not load image for visualization.")

                    # Store all results in session state
                    dlt_results = {
                        "K_part1": K_part1,
                        "P_part2": P_part2,
                        "K_part2": K_part2,
                        "R_part2": R_part2,
                        "t_part2": t_part2,
                        "rms_error_part2": rms_error_part2,
                        "visualization_fig_part2": fig,
                        "img_part2": img
                    }
                    st.session_state.dlt_calib_results = dlt_results

            except Exception as e:
                st.error(f"DLT Calibration failed: {str(e)}")
                success = False

    # --- Display DLT Results (check state *after* potential button press run) ---
    if st.session_state.get('dlt_calib_results'):
        st.divider()
        st.subheader("✅ DLT Calibration Results")
        res_dlt = st.session_state.dlt_calib_results

        # --- ADD DEBUG OUTPUT ---
        with st.expander("DEBUG: Raw DLT Results Dictionary"):
             st.write(res_dlt)
        # --- END DEBUG OUTPUT ---


        # Part 1 Results Display
        st.markdown("**Part 1 Results (Intrinsics from Known Extrinsics)**")
        if res_dlt.get("K_part1") is not None:
            st.write("Estimated Intrinsic Matrix (K):")
            st.code(f"{res_dlt['K_part1']}")
        else:
            st.info("Part 1 calibration was not successful or data was missing/invalid.")

        st.markdown("---") # Separator

        # Part 2 Results Display
        st.markdown("**Part 2 Results (Full Calibration from World Coordinates)**")
        if res_dlt.get("P_part2") is not None: # Check if Part 2 ran successfully enough to produce P
            col_p2_1, col_p2_2 = st.columns(2)
            with col_p2_1:
                # Show RMS error if calculated
                rms_val = res_dlt.get('rms_error_part2', np.inf)
                if np.isfinite(rms_val):
                     st.metric("RMS Reprojection Error", f"{rms_val:.4f} px")
                else:
                     st.metric("RMS Reprojection Error", "N/A")

                st.write("Full Projection Matrix (P):")
                st.code(f"{res_dlt['P_part2']}")

                if res_dlt.get("K_part2") is not None:
                    st.write("Decomposed Intrinsic Matrix (K):")
                    st.code(f"{res_dlt['K_part2']}")
                else:
                    st.write("Intrinsic Matrix (K): Decomposition failed.")

            with col_p2_2:
                if res_dlt.get("R_part2") is not None:
                    st.write("Decomposed Rotation Matrix (R):")
                    st.code(f"{res_dlt['R_part2']}")
                    # Use try-except for determinant calculation in case R is not valid
                    try:
                        det_val = det(res_dlt['R_part2'])
                        st.caption(f"(det(R) ≈ {det_val:.4f})") # Show determinant
                    except:
                        st.caption("(Could not compute det(R))")
                else:
                     st.write("Rotation Matrix (R): Decomposition failed.")

                if res_dlt.get("t_part2") is not None:
                    st.write("Decomposed Translation Vector (t):")
                    st.code(f"{res_dlt['t_part2']}")
                else:
                    st.write("Translation Vector (t): Decomposition failed or K was singular.")

            # Visualization Display
            st.markdown("**Part 2 Visualization**")
            # Check if the figure object exists in the results
            fig_to_plot = res_dlt.get('visualization_fig_part2')
            if fig_to_plot:
                st.pyplot(fig_to_plot) # Display the Matplotlib figure using st.pyplot
            elif res_dlt.get('img_part2') is not None: # Fallback: Show image if plot failed but image loaded
                st.warning("Visualization plot failed to generate (or reprojection failed), showing original image.")
                # Convert BGR to RGB before displaying with st.image
                try:
                    img_rgb = cv2.cvtColor(res_dlt['img_part2'], cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="DLT Input Image")
                except Exception as img_e:
                    st.error(f"Failed to display fallback image: {img_e}")
            else:
                st.warning("Visualization could not be generated (missing image or plot error).")

        else: # If P_part2 was None
            st.info("Part 2 calibration was not successful or data was missing/invalid.")