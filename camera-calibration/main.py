# ==============================================================================
# Required Libraries
# ==============================================================================
import streamlit as st
import numpy as np
import cv2 # OpenCV for computer vision tasks
import io # For handling byte streams (file uploads)
import time # For pausing briefly
import random # For data efficiency experiment sampling
import pandas as pd # For data handling in experiments
import matplotlib.pyplot as plt # For plotting
from matplotlib.patches import Circle # For DLT visualization (optional styling)
import os # Potentially useful for file paths (though not strictly needed with uploads)
import scipy.io # For loading .mat files (DLT mode)
from numpy.linalg import inv, svd, det, pinv # Linear algebra functions for DLT
from scipy.linalg import rq # RQ decomposition for DLT P matrix


# ==============================================================================
# Part 0: Configuration & Constants
# ==============================================================================
# Default checkerboard parameters (inner corners)
DEFAULT_PATTERN_SIZE = (7, 6) # (cols, rows) for OpenCV findChessboardCorners
DEFAULT_SQUARE_SIZE = 25.0 # Size in millimeters

# ==============================================================================
# Part 1: Utility Functions
# ==============================================================================

def decode_image(file_uploader_content):
    """Decodes image from Streamlit file uploader bytes."""
    try:
        # Ensure the file pointer is at the beginning
        if hasattr(file_uploader_content, 'seek'):
            file_uploader_content.seek(0)
        # Read file content as bytes
        file_bytes = np.asarray(bytearray(file_uploader_content.read()), dtype=np.uint8)
        # Decode image from byte array
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Load as color image
        if img is None:
            st.error("Failed to decode image. Content might be invalid or corrupted.")
            return None
        return img
    except Exception as e:
        st.error(f"Error decoding image '{getattr(file_uploader_content, 'name', '')}': {e}")
        return None

def load_mat_file(uploaded_file):
    """Loads data safely from an uploaded .mat file."""
    if uploaded_file is None:
        # This case should ideally be handled before calling, but added for safety
        st.error("Attempted to load a None MAT file.")
        return None
    try:
        uploaded_file.seek(0)
        # Use BytesIO to treat the uploaded file bytes as a file-like object
        data = scipy.io.loadmat(io.BytesIO(uploaded_file.read()))
        return data
    except Exception as e:
        st.error(f"Error loading MAT file '{getattr(uploaded_file, 'name', '')}': {e}")
        return None

# Helper function for Streamlit rerun functionality (handles version differences)
def rerun_if_possible():
    """Attempts to rerun the Streamlit app."""
    try:
        st.rerun() # Preferred method in newer Streamlit versions (>= 1.14.0)
    except AttributeError:
        try:
            st.experimental_rerun() # Older method (< 1.14.0)
        except AttributeError:
            # Fallback if neither method exists (very old Streamlit or unexpected environment)
            st.warning("Auto-rerun functionality not detected. Please refresh the page manually if needed.")


# ==============================================================================
# Part 2: Core Calibration Logic Functions (Standard & Fisheye)
# ==============================================================================
def process_images(image_files, pattern_size_cv, square_size):
    """
    Processes uploaded image files to find checkerboard corners.

    Args:
        image_files (list): List of uploaded file objects from Streamlit.
        pattern_size_cv (tuple): (cols, rows) of inner corners.
        square_size (float): Side length of a checkerboard square in mm.

    Returns:
        tuple: (objpoints_all, imgpoints_all, first_valid_shape, valid_indices)
               Returns (None, None, None, None) on failure.
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points in the checkerboard coordinate system
    # (0,0,0), (1,0,0), ..., (cols-1, rows-1, 0) scaled by square_size
    objp = np.zeros((pattern_size_cv[0] * pattern_size_cv[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size_cv[0], 0:pattern_size_cv[1]].T.reshape(-1, 2)
    objp *= square_size # Scale to real-world units (mm)

    # Lists to store results
    objpoints_all = [] # 3D points for each valid image
    imgpoints_all = [] # 2D detected corner points for each valid image
    image_shapes = [] # Store shapes (width, height) of ALL processed images
    valid_image_indices = [] # Indices of images where corners were successfully found

    if not image_files:
        st.warning("No image files provided for processing.")
        return None, None, None, None

    # --- Progress Reporting Elements ---
    progress_text_area = st.empty()
    prog_bar = st.progress(0)
    total_files = len(image_files)
    processed_count = 0
    found_count = 0

    # --- Process Each Image ---
    for i, uploaded_file in enumerate(image_files):
        file_name = getattr(uploaded_file, 'name', f'image_{i+1}')
        progress_text_area.text(f"Processing {i+1}/{total_files}: {file_name}...")

        try:
            img = decode_image(uploaded_file)
            if img is None:
                st.warning(f"Skipping file {i+1} ('{file_name}'): Could not decode image.")
                continue # Skip this file if decoding failed

            processed_count += 1
            # Convert to grayscale for corner detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            current_shape = gray.shape[::-1] # Get (width, height) tuple
            image_shapes.append(current_shape)

            # Find checkerboard corners
            # Add flags like CALIB_CB_ADAPTIVE_THRESH? Maybe not needed initially.
            ret, corners = cv2.findChessboardCorners(gray, pattern_size_cv, None)

            # If corners are found, refine them and store the points
            if ret:
                found_count += 1
                valid_image_indices.append(i) # Store index of the valid image
                objpoints_all.append(objp) # Add the same 3D object points for this view

                # Refine corner locations using sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints_all.append(corners2) # Add the refined 2D image points

        except Exception as e:
            # Catch any other unexpected errors during processing
            st.warning(f"Error processing image {i+1} ({file_name}): {e}")

        # Update progress bar
        prog_bar.progress((i + 1) / total_files)

    # --- Cleanup Progress Elements ---
    prog_bar.empty()
    progress_text_area.empty()

    # --- Handle No Corners Found ---
    if not imgpoints_all:
        st.error("Checkerboard corners could not be detected in any of the uploaded images.")
        return None, None, None, None

    st.success(f"Found corners in {found_count} out of {processed_count} processed images.")

    # --- Check Image Dimensions ---
    # Collect shapes only from images where corners were found
    valid_shapes = [image_shapes[idx] for idx in valid_image_indices]
    first_valid_shape = valid_shapes[0] if valid_shapes else None

    # Warn if dimensions are inconsistent among the *valid* images
    if len(set(valid_shapes)) > 1:
        st.warning(f"Inconsistent image dimensions detected among valid images ({set(valid_shapes)}). "
                   f"Calibration will use the shape of the first valid image: {first_valid_shape}. "
                   "Results might be suboptimal if dimensions vary significantly.")

    return objpoints_all, imgpoints_all, first_valid_shape, valid_image_indices


def run_standard_calibration(objpoints, imgpoints, gray_shape):
    """
    Performs standard camera calibration using cv2.calibrateCamera.

    Args:
        objpoints (list): List of 3D object points (numpy arrays).
        imgpoints (list): List of 2D image points (numpy arrays).
        gray_shape (tuple): (width, height) of the images used.

    Returns:
        tuple: (ret, mtx, dist, rvecs, tvecs) or (False, None, None, None, None) on failure.
    """
    # --- Input Validation ---
    if not objpoints or not imgpoints or gray_shape is None or len(objpoints) != len(imgpoints):
        print("[Error] Standard Calibration Preconditions Failed: Check inputs (objpoints, imgpoints, shape).")
        st.error("Internal Error: Invalid inputs provided to standard calibration function.")
        return False, None, None, None, None
    # Need at least 2 views for standard calibration typically
    if len(objpoints) < 2:
        print("[Error] Standard Calibration Failed: Need at least 2 views.")
        st.error("Standard Calibration Failed: Requires at least 2 valid views.")
        return False, None, None, None, None

    try:
        # Ensure correct data types (float32 recommended for calibrateCamera)
        objpoints_f32 = [op.astype(np.float32) for op in objpoints]
        imgpoints_f32 = [ip.astype(np.float32) for ip in imgpoints]

        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_f32, imgpoints_f32, gray_shape, None, None)

        # Check if calibration was successful
        if not ret or mtx is None or dist is None:
            print("[Error] cv2.calibrateCamera returned False or None.")
            st.error("OpenCV Calibration Failed (cv2.calibrateCamera). "
                     "Check image quality, corner detection, and view variety.")
            return False, None, None, None, None

        return ret, mtx, dist, rvecs, tvecs

    except cv2.error as e:
        # Handle specific OpenCV errors
        print(f"[Error] OpenCV Error during Standard Calibration: {e}")
        st.error(f"OpenCV Error during Standard Calibration: {e}")
        if "solvePnP" in str(e) or "more points needed" in str(e):
            print(" -> Check if enough points per view or enough views are provided.")
            st.warning("This OpenCV error might be due to insufficient points per view (>3 needed) or insufficient valid views.")
        return False, None, None, None, None
    except Exception as e:
        # Handle other unexpected errors
        print(f"[Error] Unexpected Error during Standard Calibration: {e}")
        st.error(f"Unexpected Error during Standard Calibration: {e}")
        return False, None, None, None, None


def run_fisheye_calibration(objpoints, imgpoints, gray_shape):
    """
    Performs fisheye camera calibration using cv2.fisheye.calibrate.

    Args:
        objpoints (list): List of 3D object points (numpy arrays).
        imgpoints (list): List of 2D image points (numpy arrays).
        gray_shape (tuple): (width, height) of the images used.

    Returns:
        tuple: (ret, K, D, rvecs, tvecs) or (False, None, None, None, None) on failure.
               K is the intrinsic matrix, D contains fisheye distortion coeffs (k1-k4).
    """
    # --- Input Validation ---
    if not objpoints or not imgpoints or gray_shape is None or len(objpoints) != len(imgpoints):
        print("[Error] Fisheye Calibration Preconditions Failed: Check inputs.")
        st.error("Internal Error: Invalid inputs provided to fisheye calibration function.")
        return False, None, None, None, None
    # Need multiple views
    if len(objpoints) < 2:
         print("[Error] Fisheye Calibration Failed: Need at least 2 views.")
         st.error("Fisheye Calibration Failed: Requires at least 2 valid views.")
         return False, None, None, None, None

    # --- Prepare Data and Parameters ---
    # Initialize K and D matrices as required by the function (fisheye uses 4 distortion coeffs)
    K_init = np.zeros((3, 3), dtype=np.float64)
    D_init = np.zeros((4, 1), dtype=np.float64)

    try:
        # Fisheye calibration requires specific shapes and float64 type
        # Input objpoints: list of (Npts, 3) -> Convert to list of (Npts, 1, 3) float64
        # Input imgpoints: list of (Npts, 1, 2) -> Convert to list of (Npts, 1, 2) float64
        objp_f64 = [op.reshape(-1, 1, 3).astype(np.float64) for op in objpoints]
        imgp_f64 = [ip.reshape(-1, 1, 2).astype(np.float64) for ip in imgpoints]

        # Set calibration flags
        # CALIB_FIX_SKEW assumes zero skew (alpha=0). CALIB_RECOMPUTE_EXTRINSIC is generally good.
        # CALIB_CHECK_COND checks condition number - useful for diagnosing unstable results.
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        # Termination criteria for the iterative optimization process
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # --- Perform Fisheye Calibration ---
        # print(f"Fisheye input shapes: objp[0]={objp_f64[0].shape}, imgp[0]={imgp_f64[0].shape}, gray_shape={gray_shape}") # Debug print
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objp_f64, imgp_f64, gray_shape, K_init, D_init,
            flags=flags, criteria=criteria
        )

        # --- Check Results ---
        if not ret or K is None or D is None:
             print("[Error] cv2.fisheye.calibrate returned False or None.")
             st.error("OpenCV Fisheye Calibration Failed (cv2.fisheye.calibrate). "
                      "Ensure views cover the Field of View well, especially edges/corners.")
             return False, None, None, None, None

        return ret, K, D, rvecs, tvecs

    except cv2.error as e:
        # Handle specific OpenCV errors
        print(f"[Error] OpenCV Error during Fisheye Calibration: {e}")
        st.error(f"OpenCV Error during Fisheye Calibration: {e}")
        # Check condition error often means poor view variety for the model
        if "CALIB_CHECK_COND" in str(e):
             print(" -> Check Calibration Condition Error: Ensure views have sufficient variety (angles, positions).")
             st.warning("Check Condition Error: This often means the views are too similar or don't provide enough geometric constraints for the fisheye model. Try adding more varied images.")
        return False, None, None, None, None
    except Exception as e:
        # Handle other unexpected errors
        print(f"[Error] Unexpected Error during Fisheye Calibration: {e}")
        st.error(f"Unexpected Error during Fisheye Calibration: {e}")
        return False, None, None, None, None


def calculate_reprojection_error(objpoints_full, imgpoints_full, mtx, dist, rvecs_calib, tvecs_calib, is_fisheye=False):
    """
    Calculates the Root Mean Square (RMS) reprojection error.

    Args:
        objpoints_full (list): List of 3D object points for all views.
        imgpoints_full (list): List of detected 2D image points for all views.
        mtx (np.array): Camera intrinsic matrix (K).
        dist (np.array): Distortion coefficients (standard or fisheye).
        rvecs_calib (list): List of rotation vectors from calibration (optional).
        tvecs_calib (list): List of translation vectors from calibration (optional).
        is_fisheye (bool): Flag indicating if fisheye model should be used.

    Returns:
        float: RMS reprojection error in pixels, or np.inf on failure.
    """
    total_err_sq = 0.0 # Sum of squared errors
    total_pts = 0      # Total number of points processed

    # --- Input Validation ---
    if not objpoints_full or not imgpoints_full or len(objpoints_full) != len(imgpoints_full):
        print("[Error] Reprojection Error Calc: Input lists invalid or mismatched lengths.")
        return np.inf
    if mtx is None or dist is None:
        print("[Error] Reprojection Error Calc: Missing camera matrix or distortion coefficients.")
        return np.inf


    n_views = len(objpoints_full)

    # Ensure rvecs and tvecs lists match the number of views if provided
    # If not provided (e.g., when evaluating subset calib on full data for standard model),
    # create lists of Nones to trigger internal pose estimation via solvePnP.
    rvecs = rvecs_calib if rvecs_calib is not None and len(rvecs_calib) == n_views else [None] * n_views
    tvecs = tvecs_calib if tvecs_calib is not None and len(tvecs_calib) == n_views else [None] * n_views

    # --- Iterate Through Each View ---
    for i in range(n_views):
        # Get points for the current view, ensuring correct types
        objp = np.asarray(objpoints_full[i], dtype=np.float32) # Object points (Npts, 3)
        imgp_det = np.asarray(imgpoints_full[i], dtype=np.float32).reshape(-1, 1, 2) # Detected points (Npts, 1, 2)

        if len(objp) == 0: continue # Skip if no points for this view (shouldn't happen with process_images)

        imgp_proj = None # Initialize projected points for this view to None

        try:
            rvec = rvecs[i] # Rotation vector for this view (might be None)
            tvec = tvecs[i] # Translation vector for this view (might be None)

            # --- Project 3D points to 2D image plane based on camera model ---
            if is_fisheye:
                # Fisheye projection requires rvec/tvec (cannot estimate internally here easily)
                if rvec is not None and tvec is not None:
                    # cv2.fisheye.projectPoints needs float64 inputs
                    objp_f64 = objp.reshape(-1, 1, 3).astype(np.float64)
                    rvec_f64 = np.asarray(rvec).astype(np.float64).reshape(1, 3) # Ensure shape (1,3) or (3,1)
                    tvec_f64 = np.asarray(tvec).astype(np.float64).reshape(1, 3) # Ensure shape (1,3) or (3,1)
                    mtx_f64 = np.asarray(mtx).astype(np.float64)
                    dist_f64 = np.asarray(dist).astype(np.float64)

                    imgp_proj, _ = cv2.fisheye.projectPoints(objp_f64, rvec_f64, tvec_f64, mtx_f64, dist_f64)
                else:
                    # If poses aren't provided for fisheye, we cannot calculate the error for this view
                    # print(f"[Warning] Skipping view {i} for fisheye error: Missing pose (rvec/tvec required).")
                    continue

            else: # Standard camera model (pinhole)
                # Use provided rvec/tvec if available
                if rvec is not None and tvec is not None:
                     # cv2.projectPoints arguments type check (refer to OpenCV docs if errors occur)
                     # Typically objp: F32/F64, r/tvec: F64, mtx: F64, dist: F64
                     imgp_proj, _ = cv2.projectPoints(objp.astype(np.float32),
                                                      np.asarray(rvec).astype(np.float64),
                                                      np.asarray(tvec).astype(np.float64),
                                                      mtx.astype(np.float64),
                                                      dist.astype(np.float64))
                else:
                    # Estimate pose using solvePnP if rvec/tvec are not provided
                    if len(objp) <= 3:
                        # print(f"[Warning] Skipping view {i} for std error: Need > 3 points for solvePnP.")
                        continue # solvePnP requires at least 4 points

                    # cv2.solvePnP arguments type check
                    # Typically objp: F32/F64(N,3 or Nx1x3), imgp: F32/F64(N,2 or Nx1x2), mtx: F64, dist: F64
                    ret_pnp, rvec_pnp, tvec_pnp = cv2.solvePnP(objp.astype(np.float32),
                                                               imgp_det.astype(np.float32),
                                                               mtx.astype(np.float64),
                                                               dist.astype(np.float64))
                    if ret_pnp:
                        # Project points using the estimated pose
                        imgp_proj, _ = cv2.projectPoints(objp.astype(np.float32), rvec_pnp, tvec_pnp,
                                                         mtx.astype(np.float64), dist.astype(np.float64))
                    else:
                        # print(f"[Warning] Skipping view {i} for std error: solvePnP failed to estimate pose.")
                        continue # Pose estimation failed for this view

            # --- Calculate error for this view if projection was successful ---
            if imgp_proj is not None:
                 # Ensure projected points have the same shape and type as detected points
                imgp_proj = imgp_proj.reshape(-1, 1, 2).astype(np.float32)

                if imgp_proj.shape == imgp_det.shape:
                    # Calculate squared Euclidean distance point-wise: sum((x1-x2)^2 + (y1-y2)^2)
                    err_sq_per_point = np.sum((imgp_det - imgp_proj)**2, axis=2) # Sum over x,y coords -> shape (Npts, 1)
                    total_err_sq += np.sum(err_sq_per_point) # Add sum of squared errors for this view
                    total_pts += len(objp) # Increment total point count
                # else: print(f"[Warning] Shape mismatch view {i}: Detected {imgp_det.shape}, Projected {imgp_proj.shape}")

        except cv2.error as e:
            # Catch OpenCV errors during projection or pose estimation
            # print(f"[Warning] OpenCV Error during Reprojection Calculation (View {i}): {e}")
            pass # Continue to the next image/view
        except Exception as e:
            # Catch other unexpected errors
            # print(f"[Warning] Unexpected Error during Reprojection Calculation (View {i}): {e}")
            pass # Continue to the next image/view

    # --- Calculate Final RMS Error ---
    if total_pts == 0:
        print("[Error] Reprojection Error Calc: No points were successfully projected across all views.")
        return np.inf # Avoid division by zero if no points were processed

    mean_error_sq = total_err_sq / total_pts # Mean of the squared errors
    rms_error = np.sqrt(mean_error_sq)      # Root of the mean squared error
    return rms_error


# ==============================================================================
# Part 3: DLT Calibration Functions
# ==============================================================================
# These functions are adapted to expect inputs where points are ROWS (NxD format)

def dlt_calibrate_intrinsics(pts_2D_Nx2, cam_pts_3D_Nx3):
    """
    Estimates intrinsic matrix K using DLT assuming known extrinsics (3D points
    are in the camera's coordinate system).

    Args:
        pts_2D_Nx2 (np.array): Nx2 array of 2D image points.
        cam_pts_3D_Nx3 (np.array): Nx3 array of corresponding 3D points in camera frame.

    Returns:
        np.array: 3x3 intrinsic matrix K, or None on failure.
    """
    # --- Input Validation ---
    if pts_2D_Nx2 is None or cam_pts_3D_Nx3 is None:
        st.error("DLT Intrinsics Error: Input points are None.")
        return None
    num_points = pts_2D_Nx2.shape[0]
    if num_points < 6 or num_points != cam_pts_3D_Nx3.shape[0]:
        st.error(f"DLT Intrinsics Error: Need at least 6 corresponding points. Got {num_points} pairs.")
        return None
    if pts_2D_Nx2.shape[1] != 2 or cam_pts_3D_Nx3.shape[1] != 3:
        st.error(f"DLT Intrinsics Error: Incorrect input shapes. Expected Nx2 and Nx3. "
                 f"Got {pts_2D_Nx2.shape} and {cam_pts_3D_Nx3.shape}.")
        return None

    try:
        # Points are already Nx2 and Nx3
        one = np.ones((num_points, 1))

        # Build matrix A = [u, v, 1] for all points (Nx3)
        A = np.hstack((pts_2D_Nx2, one)) # Shape (N, 3)

        # Build matrix B = [X/Z, Y/Z, 1] for all points (Nx3)
        X_cam = cam_pts_3D_Nx3[:, 0] # Shape (N,)
        Y_cam = cam_pts_3D_Nx3[:, 1] # Shape (N,)
        Z_cam = cam_pts_3D_Nx3[:, 2] # Shape (N,)

        # Avoid division by zero or very small numbers for Z
        safe_Z = np.sign(Z_cam) * np.maximum(np.abs(Z_cam), 1e-8)

        x1 = (X_cam / safe_Z).reshape(-1, 1) # Shape (N, 1)
        x2 = (Y_cam / safe_Z).reshape(-1, 1) # Shape (N, 1)
        B = np.hstack((x1, x2, one)) # Shape (N, 3)

        # Solve K * B' = A' => K = A' * pinv(B') (as derived from standalone code logic)
        K = A.T @ pinv(B.T) # (3xN) @ (Nx3) -> 3x3

        # Normalize K so K[2, 2] = 1
        if abs(K[2, 2]) > 1e-8:
            K = K / K[2, 2]
        else:
            # If K[2,2] is zero, normalization fails. The matrix might be ill-conditioned.
            st.warning("DLT Intrinsics Warning: K[2, 2] is close to zero. Normalization skipped.")
            # Keep K as is, but it might indicate problems upstream.

        return K

    except np.linalg.LinAlgError as e:
        st.error(f"DLT Intrinsics Linear Algebra Error: {e}")
        return None
    except Exception as e:
        st.error(f"DLT Intrinsics Unexpected Error: {e}")
        return None


def dlt_calibrate_projection(pts_2d_Nx2, pts_3d_Nx3):
    """
    Estimates the 3x4 projection matrix P using DLT from world 3D points
    to image 2D points.

    Args:
        pts_2d_Nx2 (np.array): Nx2 array of 2D image points.
        pts_3d_Nx3 (np.array): Nx3 array of corresponding 3D world points.

    Returns:
        np.array: 3x4 projection matrix P, or None on failure.
    """
    # --- Input Validation ---
    if pts_2d_Nx2 is None or pts_3d_Nx3 is None:
         st.error("DLT Projection Error: Input points are None.")
         return None
    n = pts_2d_Nx2.shape[0]
    if pts_2d_Nx2.shape[1] != 2 or pts_3d_Nx3.shape[1] != 3 or n != pts_3d_Nx3.shape[0]:
        st.error(f"DLT Projection Error: Shape mismatch. Expected Nx2, Nx3. "
                 f"Got pts_2d={pts_2d_Nx2.shape}, pts_3d={pts_3d_Nx3.shape}")
        return None
    if n < 6: # DLT requires at least 6 points for a unique solution
        st.error(f"DLT Projection Error: Need at least 6 points. Got {n}.")
        return None

    try:
        # Convert 3D world points to homogeneous coordinates (Nx4)
        X_h = np.hstack((pts_3d_Nx3, np.ones((n, 1)))) # Shape (N, 4)

        # Construct the DLT matrix M (size 2n x 12)
        M = np.zeros((2 * n, 12))
        for i in range(n):
            Xi = X_h[i, :]  # Homogeneous 3D point for row i (1x4)
            ui, vi = pts_2d_Nx2[i, 0], pts_2d_Nx2[i, 1] # 2D image point coords

            # Fill rows 2i and 2i+1 of M based on DLT equations
            M[2 * i, :]   = np.hstack((np.zeros(4), -Xi, vi * Xi))
            M[2 * i + 1, :] = np.hstack((Xi, np.zeros(4), -ui * Xi))

        # Solve the system M * P_vec = 0 using Singular Value Decomposition (SVD)
        U, S, Vh = svd(M)

        # The solution P_vec corresponds to the null space of M, which is the
        # last column of V (or last row of Vh, the transpose of V).
        P = Vh[-1, :].reshape((3, 4)) # Reshape the 12-element vector into a 3x4 matrix

        return P

    except np.linalg.LinAlgError as e:
         st.error(f"DLT Projection Linear Algebra Error (SVD failed?): {e}")
         return None
    except Exception as e:
        st.error(f"DLT Projection Unexpected Error: {e}")
        return None


def dlt_P_to_KRt(P):
    """
    Decomposes the 3x4 projection matrix P into K (intrinsic), R (rotation),
    and t (translation) using RQ decomposition.

    Args:
        P (np.array): 3x4 projection matrix.

    Returns:
        tuple: (K, R, t) where K is 3x3, R is 3x3, t is 3x1.
               Returns (None, None, None) on failure.
    """
    # --- Input Validation ---
    if P is None or P.shape != (3, 4):
        st.error(f"DLT KRt Error: Invalid projection matrix P. Expected 3x4, got {P.shape if P is not None else 'None'}.")
        return None, None, None

    try:
        # Extract the left 3x3 submatrix M = K * R
        M = P[0:3, 0:3]

        # Perform RQ decomposition on M
        # rq returns R (upper triangular) and Q (orthogonal = rotation)
        # Note: scipy.linalg.rq convention matches K=R, R=Q here
        K_rq, R_rq = rq(M)

        # Ensure K (the upper triangular matrix from RQ) has positive diagonal elements.
        # The sign of the diagonal elements affects the signs of columns in R (Q from RQ).
        T_sign = np.diag(np.sign(np.diag(K_rq)))
        # Handle potential zeros on the diagonal (though unlikely for valid K)
        T_sign[T_sign == 0] = 1 # Replace 0 with 1 to avoid issues

        K = K_rq @ T_sign # Adjust K: K = K_rq * T
        R = T_sign @ R_rq # Adjust R: R = inv(T) * R_rq = T * R_rq (since T=inv(T) for diagonal +/-1)

        # Normalize K so that K[2, 2] = 1
        K_norm = K
        if abs(K[2, 2]) > 1e-8:
            K_norm = K / K[2, 2]
        else:
            st.warning("DLT KRt Warning: K[2, 2] near zero during normalization. K matrix might be ill-conditioned.")

        # Check if R is a valid rotation matrix (determinant should be +1)
        det_R = det(R)
        if abs(det_R - 1.0) > 1e-3: # Allow for small numerical inaccuracies
            # If determinant is -1, it's a reflection matrix. This can happen with noisy data.
            # Common fix: Multiply P by -1 and redo decomposition? Or adjust R,t?
            # For simplicity, we just warn the user here.
            st.warning(f"DLT KRt Warning: Determinant of calculated R is {det_R:.4f} (expected ≈ +1). "
                       "This might indicate a reflection or issues with the input data/calibration.")

        # Calculate the translation vector t using P = K [R | t] => K*t = P[:, 3] => t = inv(K) * P[:, 3]
        # Use the normalized K for calculating t if possible
        try:
            t = inv(K_norm) @ P[:, 3]
        except np.linalg.LinAlgError:
             # If normalized K is singular (det=0), try with unnormalized K
             st.warning("DLT KRt Warning: Could not invert normalized K matrix (singular?). Trying unnormalized K.")
             try:
                 t = inv(K) @ P[:, 3]
             except np.linalg.LinAlgError:
                  st.error("DLT KRt Error: Could not invert K matrix (normalized or unnormalized) to find translation.")
                  # Return K and R even if t fails
                  return K_norm, R, None

        # Return results: K (normalized), R, and t (as a column vector)
        return K_norm, R, t.reshape(3, 1)

    except np.linalg.LinAlgError as e:
        st.error(f"DLT KRt Linear Algebra Error during RQ decomposition or inversion: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"DLT KRt Unexpected Error: {e}")
        return None, None, None


def dlt_compute_reprojection_error(P, pts_3d_Nx3, pts_2d_Nx2):
    """
    Computes the Root Mean Square (RMS) reprojection error given the projection
    matrix P, 3D world points, and original 2D image points.

    Args:
        P (np.array): 3x4 projection matrix.
        pts_3d_Nx3 (np.array): Nx3 array of 3D world points.
        pts_2d_Nx2 (np.array): Nx2 array of original 2D image points.

    Returns:
        tuple: (rms_error, x_proj) where rms_error is the RMS error in pixels
               (or np.inf on failure) and x_proj is the Nx2 array of reprojected points.
    """
    # --- Input Validation ---
    if P is None or pts_3d_Nx3 is None or pts_2d_Nx2 is None:
        st.error("DLT Reprojection Error: Input P, pts_3d, or pts_2d is None.")
        return np.inf, None
    n = pts_3d_Nx3.shape[0]
    if n == 0 or n != pts_2d_Nx2.shape[0] or pts_3d_Nx3.shape[1] != 3 or pts_2d_Nx2.shape[1] != 2:
        st.error(f"DLT Reprojection Error: Mismatched or invalid point shapes. "
                 f"Expected Nx3, Nx2. Got {pts_3d_Nx3.shape}, {pts_2d_Nx2.shape}")
        return np.inf, None

    try:
        # Convert 3D world points to homogeneous coordinates (Nx4)
        X_h = np.hstack((pts_3d_Nx3, np.ones((n, 1)))) # Shape (N, 4)

        # Project 3D points to 2D image plane using P: x_proj_h = P * X_h'
        # P(3x4) @ X_h.T(4xN) results in (3xN) homogeneous 2D coordinates
        x_proj_h_3xN = P @ X_h.T

        # Transpose to Nx3 for easier handling: [u*w, v*w, w] for each point
        x_proj_h = x_proj_h_3xN.T # Shape (N, 3)

        # --- Convert projected homogeneous coordinates to Euclidean 2D coordinates ---
        # Extract the scaling factor w (last column)
        w = x_proj_h[:, 2]
        # Avoid division by zero or very small numbers
        w_safe = np.sign(w) * np.maximum(np.abs(w), 1e-8) # Avoid division by zero

        # Divide the first two columns (u*w, v*w) by w to get (u, v)
        x_proj = x_proj_h[:, :2] / w_safe[:, np.newaxis] # Shape (N, 2)

        # --- Calculate Reprojection Error ---
        # Compute the Euclidean distance between original (pts_2d_Nx2) and reprojected (x_proj) points
        # errors = sqrt((u_orig - u_proj)^2 + (v_orig - v_proj)^2) for each point
        errors = np.linalg.norm(pts_2d_Nx2 - x_proj, axis=1) # Calculate norm along axis 1 -> shape (N,)

        # Calculate the Root Mean Square (RMS) error
        rms_error = np.sqrt(np.mean(errors**2))

        return rms_error, x_proj # Return RMS error and the reprojected points (Nx2)

    except Exception as e:
        st.error(f"DLT Reprojection Calculation Error: {e}")
        return np.inf, None


def dlt_visualize_calibration(img, pts_2d_original_Nx2, pts_2d_reprojected_Nx2):
    """
    Visualizes DLT calibration by plotting original and reprojected points
    on the provided image.

    Args:
        img (np.array): The image (loaded as a NumPy array, e.g., using cv2.imread/decode_image).
        pts_2d_original_Nx2 (np.array): Nx2 array of original detected 2D points.
        pts_2d_reprojected_Nx2 (np.array): Nx2 array of reprojected 2D points.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the plot,
                                  or None on failure.
    """
    # --- Input Validation ---
    if img is None or pts_2d_original_Nx2 is None or pts_2d_reprojected_Nx2 is None:
        st.warning("DLT Visualization Warning: Missing image or point data.")
        return None
    if len(pts_2d_original_Nx2) != len(pts_2d_reprojected_Nx2) or pts_2d_original_Nx2.shape[1]!=2 or pts_2d_reprojected_Nx2.shape[1]!=2:
         st.warning(f"DLT Visualization Warning: Mismatched number/shape of original/reprojected points. "
                    f"Got {pts_2d_original_Nx2.shape} and {pts_2d_reprojected_Nx2.shape}")
         # Attempt to plot anyway if lengths match, might indicate upstream issue
         if len(pts_2d_original_Nx2) != len(pts_2d_reprojected_Nx2):
              return None


    try:
        # --- Create Plot ---
        fig, ax = plt.subplots(1, figsize=(10, 8)) # Adjust figure size as needed
        ax.set_aspect('equal') # Ensure correct aspect ratio

        # Display the image. Convert BGR (OpenCV default) to RGB (Matplotlib default).
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Plot original points (e.g., red circles)
        ax.plot(pts_2d_original_Nx2[:, 0], pts_2d_original_Nx2[:, 1],
                marker='o', color='red', markersize=6, linestyle='None', # No connecting line
                mfc='none', mew=1.5, # Hollow circles with red edge
                label='Original Detected Points')

        # Plot reprojected points (e.g., green crosses)
        ax.plot(pts_2d_reprojected_Nx2[:, 0], pts_2d_reprojected_Nx2[:, 1],
                marker='x', color='lime', markersize=8, linestyle='None', # No connecting line
                mew=1.5, # Marker edge width
                label='Reprojected Points')

        # Optionally: Add lines connecting original to reprojected points (error vectors)
        for i in range(len(pts_2d_original_Nx2)):
             ax.plot([pts_2d_original_Nx2[i, 0], pts_2d_reprojected_Nx2[i, 0]],
                     [pts_2d_original_Nx2[i, 1], pts_2d_reprojected_Nx2[i, 1]],
                     color='yellow', linestyle='-', linewidth=0.7, alpha=0.8) # Thin yellow lines

        # --- Customize Plot Appearance ---
        ax.legend()
        ax.set_title("DLT Calibration: Original vs. Reprojected Points")
        # Set axis limits to match image dimensions
        h, w = img.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0) # Invert y-axis to match image coordinates (origin top-left)
        ax.set_xlabel("X Pixel Coordinate")
        ax.set_ylabel("Y Pixel Coordinate")
        ax.axis('on') # Ensure axes lines and ticks are visible
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        return fig # Return the figure object for display in Streamlit

    except Exception as e:
        st.error(f"DLT Visualization Error: {e}")
        return None # Return None if plotting fails


# ==============================================================================
# Part 4: Experiment Function (Data Efficiency - Images Only)
# ==============================================================================
def run_data_efficiency_images_experiment(calibrate_func, image_files, pattern_size_cv, square_size, num_images_list, is_fisheye, num_runs=3):
    """
    Runs calibration (standard or fisheye) with varying numbers of images and
    evaluates the reprojection error on the *full* set of valid images.

    Args:
        calibrate_func (function): The calibration function to use
                                   (e.g., run_standard_calibration or run_fisheye_calibration).
        image_files (list): The original list of uploaded image file objects.
        pattern_size_cv (tuple): Checkerboard pattern size (cols, rows).
        square_size (float): Checkerboard square size (mm).
        num_images_list (list): List of image counts to test (e.g., [2, 3, 5, 10]).
        is_fisheye (bool): True if using fisheye model, False for standard.
        num_runs (int): Number of random trials per image count.

    Returns:
        pd.DataFrame: DataFrame containing 'num_images', 'avg_rms', 'std_rms',
                      or an empty DataFrame on failure.
    """
    method_name = "Fisheye" if is_fisheye else "Standard"
    results = {'num_images': [], 'avg_rms': [], 'std_rms': []} # To store experiment results

    _placeholder = st.empty() # For status messages

    # --- Ensure Base Data is Processed ---
    _placeholder.info(f"Checking/Processing all images for {method_name} baseline evaluation...")
    # Use existing processed data from session state if available
    objpoints_full = st.session_state.get('objpoints_processed')
    imgpoints_full = st.session_state.get('imgpoints_processed')
    img_shape = st.session_state.get('img_shape_processed')

    # If not processed yet, process them now
    if not objpoints_full or not imgpoints_full or not img_shape:
        print("Reprocessing images required for experiment...")
        objpoints_full, imgpoints_full, img_shape, valid_indices = process_images(image_files, pattern_size_cv, square_size)
        # Store processed data back into session state if successful
        if objpoints_full:
            st.session_state.objpoints_processed = objpoints_full
            st.session_state.imgpoints_processed = imgpoints_full
            st.session_state.img_shape_processed = img_shape
            st.session_state.valid_indices_processed = valid_indices # Store indices too
        else:
             # If processing fails here, cannot run the experiment
             _placeholder.error("Initial image processing failed. Cannot run data efficiency experiment.")
             return pd.DataFrame(results) # Return empty dataframe
    else:
         _placeholder.success(f"Using {len(objpoints_full)} pre-processed valid views for evaluation.")
         time.sleep(1) # Show message briefly

    _placeholder.empty() # Clear the status message

    # --- Pre-computation Checks ---
    num_valid_views = len(imgpoints_full)
    if num_valid_views < 2:
         st.error(f"Cannot run {method_name} experiment: Need at least 2 valid views, found {num_valid_views}.")
         return pd.DataFrame(results)

    st.write(f"Starting {method_name} data efficiency experiment...")
    st.write(f"- Calibrating using random subsets of images.")
    st.write(f"- Evaluating RMS error using **all {num_valid_views} valid views** for each calibration.")
    st.write(f"- Averaging results over **{num_runs} random run(s)** per subset size.")

    prog_bar = st.progress(0)
    status_text = st.empty()

    # Determine the list of image counts to test (subset sizes)
    if not num_images_list: # If empty, generate default range [2, N]
        num_images_list = list(range(2, num_valid_views + 1))
    # Filter counts to be within the valid range [2, num_valid_views]
    valid_counts_to_test = sorted([n for n in num_images_list if 2 <= n <= num_valid_views])

    if not valid_counts_to_test:
        st.warning(f"No valid image counts between 2 and {num_valid_views} were specified for the {method_name} experiment.")
        prog_bar.empty()
        return pd.DataFrame(results)

    total_steps = len(valid_counts_to_test)
    run_errors_raw = {} # Optional: Store all raw RMS values

    # --- Main Experiment Loop (Iterate through subset sizes) ---
    for i, num_images_subset in enumerate(valid_counts_to_test):
        step_label = f"Testing with {num_images_subset} images ({i+1}/{total_steps})"
        status_text.text(step_label + "...")
        print(f"--- {step_label} ---") # Console log

        errors_for_this_subset_size = [] # Store RMS values for this specific subset size across runs
        successful_calibrations_count = 0
        successful_rms_calculations_count = 0

        # --- Inner Loop (Repeat multiple runs for statistical significance) ---
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs} for {num_images_subset} images...")
            try:
                # 1. Sample a random subset of image indices
                indices_subset = random.sample(range(num_valid_views), num_images_subset)
                # Get the corresponding object and image points for this subset
                obj_subset = [objpoints_full[j] for j in indices_subset]
                img_subset = [imgpoints_full[j] for j in indices_subset]

                # 2. Run the specified calibration function on the SUBSET
                ret_calib, mtx_sub, dist_sub, _, _ = calibrate_func(obj_subset, img_subset, img_shape)
                # We only need K (mtx_sub) and D (dist_sub) from this calibration

                # 3. If calibration on subset succeeded, evaluate on the FULL dataset
                if ret_calib and mtx_sub is not None and dist_sub is not None:
                    successful_calibrations_count += 1
                    current_run_rms = np.inf # Initialize RMS for this run to infinity

                    # 3a. Estimate poses (rvecs, tvecs) for ALL views using the subset's K, D
                    rvecs_full_est = [None] * num_valid_views
                    tvecs_full_est = [None] * num_valid_views
                    poses_estimated_count = 0
                    pose_estimation_failed_for_any_view = False

                    for view_idx in range(num_valid_views):
                        objp_view = objpoints_full[view_idx].astype(np.float32) # Ensure type (N, 3)
                        imgp_view = imgpoints_full[view_idx].astype(np.float32) # Ensure type (N, 1, 2)

                        try:
                            if is_fisheye:
                                # For fisheye evaluation: Undistort points, then use solvePnP with K_sub, D=None
                                if len(objp_view) > 3:
                                    # undistortPoints needs (N,1,2) or (N,2), K, D. Returns (N,1,2)
                                    imgp_undistorted = cv2.fisheye.undistortPoints(imgp_view, mtx_sub, dist_sub, None, mtx_sub)
                                    # solvePnP needs obj(N,3 or Nx1x3), img(N,2 or Nx1x2), K, D=None
                                    ret_pnp, rvec, tvec = cv2.solvePnP(objp_view, imgp_undistorted, mtx_sub, None)
                                else: ret_pnp = False # PnP fails with <=3 points
                            else:
                                # For standard evaluation: Use solvePnP directly with K_sub, D_sub
                                if len(objp_view) > 3:
                                    ret_pnp, rvec, tvec = cv2.solvePnP(objp_view, imgp_view, mtx_sub, dist_sub)
                                else: ret_pnp = False

                            # Store pose if successfully estimated
                            if ret_pnp:
                                rvecs_full_est[view_idx] = rvec
                                tvecs_full_est[view_idx] = tvec
                                poses_estimated_count += 1
                            else:
                                pose_estimation_failed_for_any_view = True
                                # print(f"    [Warning] Pose estimation (solvePnP) failed for view {view_idx} in run {run+1}")

                        except cv2.error as pose_cv_e:
                            # print(f"    [Error] CV2 Pose Est Error (View {view_idx}, Run {run+1}): {pose_cv_e}")
                            pose_estimation_failed_for_any_view = True
                        except Exception as pose_e:
                            # print(f"    [Error] Other Pose Est Error (View {view_idx}, Run {run+1}): {pose_e}")
                            pose_estimation_failed_for_any_view = True
                            # Break inner loop? Or just mark failure? Mark failure for now.
                            # break # Optional: stop checking poses for this run if one fails catastrophically

                    # 3b. Calculate RMS error using FULL dataset points and the estimated poses
                    # Only proceed if poses were estimated successfully for ALL views
                    if not pose_estimation_failed_for_any_view and poses_estimated_count == num_valid_views:
                         current_run_rms = calculate_reprojection_error(
                             objpoints_full, imgpoints_full,
                             mtx_sub, dist_sub, # Use subset's K, D
                             rvecs_full_est, tvecs_full_est, # Use estimated poses for ALL views
                             is_fisheye=is_fisheye
                         )
                         # Store valid RMS value for averaging later
                         if np.isfinite(current_run_rms):
                             errors_for_this_subset_size.append(current_run_rms)
                             successful_rms_calculations_count += 1
                             # print(f"    Run {run+1} RMS: {current_run_rms:.4f}")
                         # else: print(f"    [Warning] Run {run+1} RMS calculation returned non-finite value.")
                    # else: print(f"    [Warning] Skipping RMS calculation for run {run+1} due to pose estimation failure(s).")


                # else: # Calibration itself failed for this subset/run
                    # print(f"  [Warning] Calibration function failed for subset in run {run+1}.")

            except Exception as outer_loop_e:
                # Catch errors during the subset sampling or main calibration call
                st.warning(f"Error during {method_name} experiment run "
                           f"(Images: {num_images_subset}, Run: {run+1}): {outer_loop_e}")
                print(f"  [Error] Outer loop error in run {run+1}: {outer_loop_e}")
                continue # Skip to the next run

        # --- After all runs for a given subset size ---
        run_errors_raw[num_images_subset] = errors_for_this_subset_size # Store raw errors

        # Calculate and store average/std dev if we got valid RMS values
        if errors_for_this_subset_size:
            avg_rms = np.mean(errors_for_this_subset_size)
            std_rms = np.std(errors_for_this_subset_size)
            results['num_images'].append(num_images_subset)
            results['avg_rms'].append(avg_rms)
            results['std_rms'].append(std_rms)
            print(f"  Avg RMS for {num_images_subset} images: {avg_rms:.4f} +/- {std_rms:.4f}")
        elif successful_calibrations_count > 0:
            # Warn if calibration worked sometimes but RMS calculation always failed
             if successful_rms_calculations_count == 0:
                  st.warning(f"For {num_images_subset} images: Calibration succeeded {successful_calibrations_count}/{num_runs} times, "
                             "but RMS calculation failed for all successful runs (check pose estimation steps).")
                  print(f"  [Warning] RMS failed for all successful calibrations with {num_images_subset} images.")
        # else: # Calibration failed for all runs - This case is less informative usually
             # st.warning(f"Calibration failed for all {num_runs} runs using {num_images_subset} images.")

        # Update overall progress bar
        prog_bar.progress((i + 1) / total_steps)

    # --- Experiment End ---
    prog_bar.empty()
    status_text.empty()
    df_results = pd.DataFrame(results)

    if df_results.empty:
        st.warning(f"The {method_name} data efficiency experiment completed but yielded no valid numerical results.")
        # Optional: Display raw errors for debugging if needed
        # st.write("Raw errors per run (for debugging):", run_errors_raw)

    return df_results


# ==============================================================================
# Part 5: Streamlit Application UI
# ==============================================================================

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Camera Calibration Tool")

# --- Initialize Session State (stores data between interactions) ---
# Use a helper function to avoid repetition and ensure keys exist
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Calibration results storage
init_state('std_calib_results', None)    # Results from standard calibration
init_state('fisheye_calib_results', None) # Results from fisheye calibration
init_state('dlt_calib_results', None)    # Results from DLT calibration

# Input data / parameters storage
init_state('calib_images', [])           # List of uploaded file objects (for std/fisheye)
init_state('pattern_params', {'rows': DEFAULT_PATTERN_SIZE[1],
                              'cols': DEFAULT_PATTERN_SIZE[0],
                              'size': DEFAULT_SQUARE_SIZE}) # Checkerboard parameters

# Processed data storage (from checkerboard images)
init_state('objpoints_processed', None) # List of 3D object points arrays
init_state('imgpoints_processed', None) # List of 2D image points arrays
init_state('img_shape_processed', None) # Shape (width, height) of first valid image
init_state('valid_indices_processed', None) # Indices of images where corners were found

# Data efficiency experiment results storage
init_state('eff_img_results_std', None)  # DataFrame for standard experiment
init_state('eff_img_results_fish', None) # DataFrame for fisheye experiment

# DLT file storage (holds the Streamlit UploadedFile objects)
init_state('dlt_mat_file_3d_pts', None) # For rubik_3D_pts.mat (World Points)
init_state('dlt_mat_file_2d_pts', None) # For rubik_2D_pts.mat (Image Points)
init_state('dlt_mat_file_corres', None) # For pt_corres.mat (Camera Points + Image Points)
init_state('dlt_image_file', None) # For rubik_cube.jpg


# ========================
# --- Sidebar UI ---
# ========================
st.sidebar.title("📷 Calibration Methods")

# Radio button to select calibration mode
app_mode = st.sidebar.radio(
    "Choose Calibration Type",
    ["Standard Calibration", "Fisheye Calibration", "DLT Calibration (MAT Files)"],
    key="app_mode_radio", # Key to access the selected value in session state
    help="Select the camera model or calibration method to use."
)
st.sidebar.divider()

# --- Conditional Sidebar Inputs Based on Mode ---

# Checkerboard Inputs (for Standard & Fisheye modes)
if app_mode in ["Standard Calibration", "Fisheye Calibration"]:
    st.sidebar.header("Checkerboard Input")
    st.sidebar.markdown("Define the checkerboard pattern dimensions (inner corners) and square size.")

    # Use pattern_params from session state for sticky values
    current_params = st.session_state.pattern_params
    # Input fields for pattern parameters
    rows_input = st.sidebar.number_input("Inner Corners Height (Rows)", min_value=2, value=current_params['rows'], key="pattern_rows", help="Number of inner corners vertically.")
    cols_input = st.sidebar.number_input("Inner Corners Width (Cols)", min_value=2, value=current_params['cols'], key="pattern_cols", help="Number of inner corners horizontally.")
    size_input = st.sidebar.number_input("Square Size (mm)", min_value=0.01, value=current_params['size'], step=0.1, format="%.2f", key="pattern_size", help="The actual side length of one square on the checkerboard.")

    # --- Handle Parameter Changes ---
    # If any parameter changed, update session state and clear downstream results
    if (rows_input != current_params['rows'] or
        cols_input != current_params['cols'] or
        size_input != current_params['size']):
        st.session_state.pattern_params = {'rows': rows_input, 'cols': cols_input, 'size': size_input}
        # Clear potentially stale processed data and calibration results
        st.session_state.objpoints_processed = None
        st.session_state.imgpoints_processed = None
        st.session_state.img_shape_processed = None
        st.session_state.valid_indices_processed = None
        st.session_state.std_calib_results = None
        st.session_state.fisheye_calib_results = None
        st.session_state.eff_img_results_std = None
        st.session_state.eff_img_results_fish = None
        rerun_if_possible() # Rerun to reflect changes immediately

    # Define pattern size tuple for OpenCV functions (cols, rows)
    pattern_size_cv = (cols_input, rows_input)

    # --- File Uploader for Checkerboard Images ---
    uploaded_files = st.sidebar.file_uploader(
        "Upload Checkerboard Images",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg'], # Allowed file types
        key="calib_uploads", # Session state key
        help="Upload multiple images of the checkerboard taken from different angles and positions."
    )

    # --- Handle File Upload Changes ---
    # Check if the list of uploaded files has changed since the last run
    if uploaded_files is not None and uploaded_files != st.session_state.get('calib_images', []):
         st.session_state.calib_images = uploaded_files # Update the list in session state
         # Clear downstream results as the input images have changed
         st.session_state.objpoints_processed = None
         st.session_state.imgpoints_processed = None
         st.session_state.img_shape_processed = None
         st.session_state.valid_indices_processed = None
         st.session_state.std_calib_results = None
         st.session_state.fisheye_calib_results = None
         st.session_state.eff_img_results_std = None
         st.session_state.eff_img_results_fish = None
         # No rerun needed here, button press will trigger processing.
         # Update the display count immediately though.
         st.sidebar.info(f"{len(st.session_state.calib_images)} image(s) staged for processing.")

    # --- Display Info and Processing Button ---
    st.sidebar.info(f"Pattern: {pattern_size_cv[0]}x{pattern_size_cv[1]} (Cols x Rows), Square Size: {size_input:.2f} mm")
    num_images_in_state = len(st.session_state.get('calib_images', []))
    if num_images_in_state > 0:
        st.sidebar.write(f"{num_images_in_state} image(s) ready.")
    else:
        st.sidebar.warning("Upload checkerboard images to proceed.")

    # Button to trigger corner detection and point extraction
    if st.sidebar.button("Process Checkerboard Images", key="process_images_button",
                         disabled=(num_images_in_state == 0), # Disable if no images uploaded
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

            # Call the image processing function
            objp_all, imgp_all, shape, valid_indices = process_images(
                st.session_state.calib_images,
                pattern_size_cv,
                size_input
            )
            # Store results in session state (even if None, to indicate processing was attempted)
            st.session_state.objpoints_processed = objp_all
            st.session_state.imgpoints_processed = imgp_all
            st.session_state.img_shape_processed = shape
            st.session_state.valid_indices_processed = valid_indices
        # No rerun needed here, Streamlit automatically updates UI based on state changes


# DLT Inputs (for DLT Calibration mode)
elif app_mode == "DLT Calibration (MAT Files)":
    st.sidebar.header("DLT Input Files")
    st.sidebar.markdown("Upload specific `.mat` files and the corresponding image.")
    # Provide more detailed info on expected files and variable names
    st.sidebar.info("""
    **Required Files & Variables:**
    1.  **Correspondences File:** (`pt_corres.mat` expected)
        - `pts_2D`: Image points (Nx2 or 2xN).
        - `cam_pts_3D`: Points in **camera** coordinates (Nx3 or 3xN).
    2.  **3D World Points File:** (`rubik_3D_pts.mat` expected)
        - `pts_3d`: Points in **world** coordinates (Nx3 or 3xN).
    3.  **2D Image Points File:** (`rubik_2D_pts.mat` expected)
        - `pts_2d`: Image points corresponding to `pts_3d` (Nx2 or 2xN).
    4.  **Image File:** (`rubik_cube.jpg` expected)
        - The image corresponding to `pts_2d`.
    *(Code attempts to handle both NxD and DxN formats automatically)*
    """)

    # --- File Uploaders for DLT ---
    # Using consistent naming for session state keys
    mat_corres_f = st.sidebar.file_uploader("1. Correspondences (`pt_corres.mat`)", type=['mat'], key='dlt_mat_corres_up')
    mat_3d_rubik_f = st.sidebar.file_uploader("2. 3D World Points (`rubik_3D_pts.mat`)", type=['mat'], key='dlt_mat3d_rubik_up')
    mat_2d_rubik_f = st.sidebar.file_uploader("3. 2D Image Points (`rubik_2D_pts.mat`)", type=['mat'], key='dlt_mat2d_rubik_up')
    dlt_img_f = st.sidebar.file_uploader("4. Image (`rubik_cube.jpg`)", type=['png', 'jpg', 'jpeg'], key='dlt_img_up')

    # --- Handle DLT File Upload Changes ---
    # Update session state immediately if a file uploader changes. Clear old DLT results.
    # This allows the main area status display to update live without needing a rerun call.
    if mat_corres_f is not None and mat_corres_f != st.session_state.get('dlt_mat_file_corres'):
        st.session_state.dlt_mat_file_corres = mat_corres_f
        st.session_state.dlt_calib_results = None
    if mat_3d_rubik_f is not None and mat_3d_rubik_f != st.session_state.get('dlt_mat_file_3d_pts'):
        st.session_state.dlt_mat_file_3d_pts = mat_3d_rubik_f
        st.session_state.dlt_calib_results = None
    if mat_2d_rubik_f is not None and mat_2d_rubik_f != st.session_state.get('dlt_mat_file_2d_pts'):
        st.session_state.dlt_mat_file_2d_pts = mat_2d_rubik_f
        st.session_state.dlt_calib_results = None
    if dlt_img_f is not None and dlt_img_f != st.session_state.get('dlt_image_file'):
        st.session_state.dlt_image_file = dlt_img_f
        st.session_state.dlt_calib_results = None


# --- Common Sidebar Section ---
st.sidebar.divider()
st.sidebar.markdown("### Resources & Info")
st.sidebar.markdown("""
*   **Checkerboard Samples:** [OpenCV `left*.jpg`](https://github.com/opencv/opencv/tree/4.x/samples/data), [Caltech Dataset](http://www.vision.caltech.edu/bouguetj/calib_doc/)
*   **DLT Files:** Ensure `.mat` files contain the expected variable names. Format (NxD or DxN) should be handled automatically.
*   **Fisheye Tips:** Use images covering the full Field of View (FoV), especially edges/corners, with varied checkerboard poses (tilts, distances).
""")

# ========================
# --- Main Application Area UI ---
# ========================

st.title("📷 Camera Calibration Tool")

# --- Main Area Content Based on Selected Mode ---

# --- Standard Calibration Mode UI ---
if app_mode == "Standard Calibration":
    st.header("Standard Camera Calibration (Pinhole Model)")
    st.markdown("Uses `cv2.calibrateCamera` (Zhang's method). Assumes a standard lens. Requires checkerboard images.")
    st.markdown("---")

    # Check if checkerboard data is processed and ready
    objpoints_ready = st.session_state.get('objpoints_processed')
    imgpoints_ready = st.session_state.get('imgpoints_processed')
    if not objpoints_ready or not imgpoints_ready:
        st.info("⬅️ Please upload and **process** checkerboard images using the sidebar first.")
    else:
        num_valid_std = len(imgpoints_ready)
        st.success(f"✅ Checkerboard data processed for **{num_valid_std}** valid images.")

        # --- Calibration Button ---
        can_calibrate_std = num_valid_std >= 2 # Need at least 2 views
        if st.button("🚀 Run Standard Calibration", key="std_calib_button",
                     help="Perform calibration using the processed checkerboard points.",
                     disabled=(not can_calibrate_std)):

            if not can_calibrate_std: # Redundant check, but safe
                st.error("Need at least 2 valid views for standard calibration.")
            else:
                with st.spinner("Running Standard Calibration..."):
                    # Get processed data from session state
                    objp_all = st.session_state.objpoints_processed
                    imgp_all = st.session_state.imgpoints_processed
                    shape = st.session_state.img_shape_processed

                    # Clear previous results before running new calibration
                    st.session_state.std_calib_results = None
                    st.session_state.fisheye_calib_results = None # Ensure only one result type active

                    # --- Call Calibration Function ---
                    ret, mtx, dist, rvecs, tvecs = run_standard_calibration(objp_all, imgp_all, shape)

                    # --- Store Results if Successful ---
                    if ret:
                         # Calculate RMS error on the calibration data itself
                         rms = calculate_reprojection_error(objp_all, imgp_all, mtx, dist, rvecs, tvecs, is_fisheye=False)
                         st.session_state.std_calib_results = {
                             "K": mtx, "dist": dist, "rms_error": rms,
                             "img_shape": shape, "rvecs": rvecs, "tvecs": tvecs, # Store extrinsics
                             "num_views": num_valid_std
                         }
                         st.success(f"Standard Calibration Successful! RMS Error: {rms:.4f} px")
                         # st.balloons() # REMOVED
                    else:
                         # Error message handled within run_standard_calibration
                         st.session_state.std_calib_results = None # Ensure results are cleared on failure

        # --- Display Standard Calibration Results ---
        std_results = st.session_state.get('std_calib_results')
        if std_results:
            st.divider()
            st.subheader("✅ Standard Calibration Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMS Reprojection Error", f"{std_results['rms_error']:.4f} px",
                          help="Average error (in pixels) between detected and reprojected checkerboard corners on the calibration images.")
                st.write("**Intrinsic Matrix (K):**")
                st.code(f"{std_results['K']}")
            with col2:
                st.write(f"**Image Size (Width x Height):** `{std_results['img_shape'][0]}x{std_results['img_shape'][1]}`")
                st.write(f"**Number of Views Used:** `{std_results['num_views']}`")
                st.write("**Distortion Coefficients (k1, k2, p1, p2, k3):**")
                st.code(f"{std_results['dist'].flatten()}")

            # --- Display Extrinsic Parameters ---
            with st.expander("Extrinsic Parameters (Per View)", expanded=False):
                st.markdown("Rotation (R) and Translation (t) vectors describe the transformation "
                            "from the checkerboard coordinate system to the camera coordinate system for each view.")
                rvecs = std_results.get('rvecs')
                tvecs = std_results.get('tvecs')
                if rvecs is not None and tvecs is not None and len(rvecs) == len(tvecs):
                    # Use columns for better layout if many views
                    ext_cols = st.columns(min(len(rvecs), 3)) # Show up to 3 side-by-side
                    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                        with ext_cols[i % len(ext_cols)]:
                            st.markdown(f"--- \n**View {i+1}**")
                            st.write("**Rotation Vector (rvec):**")
                            st.code(f"{rvec.flatten()}")
                            st.write("**Translation Vector (tvec) (mm):**")
                            st.code(f"{tvec.flatten()}")
                            # Optionally show Rotation Matrix
                            # try:
                            #     R_mat, _ = cv2.Rodrigues(rvec)
                            #     st.write("**Rotation Matrix (R):**")
                            #     st.code(f"{R_mat}")
                            # except Exception as e:
                            #     st.caption(f"Error converting rvec {i+1}: {e}")

                else:
                    st.warning("Extrinsic parameters (rvecs, tvecs) not found in results.")

            # --- Data Efficiency Experiment Section ---
            st.divider()
            st.subheader("📊 Data Efficiency Experiment")
            st.markdown("Analyze how the number of calibration images affects the final RMS reprojection error "
                        "(evaluated on the full dataset).")
            num_views_available = len(st.session_state.get('imgpoints_processed', []))

            if num_views_available < 2:
                st.info("Need at least 2 valid views to run the data efficiency experiment.")
            else:
                # UI for running the experiment
                default_steps = list(range(2, num_views_available + 1))
                st.write(f"Will test using image counts: `{default_steps}`")
                num_runs_per_step = st.slider("Number of Random Runs per Step", min_value=1, max_value=10, value=3, key="eff_runs_std",
                                               help="How many times to randomly select images for each count. More runs yield smoother results but increase computation time.")

                if st.button("Run Standard Efficiency Test", key="eff_run_std"):
                     with st.spinner("Running Efficiency Test... This may take some time."):
                         st.session_state.eff_img_results_std = None # Clear previous results
                         # Retrieve necessary parameters from state
                         current_pattern = (st.session_state.pattern_params['cols'], st.session_state.pattern_params['rows'])
                         current_sq_size = st.session_state.pattern_params['size']
                         current_images = st.session_state.calib_images

                         df_eff = run_data_efficiency_images_experiment(
                             calibrate_func=run_standard_calibration,
                             image_files=current_images,
                             pattern_size_cv=current_pattern,
                             square_size=current_sq_size,
                             num_images_list=default_steps,
                             is_fisheye=False,
                             num_runs=num_runs_per_step
                         )
                         st.session_state.eff_img_results_std = df_eff # Store results

                # Display Experiment Results (Table and Plot) if available
                df_display_std = st.session_state.get('eff_img_results_std')
                if df_display_std is not None:
                     st.markdown("**Experiment Results:**")
                     if not df_display_std.empty:
                         # Show results table
                         df_table = df_display_std[['num_images', 'avg_rms', 'std_rms']].rename(
                             columns={'num_images': '# Images', 'avg_rms': 'Mean RMS', 'std_rms': 'Std Dev RMS'}
                         )
                         df_table['Mean RMS'] = df_table['Mean RMS'].map('{:.4f}'.format)
                         df_table['Std Dev RMS'] = df_table['Std Dev RMS'].map('{:.4f}'.format)
                         st.dataframe(df_table, hide_index=True, use_container_width=True)

                         # Show results plot
                         try:
                             fig_eff, ax_eff = plt.subplots(figsize=(8, 4))
                             ax_eff.plot(df_display_std['num_images'], df_display_std['avg_rms'], marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=5, label='Mean RMS Error')
                             # Add shaded region for standard deviation if available
                             if 'std_rms' in df_display_std.columns and not df_display_std['std_rms'].isnull().all():
                                 ax_eff.fill_between(df_display_std['num_images'],
                                                     df_display_std['avg_rms'] - df_display_std['std_rms'],
                                                     df_display_std['avg_rms'] + df_display_std['std_rms'],
                                                     color='dodgerblue', alpha=0.2, label='±1 Std Dev')

                             # Customize plot appearance
                             valid_rms = df_display_std[np.isfinite(df_display_std['avg_rms'])]['avg_rms']
                             max_error_plot = valid_rms.max() if not valid_rms.empty else 0.5
                             min_error_plot = valid_rms.min() if not valid_rms.empty else 0.0
                             y_top_limit = max(0.3, max_error_plot * 1.2) # Sensible upper limit
                             ax_eff.set_ylim(bottom=0, top=y_top_limit)

                             tested_ticks = sorted(df_display_std['num_images'].unique())
                             ax_eff.set_xticks(tested_ticks)
                             if len(tested_ticks) > 15: # Reduce ticks if too dense
                                  ax_eff.set_xticks(tested_ticks[::max(1, len(tested_ticks)//10)])

                             ax_eff.set_xlabel("Number of Images Used for Calibration")
                             ax_eff.set_ylabel("Mean RMS Error (px) on Full Dataset")
                             ax_eff.set_title("Standard Calibration: Data Efficiency")
                             ax_eff.grid(axis='y', linestyle='--', alpha=0.6)
                             ax_eff.legend()
                             plt.tight_layout()
                             st.pyplot(fig_eff) # Display plot in Streamlit
                         except Exception as plot_e:
                             st.warning(f"Could not generate standard efficiency plot: {plot_e}")
                     # else: No results message handled by experiment function


# --- Fisheye Calibration Mode UI ---
elif app_mode == "Fisheye Calibration":
    st.header("🐠 Fisheye Camera Calibration")
    st.markdown("Uses `cv2.fisheye.calibrate`. Suitable for wide-angle/fisheye lenses. Requires checkerboard images.")
    st.markdown("---")

    # Check if checkerboard data is processed
    objpoints_ready = st.session_state.get('objpoints_processed')
    imgpoints_ready = st.session_state.get('imgpoints_processed')
    if not objpoints_ready or not imgpoints_ready:
        st.info("⬅️ Please upload and **process** checkerboard images using the sidebar first.")
    else:
        num_valid_fish = len(imgpoints_ready)
        st.success(f"✅ Checkerboard data processed for **{num_valid_fish}** valid images.")

        # --- Calibration Button ---
        can_calibrate_fish = num_valid_fish >= 2 # Need at least 2 views
        if st.button("🚀 Run Fisheye Calibration", key="fisheye_calib_button",
                     help="Perform fisheye calibration using processed points.",
                     disabled=(not can_calibrate_fish)):

            if not can_calibrate_fish:
                 st.error("Need at least 2 valid views for fisheye calibration.")
            else:
                with st.spinner("Running Fisheye Calibration..."):
                    # Get processed data
                    objp_all = st.session_state.objpoints_processed
                    imgp_all = st.session_state.imgpoints_processed
                    shape = st.session_state.img_shape_processed

                    # Clear previous results
                    st.session_state.std_calib_results = None
                    st.session_state.fisheye_calib_results = None

                    # --- Call Fisheye Calibration Function ---
                    ret, K, D, rvecs, tvecs = run_fisheye_calibration(objp_all, imgp_all, shape)

                    # --- Store Results if Successful ---
                    if ret:
                        # Calculate RMS error using the fisheye model
                        rms = calculate_reprojection_error(objp_all, imgp_all, K, D, rvecs, tvecs, is_fisheye=True)
                        st.session_state.fisheye_calib_results = {
                            "K": K, "dist": D, "rms_error": rms,
                            "img_shape": shape, "rvecs": rvecs, "tvecs": tvecs, # Store extrinsics
                            "num_views": num_valid_fish
                        }
                        st.success(f"Fisheye Calibration Successful! RMS Error: {rms:.4f} px")
                        # st.balloons() # REMOVED
                    else:
                        # Error handled within run_fisheye_calibration
                        st.session_state.fisheye_calib_results = None # Clear results on failure

        # --- Display Fisheye Calibration Results ---
        fisheye_results = st.session_state.get('fisheye_calib_results')
        if fisheye_results:
            st.divider()
            st.subheader("✅ Fisheye Calibration Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMS Reprojection Error", f"{fisheye_results['rms_error']:.4f} px",
                          help="Average error (pixels) between detected and reprojected corners using the fisheye model.")
                st.write("**Intrinsic Matrix (K):**")
                st.code(f"{fisheye_results['K']}")
            with col2:
                 st.write(f"**Image Size (Width x Height):** `{fisheye_results['img_shape'][0]}x{fisheye_results['img_shape'][1]}`")
                 st.write(f"**Number of Views Used:** `{fisheye_results['num_views']}`")
                 st.write("**Fisheye Distortion (k1, k2, k3, k4):**")
                 st.code(f"{fisheye_results['dist'].flatten()}")

            # --- Display Extrinsic Parameters ---
            with st.expander("Extrinsic Parameters (Per View)", expanded=False):
                st.markdown("Rotation (R) and Translation (t) vectors describe the transformation "
                            "from the checkerboard coordinate system to the camera coordinate system for each view.")
                rvecs = fisheye_results.get('rvecs')
                tvecs = fisheye_results.get('tvecs')
                if rvecs is not None and tvecs is not None and len(rvecs) == len(tvecs):
                    # Use columns for better layout
                    ext_cols = st.columns(min(len(rvecs), 3))
                    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                        with ext_cols[i % len(ext_cols)]:
                            st.markdown(f"--- \n**View {i+1}**")
                            st.write("**Rotation Vector (rvec):**")
                            st.code(f"{rvec.flatten()}")
                            st.write("**Translation Vector (tvec) (mm):**")
                            st.code(f"{tvec.flatten()}")
                            # Optionally show Rotation Matrix
                            # try:
                            #     R_mat, _ = cv2.Rodrigues(rvec)
                            #     st.write("**Rotation Matrix (R):**")
                            #     st.code(f"{R_mat}")
                            # except Exception as e:
                            #     st.caption(f"Error converting rvec {i+1}: {e}")
                else:
                    st.warning("Extrinsic parameters (rvecs, tvecs) not found in results.")


            # --- Fisheye Data Efficiency Experiment Section ---
            st.divider()
            st.subheader("📊 Data Efficiency Experiment (Fisheye)")
            st.markdown("Analyze how the number of calibration images affects the final RMS reprojection error "
                        "(evaluated on the full dataset using the fisheye model).")
            num_views_available = len(st.session_state.get('imgpoints_processed', []))

            if num_views_available < 2:
                st.info("Need at least 2 valid views to run the data efficiency experiment.")
            else:
                # UI for running the experiment
                default_steps = list(range(2, num_views_available + 1))
                st.write(f"Will test using image counts: `{default_steps}`")
                num_runs_per_step = st.slider("Number of Random Runs per Step", min_value=1, max_value=10, value=3, key="eff_runs_fish",
                                               help="How many times to randomly select images for each count.")

                if st.button("Run Fisheye Efficiency Test", key="eff_run_fish"):
                     with st.spinner("Running Fisheye Efficiency Test... This may take some time."):
                         st.session_state.eff_img_results_fish = None # Clear previous results
                         # Retrieve necessary parameters from state
                         current_pattern = (st.session_state.pattern_params['cols'], st.session_state.pattern_params['rows'])
                         current_sq_size = st.session_state.pattern_params['size']
                         current_images = st.session_state.calib_images

                         df_eff_fish = run_data_efficiency_images_experiment(
                             calibrate_func=run_fisheye_calibration, # Pass fisheye function
                             image_files=current_images,
                             pattern_size_cv=current_pattern,
                             square_size=current_sq_size,
                             num_images_list=default_steps,
                             is_fisheye=True, # Specify fisheye model
                             num_runs=num_runs_per_step
                         )
                         st.session_state.eff_img_results_fish = df_eff_fish # Store results

                # Display Fisheye Experiment Results if available
                df_display_fish = st.session_state.get('eff_img_results_fish')
                if df_display_fish is not None:
                     st.markdown("**Experiment Results:**")
                     if not df_display_fish.empty:
                         # Show results table
                         df_table_fish = df_display_fish[['num_images', 'avg_rms', 'std_rms']].rename(
                             columns={'num_images': '# Images', 'avg_rms': 'Mean RMS', 'std_rms': 'Std Dev RMS'}
                         )
                         df_table_fish['Mean RMS'] = df_table_fish['Mean RMS'].map('{:.4f}'.format)
                         df_table_fish['Std Dev RMS'] = df_table_fish['Std Dev RMS'].map('{:.4f}'.format)
                         st.dataframe(df_table_fish, hide_index=True, use_container_width=True)

                         # Show results plot
                         try:
                             fig_eff_fish, ax_eff_fish = plt.subplots(figsize=(8, 4))
                             ax_eff_fish.plot(df_display_fish['num_images'], df_display_fish['avg_rms'], marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=5, label='Mean RMS Error')
                             # Add shaded region for standard deviation if available
                             if 'std_rms' in df_display_fish.columns and not df_display_fish['std_rms'].isnull().all():
                                 ax_eff_fish.fill_between(df_display_fish['num_images'],
                                                     df_display_fish['avg_rms'] - df_display_fish['std_rms'],
                                                     df_display_fish['avg_rms'] + df_display_fish['std_rms'],
                                                     color='mediumseagreen', alpha=0.2, label='±1 Std Dev')

                             # Customize plot appearance
                             valid_rms_f = df_display_fish[np.isfinite(df_display_fish['avg_rms'])]['avg_rms']
                             max_err_plot_f = valid_rms_f.max() if not valid_rms_f.empty else 1.0
                             min_err_plot_f = valid_rms_f.min() if not valid_rms_f.empty else 0.0
                             y_top_limit_f = max(max_err_plot_f * 1.2, min_err_plot_f + 0.1) if np.isfinite(max_err_plot_f) else 1.0
                             ax_eff_fish.set_ylim(bottom=0, top=y_top_limit_f)

                             tested_ticks_f = sorted(df_display_fish['num_images'].unique())
                             ax_eff_fish.set_xticks(tested_ticks_f)
                             if len(tested_ticks_f) > 15: # Reduce ticks
                                  ax_eff_fish.set_xticks(tested_ticks_f[::max(1, len(tested_ticks_f)//10)])

                             ax_eff_fish.set_xlabel("Number of Images Used for Calibration")
                             ax_eff_fish.set_ylabel("Mean RMS Error (px) on Full Dataset")
                             ax_eff_fish.set_title("Fisheye Calibration: Data Efficiency")
                             ax_eff_fish.grid(axis='y', linestyle='--', alpha=0.6)
                             ax_eff_fish.legend()
                             plt.tight_layout()
                             st.pyplot(fig_eff_fish) # Display plot
                         except Exception as plot_e:
                             st.warning(f"Could not generate fisheye efficiency plot: {plot_e}")
                     # else: No results message handled by experiment function


# --- DLT Calibration Mode UI ---
elif app_mode == "DLT Calibration (MAT Files)":
    st.header("📐 Direct Linear Transformation (DLT) Calibration")
    st.markdown("""
    Uses specific `.mat` files containing 3D-2D point correspondences. Assumes a pinhole camera model.
    - **Part 1:** Estimates intrinsics (K) assuming known extrinsics (3D points are in **camera** coordinates). Uses `pt_corres.mat`.
    - **Part 2:** Estimates the full 3x4 projection matrix (P) from **world** 3D points to 2D image points, then decomposes P into K, R, t. Uses `rubik_*.mat` files and the image.
    """)
    st.warning("Ensure correct files with expected variable names are uploaded. Code attempts to handle NxD or DxN formats.")
    st.markdown("---")

    # --- Display Status of Uploaded DLT Files ---
    st.subheader("Uploaded DLT Files Status:")
    # Use columns for a compact layout
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    files_present = {} # Dictionary to track if each required file is present
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

    # Check if all required files have been uploaded
    all_files_ready = all(files_present.values())

    # --- DLT Calibration Button ---
    if st.button("🚀 Run DLT Calibration", key="dlt_calib_button",
                 disabled=(not all_files_ready), # Disable button if files are missing
                 help="Run DLT calibration using the uploaded files."):

        # Clear previous DLT results before starting calculation
        st.session_state.dlt_calib_results = None
        # Initialize a dictionary to store results for this run
        dlt_results_dict = {
                "K_part1": None, "P_part2": None, "K_part2": None,
                "R_part2": None, "t_part2": None, "rms_error_part2": np.inf,
                "reprojected_pts_part2": None, "visualization_fig_part2": None,
                "img_part2": None # Store the loaded image array
            }
        part1_success = False # Flag for Part 1 success
        part2_success = False # Flag for Part 2 success

        # --- Load Data within Button Click ---
        with st.spinner("Running DLT Calibration..."):
            st.write("Loading DLT data...")
            load_status = st.empty() # Placeholder for status messages
            # Load MAT files safely using the utility function
            mat_corres_data = load_mat_file(st.session_state.dlt_mat_file_corres) if files_present['corres'] else None
            mat_3d_data = load_mat_file(st.session_state.dlt_mat_file_3d_pts) if files_present['3d_pts'] else None
            mat_2d_data = load_mat_file(st.session_state.dlt_mat_file_2d_pts) if files_present['2d_pts'] else None
            # Decode the image safely using the utility function
            img_vis_part2 = decode_image(st.session_state.dlt_image_file) if files_present['image'] else None
            load_status.success("Data loading attempt complete.")
            time.sleep(0.5) # Brief pause
            load_status.empty()

            # Store the loaded image array in the results dict if successful
            if img_vis_part2 is not None:
                dlt_results_dict["img_part2"] = img_vis_part2


            # --- Helper function to get points and handle transposition ---
            # This ensures data is in NxD format for the DLT functions below
            def get_points(data_dict, key, expected_dim):
                """Extracts point data, checks shape, and transposes if needed."""
                if data_dict is None or key not in data_dict:
                    print(f"[Warning] Key '{key}' not found in MAT file data.")
                    st.warning(f"Key '{key}' not found in loaded MAT data.")
                    return None
                pts = data_dict[key]
                if not isinstance(pts, np.ndarray) or pts.ndim != 2:
                    st.error(f"Variable '{key}' in MAT file is not a 2D numpy array (found shape={pts.shape}).")
                    return None

                rows, cols = pts.shape
                # Case 1: NxD format (e.g., 28x3) - Correct format
                if cols == expected_dim and rows != expected_dim:
                    print(f"Using '{key}' as is (shape={pts.shape}).")
                    return pts
                # Case 2: DxN format (e.g., 3x28) - Needs transpose
                elif rows == expected_dim and cols != expected_dim:
                    print(f"Transposing '{key}' from {pts.shape} to {(cols, rows)}.")
                    return pts.T
                # Case 3: Ambiguous or incorrect shape
                else:
                    st.error(f"Unexpected shape for '{key}': {pts.shape}. Expected {expected_dim} columns (NxD) or rows (DxN).")
                    return None

            # --- Run Part 1: Intrinsics from Camera Coordinates ---
            st.markdown("**Running DLT Part 1 (Intrinsics from Known Extrinsics)...**")
            if mat_corres_data:
                # Extract points, handling transposition automatically
                cam_pts_3D_p1 = get_points(mat_corres_data, 'cam_pts_3D', 3) # Expect Nx3
                pts_2D_p1 = get_points(mat_corres_data, 'pts_2D', 2)       # Expect Nx2

                if cam_pts_3D_p1 is not None and pts_2D_p1 is not None:
                    st.write(f"Using Part 1 data shapes: cam_pts_3D={cam_pts_3D_p1.shape}, pts_2D={pts_2D_p1.shape}")
                    # Call the DLT intrinsics calibration function
                    K_part1 = dlt_calibrate_intrinsics(pts_2D_p1, cam_pts_3D_p1)

                    if K_part1 is not None:
                         dlt_results_dict["K_part1"] = K_part1 # Store result
                         part1_success = True
                         # st.success("DLT Part 1 calibration successful.") # Keep messages till end?
                    else:
                         st.error("DLT Part 1 calibration calculation failed.")
                else:
                     st.error("Could not extract valid 'cam_pts_3D' (Nx3) or 'pts_2D' (Nx2) from correspondences file for Part 1.")
            else:
                st.error("Failed to load correspondences MAT file for Part 1.")


            # --- Run Part 2: Full Projection Matrix Calibration ---
            st.markdown("**Running DLT Part 2 (Full Calibration from World Coordinates)...**")
            # Check if all necessary data for Part 2 is loaded
            if mat_3d_data and mat_2d_data and img_vis_part2 is not None:
                # Extract points, handling transposition
                pts_3d_world_p2 = get_points(mat_3d_data, 'pts_3d', 3) # Expect Nx3
                pts_2d_image_p2 = get_points(mat_2d_data, 'pts_2d', 2) # Expect Nx2

                if pts_3d_world_p2 is not None and pts_2d_image_p2 is not None:
                     st.write(f"Using Part 2 data shapes: pts_3d={pts_3d_world_p2.shape}, pts_2d={pts_2d_image_p2.shape}")

                     # --- Calculate Projection Matrix P ---
                     P_part2 = dlt_calibrate_projection(pts_2d_image_p2, pts_3d_world_p2)

                     if P_part2 is not None:
                         dlt_results_dict["P_part2"] = P_part2 # Store P
                         # st.success("Projection Matrix (P) calculated.")

                         # --- Decompose P into K, R, t ---
                         K_p2, R_p2, t_p2 = dlt_P_to_KRt(P_part2)

                         if K_p2 is not None and R_p2 is not None and t_p2 is not None:
                              # Store decomposed results
                              dlt_results_dict["K_part2"] = K_p2
                              dlt_results_dict["R_part2"] = R_p2
                              dlt_results_dict["t_part2"] = t_p2
                              # st.success("Decomposition into K, R, t successful.")

                              # --- Calculate Reprojection Error & Visualize ---
                              # Pass Nx3, Nx2 points to error function
                              rms_err_p2, x_proj_p2 = dlt_compute_reprojection_error(P_part2, pts_3d_world_p2, pts_2d_image_p2)
                              # Pass image array and Nx2 points to visualization
                              vis_fig_p2 = dlt_visualize_calibration(img_vis_part2, pts_2d_image_p2, x_proj_p2)

                              # Store error and visualization results
                              if np.isfinite(rms_err_p2):
                                  dlt_results_dict["rms_error_part2"] = rms_err_p2
                                  dlt_results_dict["reprojected_pts_part2"] = x_proj_p2 # Store reprojected points too
                                  # st.success(f"Reprojection RMS Error: {rms_err_p2:.4f} px")
                              else:
                                  st.warning("Could not calculate valid reprojection error for Part 2.")

                              if vis_fig_p2:
                                  dlt_results_dict["visualization_fig_part2"] = vis_fig_p2 # Store the figure object
                                  # st.success("Visualization generated.")
                              else:
                                  st.warning("Could not generate visualization for Part 2.")

                              part2_success = True # Mark Part 2 as fully successful

                         else: # Decomposition failed
                             st.error("Failed to decompose P into K, R, t.")
                     else: # P calculation failed
                         st.error("Failed to calculate Projection Matrix (P) for Part 2.")
                else: # Point extraction failed
                     st.error("Could not extract valid 'pts_3d' (Nx3) or 'pts_2d' (Nx2) from MAT files for Part 2.")
            else: # Data loading failed
                st.error("Failed to load required MAT files or image for Part 2.")

            # --- Final Step: Store results in session state if anything succeeded ---
            if part1_success or part2_success:
                st.session_state.dlt_calib_results = dlt_results_dict
                st.success("DLT Calibration Processing Complete.")
                # st.balloons() # REMOVED
            else:
                st.session_state.dlt_calib_results = None # Ensure state is clear if both parts failed
                st.error("DLT Calibration Failed.")


    # --- Display DLT Results (Checks session state *after* potential button run) ---
    dlt_results = st.session_state.get('dlt_calib_results')
    if dlt_results: # Only display if results exist in session state
        st.divider()
        st.subheader("✅ DLT Calibration Results")

        # Optional Debug Output (can be removed for cleaner final version)
        # with st.expander("DEBUG: Raw DLT Results Dictionary"):
        #      st.write(dlt_results)

        # --- Display Part 1 Results ---
        st.markdown("**Part 1 Results (Intrinsics from Known Extrinsics)**")
        K1_res = dlt_results.get("K_part1")
        if K1_res is not None:
            st.write("Estimated Intrinsic Matrix (K):")
            st.code(f"{K1_res}")
        else:
            st.info("Part 1 calibration was not successful or data was missing/invalid.")

        st.markdown("---") # Separator

        # --- Display Part 2 Results ---
        st.markdown("**Part 2 Results (Full Calibration from World Coordinates)**")
        P2_res = dlt_results.get("P_part2")
        if P2_res is not None: # Check if Part 2 calculation produced P matrix
            col_p2_1, col_p2_2 = st.columns(2) # Use columns for layout
            with col_p2_1:
                # Display RMS error
                rms_val = dlt_results.get('rms_error_part2', np.inf)
                if np.isfinite(rms_val):
                     st.metric("RMS Reprojection Error", f"{rms_val:.4f} px", help="Average error between original 2D points and points reprojected using the calculated P matrix.")
                else:
                     st.metric("RMS Reprojection Error", "N/A")

                st.write("Full Projection Matrix (P):")
                st.code(f"{P2_res}")

                # Display Decomposed K
                K2_res = dlt_results.get("K_part2")
                if K2_res is not None:
                    st.write("Decomposed Intrinsic Matrix (K):")
                    st.code(f"{K2_res}")
                else:
                    st.write("Intrinsic Matrix (K): Decomposition failed.")

            with col_p2_2:
                # Display Decomposed R
                R2_res = dlt_results.get("R_part2")
                if R2_res is not None:
                    st.write("Decomposed Rotation Matrix (R):")
                    st.code(f"{R2_res}")
                    # Show determinant check for rotation validity
                    try:
                        det_val = det(R2_res)
                        st.caption(f"(det(R) ≈ {det_val:.4f})") # Should be close to +1
                    except:
                        st.caption("(Could not compute det(R))")
                else:
                     st.write("Rotation Matrix (R): Decomposition failed.")

                # Display Decomposed t
                t2_res = dlt_results.get("t_part2")
                if t2_res is not None:
                    st.write("Decomposed Translation Vector (t):")
                    st.code(f"{t2_res}")
                else:
                    st.write("Translation Vector (t): Decomposition failed or K was singular.")

            # --- Display Part 2 Visualization ---
            st.markdown("**Part 2 Visualization**")
            fig_to_plot = dlt_results.get('visualization_fig_part2') # Get the figure object
            if fig_to_plot:
                st.pyplot(fig_to_plot) # Display the Matplotlib figure
            elif dlt_results.get('img_part2') is not None:
                # Fallback: Show the original image if plot failed but image exists
                st.warning("Visualization plot could not be generated (check for reprojection/plotting errors), showing original image.")
                try:
                    # Convert BGR (OpenCV) to RGB for display
                    img_rgb = cv2.cvtColor(dlt_results['img_part2'], cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="DLT Input Image")
                except Exception as img_e:
                    st.error(f"Failed to display fallback image: {img_e}")
            else:
                # If both plot and image are missing
                st.warning("Visualization could not be generated (missing image or plot error).")

        else: # If P_part2 was None (Part 2 failed early)
            st.info("Part 2 calibration was not successful or required data was missing/invalid.")

# --- End of DLT Mode UI ---
