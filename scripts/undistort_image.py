#!/usr/bin/env python3
import argparse
import os
import sys
import cv2

def load_intrinsics(xml_path):
    """Load OpenCV camera matrix and distortion coefficients from XML storage file."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Intrinsics file not found: {xml_path}")
        
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics file: {xml_path}")
    
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    
    fisheye_node = fs.getNode("fisheye_model")
    use_fisheye = int(fisheye_node.real()) if not fisheye_node.empty() else 0
    fs.release()
    
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"Invalid intrinsics data in XML file: {xml_path}")

    return camera_matrix, dist_coeffs, use_fisheye

def undistort_image(image_path, intrinsics_path, output_path=None):
    """Undistort an image using specified intrinsic parameters."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read input image: {image_path}")

    camera_matrix, dist_coeffs, use_fisheye = load_intrinsics(intrinsics_path)

    h, w = image.shape[:2]

    if use_fisheye:
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), None, balance=1.0
        )
        undistorted = cv2.fisheye.undistortImage(
            image, camera_matrix, dist_coeffs, Knew=new_camera_matrix
        )
    else:
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_undistorted{ext}"

    cv2.imwrite(output_path, undistorted)
    print(f"[Success] Undistorted image saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Undistort a single image using camera intrinsic calibration XML."
    )
    parser.add_argument(
        "--image", "-i", required=True, help="Path to input image to undistort"
    )
    parser.add_argument(
        "--intrinsics",
        "-k",
        default="assets/calibration/intrinsic/emeet/emeet_intrinsics.xml",
        help="Path to camera intrinsics XML file (default: EMeet intrinsics)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save output undistorted image (default: <input>_undistorted.jpg)",
    )

    args = parser.parse_args()

    try:
        undistort_image(args.image, args.intrinsics, args.output)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
