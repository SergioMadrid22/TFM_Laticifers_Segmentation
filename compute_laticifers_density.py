import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from glob import glob

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute laticifer density by counting intersections with transect lines in a specified direction."
    )
    parser.add_argument(
        '-i', '--input_folder',
        required=True,
        help="Path to the folder containing binary segmentation masks (PNG, TIF, etc.)."
    )
    parser.add_argument(
        '-o', '--output_folder',
        required=True,
        help="Path to the folder where visual outputs and the CSV report will be saved."
    )
    parser.add_argument(
        '-n', '--num_lines',
        type=int,
        default=10,
        help="Number of transect lines to draw."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        '-d', '--direction',
        type=str,
        default='both',
        choices=['horizontal', 'vertical', 'both'],
        help="Direction of the transect lines ('horizontal', 'vertical', or 'both'). Default is 'both'."
    )
    return parser.parse_args()

def analyze_density(mask_path, num_lines, direction='both'):
    """
    Analyzes a single binary mask to compute laticifer density in a specified direction.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
        
    mask_bin = (mask > 127).astype(np.uint8)
    H, W = mask_bin.shape

    vis_image = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
    all_intersections = []
    horizontal_intersections = []
    vertical_intersections = []
    
    # 1. Analyze HORIZONTAL transects if requested
    if direction in ['horizontal', 'both']:
        line_positions_h = np.linspace(0.1 * H, 0.9 * H, num_lines, dtype=int)
        for y in line_positions_h:
            cv2.line(vis_image, (0, y), (W, y), (0, 255, 0), 2)
            line_pixels = mask_bin[y, :]
            diffs = np.diff(line_pixels.astype(np.int8))
            num_intersections = np.count_nonzero(diffs == 1)
            horizontal_intersections.append(num_intersections)
            
            intersection_points_x = np.where(diffs == 1)[0] + 1
            for x_coord in intersection_points_x:
                cv2.circle(vis_image, (x_coord, y), 5, (0, 0, 255), -1)

    # 2. Analyze VERTICAL transects if requested
    if direction in ['vertical', 'both']:
        line_positions_v = np.linspace(0.1 * W, 0.9 * W, num_lines, dtype=int)
        for x in line_positions_v:
            cv2.line(vis_image, (x, 0), (x, H), (255, 0, 0), 2)
            line_pixels = mask_bin[:, x]
            diffs = np.diff(line_pixels.astype(np.int8))
            num_intersections = np.count_nonzero(diffs == 1)
            vertical_intersections.append(num_intersections)

            intersection_points_y = np.where(diffs == 1)[0] + 1
            for y_coord in intersection_points_y:
                cv2.circle(vis_image, (x, y_coord), 5, (0, 0, 255), -1)
    
    # Combine results for overall stats
    all_intersections = horizontal_intersections + vertical_intersections

    # 3. Calculate statistics
    stats = {'filename': os.path.basename(mask_path)}
    
    if all_intersections:
        stats['mean_intersections_per_line'] = np.mean(all_intersections)
        stats['std_intersections_per_line'] = np.std(all_intersections)
    
    if horizontal_intersections:
        stats['mean_horizontal_intersections'] = np.mean(horizontal_intersections)
            
    if vertical_intersections:
        stats['mean_vertical_intersections'] = np.mean(vertical_intersections)
    
    return stats, vis_image

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    
    mask_files = glob(os.path.join(args.input_folder, '*.*'))
    if not mask_files:
        print(f"Error: No image files found in {args.input_folder}")
        return

    all_stats = []
    for mask_path in tqdm(mask_files, desc="Analyzing Images"):
        stats, vis_image = analyze_density(mask_path, args.num_lines, args.direction)
        
        if stats:
            all_stats.append(stats)
            base_name = os.path.splitext(stats['filename'])[0]
            save_path = os.path.join(args.output_folder, f"{base_name}_density_{args.direction}.png")
            cv2.imwrite(save_path, vis_image)

    if all_stats:
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(args.output_folder, f"laticifer_density_report_{args.direction}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nAnalysis complete. Report saved to: {csv_path}")
        print("\nSummary Statistics:")
        print(df.describe())

if __name__ == '__main__':
    main()