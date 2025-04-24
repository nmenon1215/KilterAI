"""
Improved Kilter Board Data Collection Script

This script is specifically modified to handle cases where grade data isn't directly available
or mappable from the database. It uses the display_difficulty values directly
and converts them to corresponding V-grades.

Usage:
    python KilterBoardDataCollection.py --database kilter.db --output dataset/ --skip-sync
"""

import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
import json
import subprocess
from pathlib import Path

def setup_boardlib(board_name='kilter', db_path='kilter.db'):
    """
    Download and synchronize the board database if not already present.

    Args:
        board_name: The name of the climbing board (default: kilter)
        db_path: Path to save the database file
    """
    if not os.path.exists(db_path):
        print(f"Downloading {board_name} database to {db_path}...")
        subprocess.run(['boardlib', 'database', board_name, db_path])
    else:
        print(f"Database already exists at {db_path}. Syncing latest data...")
        subprocess.run(['boardlib', 'database', board_name, db_path])

    return db_path

def map_difficulty_to_grade(difficulty):
    """
    Map the numerical difficulty value to boulder grades (V-scale)
    based on the difficulty_grades table.

    Args:
        difficulty: Numerical difficulty value

    Returns:
        Boulder grade string (e.g., 'V3')
    """
    if pd.isna(difficulty):
        return None

    difficulty = int(difficulty)

    # Mapping based on the provided difficulty_grades table
    if difficulty <= 12:  # 1-12 are all V0
        return "V0"
    elif difficulty <= 14:  # 13-14 are V1
        return "V1"
    elif difficulty <= 15:  # 15 is V2
        return "V2"
    elif difficulty <= 17:  # 16-17 are V3
        return "V3"
    elif difficulty <= 19:  # 18-19 are V4
        return "V4"
    elif difficulty <= 21:  # 20-21 are V5
        return "V5"
    elif difficulty <= 22:  # 22 is V6
        return "V6"
    elif difficulty <= 23:  # 23 is V7
        return "V7"
    elif difficulty <= 25:  # 24-25 are V8
        return "V8"
    elif difficulty <= 26:  # 26 is V9
        return "V9"
    elif difficulty <= 27:  # 27 is V10
        return "V10"
    elif difficulty <= 28:  # 28 is V11
        return "V11"
    elif difficulty <= 29:  # 29 is V12
        return "V12"
    elif difficulty <= 30:  # 30 is V13
        return "V13"
    elif difficulty <= 31:  # 31 is V14
        return "V14"
    elif difficulty <= 32:  # 32 is V15
        return "V15"
    elif difficulty <= 33:  # 33 is V16
        return "V16"
    elif difficulty <= 34:  # 34 is V17
        return "V17"
    elif difficulty <= 35:  # 35 is V18
        return "V18"
    elif difficulty <= 36:  # 36 is V19
        return "V19"
    elif difficulty <= 37:  # 37 is V20
        return "V20"
    elif difficulty <= 38:  # 38 is V21
        return "V21"
    else:  # 39+ is V22
        return "V22"

def extract_climbs_data(db_path):
    """
    Extract climb data from the SQLite database using the actual schema.

    Args:
        db_path: Path to the database file

    Returns:
        DataFrames containing climbs data, holds data, and board dimensions
    """
    print(f"Extracting data from {db_path}...")
    conn = sqlite3.connect(db_path)

    # Query to get basic climb data without requiring a join to difficulty_grades
    query_climbs = """
    SELECT 
        c.uuid AS id,
        c.name,
        c.setter_username,
        c.angle,
        c.is_listed,
        c.layout_id,
        c.hsm,
        c.frames,
        CASE WHEN c.hsm = 3 THEN 1 ELSE 0 END AS is_benchmark
    FROM 
        climbs c
    WHERE 
        c.frames IS NOT NULL
    LIMIT 20000  -- Limit to a reasonable number of climbs for now
    """

    try:
        climbs_df = pd.read_sql_query(query_climbs, conn)
        print(f"Found {len(climbs_df)} climbs with frames data")

        # Get difficulty information from climb_stats
        # We have to do this separately because of the structure
        query_difficulties = """
        SELECT 
            climb_uuid AS id,
            angle,
            display_difficulty
        FROM 
            climb_stats
        """

        difficulties_df = pd.read_sql_query(query_difficulties, conn)
        print(f"Found {len(difficulties_df)} climb stats entries")

        # Filter out potential duplicates (same climb with different angles)
        # Keep the entry with the highest difficulty for each climb
        difficulties_df = (difficulties_df
            .sort_values(['id', 'display_difficulty'], ascending=[True, False])
            .drop_duplicates(subset=['id'], keep='first')
        )

        # Merge the difficulty data with the climb data
        climbs_df = pd.merge(
            climbs_df,
            difficulties_df[['id', 'display_difficulty']],
            on='id',
            how='left'
        )

        # Map difficulty to grade
        climbs_df['grade'] = climbs_df['display_difficulty'].apply(map_difficulty_to_grade)

        # Fill missing grades with a default value if display_difficulty is present
        # This is just to ensure we have some workable data even if grade mapping isn't perfect
        for i, row in climbs_df.iterrows():
            if pd.isna(row['grade']) and not pd.isna(row['display_difficulty']):
                difficulty_value = row['display_difficulty']
                # Simple linear mapping: difficulty/3 roughly corresponds to V-grade
                v_grade = max(0, int(difficulty_value / 3))
                climbs_df.at[i, 'grade'] = f"V{v_grade}"

        print(f"Mapped {climbs_df['grade'].notna().sum()} climbs to grades")

    except Exception as e:
        print(f"Error querying climbs: {e}")
        climbs_df = pd.DataFrame()

    # Extract hold data from the frames field in the climbs table
    holds_data = []

    if not climbs_df.empty:
        # Query to get placements data
        query_placements = """
        SELECT 
            p.id AS placement_id,
            p.hole_id,
            h.x,
            h.y,
            pr.name AS role_name
        FROM 
            placements p
        JOIN 
            holes h ON p.hole_id = h.id
        LEFT JOIN 
            placement_roles pr ON p.default_placement_role_id = pr.id
        """

        try:
            placements_df = pd.read_sql_query(query_placements, conn)
            print(f"Found {len(placements_df)} placements")

            # Create a mapping of hole_id to x,y coordinates
            hole_coords = {}
            for _, row in placements_df.iterrows():
                hole_coords[row['hole_id']] = (row['x'], row['y'])

            # Function to parse frames data
            def parse_frames(frames_str):
                if pd.isna(frames_str) or not frames_str:
                    return []

                # Format is typically: pXXXXrYY where XXXX is hole_id and YY is role
                holds = []
                parts = frames_str.split('p')

                for part in parts[1:]:  # Skip the first empty part
                    if 'r' in part:
                        hole_id_str, role_str = part.split('r', 1)
                        try:
                            hole_id = int(hole_id_str)
                            role_id = int(role_str[:2])  # Take first two chars as role id

                            # Map role_id to role name
                            role_name = "middle"  # Default
                            if role_id == 12:
                                role_name = "start"
                            elif role_id == 13:
                                role_name = "middle"
                            elif role_id == 15:
                                role_name = "finish"

                            if hole_id in hole_coords:
                                x, y = hole_coords[hole_id]
                                holds.append({
                                    'hole_id': hole_id,
                                    'x': x,
                                    'y': y,
                                    'is_start': role_name == 'start',
                                    'is_finish': role_name == 'finish'
                                })
                        except (ValueError, IndexError):
                            continue

                return holds

            # Process each climb's frames to extract holds
            for _, climb in climbs_df.iterrows():
                climb_id = climb['id']
                holds = parse_frames(climb['frames'])

                for hold in holds:
                    holds_data.append({
                        'climb_id': climb_id,
                        'hole_id': hold['hole_id'],
                        'x': hold['x'],
                        'y': hold['y'],
                        'is_start': hold['is_start'],
                        'is_finish': hold['is_finish']
                    })

            holds_df = pd.DataFrame(holds_data)
            print(f"Extracted {len(holds_df)} holds from frames data")

        except Exception as e:
            print(f"Error processing placements or frames: {e}")
            holds_df = pd.DataFrame(holds_data)
    else:
        holds_df = pd.DataFrame(holds_data)

    # Get board dimensions from holes table
    query_board = """
    SELECT 
        MIN(x) as min_x, 
        MAX(x) as max_x, 
        MIN(y) as min_y, 
        MAX(y) as max_y
    FROM 
        holes
    """

    try:
        board_dimensions = pd.read_sql_query(query_board, conn).iloc[0]
        print(f"Board dimensions: X: {board_dimensions['min_x']} to {board_dimensions['max_x']}, "
              f"Y: {board_dimensions['min_y']} to {board_dimensions['max_y']}")
    except Exception as e:
        print(f"Error getting board dimensions: {e}")
        board_dimensions = pd.Series({
            'min_x': 0, 'max_x': 1000, 'min_y': 0, 'max_y': 1000
        })

    conn.close()

    # Remove the frames column as it's no longer needed
    if 'frames' in climbs_df.columns:
        climbs_df = climbs_df.drop(columns=['frames'])

    return climbs_df, holds_df, board_dimensions

def process_data_for_ml(climbs_df, holds_df, board_dimensions, output_dir='dataset'):
    """
    Process the extracted data and create features for machine learning.

    Args:
        climbs_df: DataFrame containing climbs data
        holds_df: DataFrame containing holds data
        board_dimensions: Series with board dimensions
        output_dir: Directory to save processed data

    Returns:
        DataFrame with features ready for machine learning
    """
    print("Processing data for machine learning...")
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have valid data
    if climbs_df.empty:
        print("Error: No climb data available. Cannot process for ML.")
        return pd.DataFrame()

    # Save raw data
    climbs_df.to_csv(os.path.join(output_dir, 'climbs.csv'), index=False)
    if not holds_df.empty:
        holds_df.to_csv(os.path.join(output_dir, 'holds.csv'), index=False)

    # Process grades to numerical values
    def extract_grade_value(grade_str):
        if pd.isna(grade_str) or not isinstance(grade_str, str):
            return np.nan
        # Extract the number after 'V'
        try:
            if grade_str.startswith('V'):
                if '+' in grade_str:  # Handle 'V11+'
                    return float(grade_str[1:].replace('+', ''))
                return float(grade_str[1:].split('/')[0])  # Handle cases like 'V1/2'
            return float(grade_str)  # Try direct conversion if no 'V'
        except (ValueError, IndexError):
            return np.nan

    # Add grade_value column if 'grade' exists
    if 'grade' in climbs_df.columns:
        climbs_df['grade_value'] = climbs_df['grade'].apply(extract_grade_value)
    elif 'display_difficulty' in climbs_df.columns:
        # Map display_difficulty directly to grade_value (scaled down)
        climbs_df['grade_value'] = climbs_df['display_difficulty'] / 3.0
    else:
        climbs_df['grade_value'] = np.nan

    # Filter out climbs with missing grade values if needed
    if 'grade_value' in climbs_df.columns:
        valid_climbs = climbs_df.dropna(subset=['grade_value']).copy()
        print(f"Keeping {len(valid_climbs)} climbs with valid grade values")
    else:
        valid_climbs = climbs_df.copy()
        print("Warning: No grade values available. Using all climbs.")

    # If we still don't have any valid climbs, use the display_difficulty directly
    if len(valid_climbs) == 0 and 'display_difficulty' in climbs_df.columns:
        print("No valid grade values found. Using display_difficulty directly.")
        climbs_df['grade_value'] = climbs_df['display_difficulty'] / 3.0
        valid_climbs = climbs_df.dropna(subset=['display_difficulty']).copy()
        print(f"Using {len(valid_climbs)} climbs with display_difficulty values")

    # Create features for each climb
    ml_features = []

    # Calculate normalized dimensions for the grid
    width = board_dimensions['max_x'] - board_dimensions['min_x']
    height = board_dimensions['max_y'] - board_dimensions['min_y']

    # Define grid size for hold density features
    grid_size = 12  # 12x12 grid

    # Process each climb
    for _, climb_row in valid_climbs.iterrows():
        # Get holds for this climb
        climb_id = climb_row['id']
        climb_holds = holds_df[holds_df['climb_id'] == climb_id]

        # Skip climbs with no hold data
        if len(climb_holds) == 0:
            continue

        # Basic features
        features = {
            'climb_id': climb_id,
            'name': climb_row['name'],
            'setter_username': climb_row.get('setter_username', ''),
            'angle': climb_row.get('angle', 0),
            'is_listed': int(climb_row.get('is_listed', 0)),
            'is_benchmark': int(climb_row.get('is_benchmark', 0)),
            'display_difficulty': climb_row.get('display_difficulty', np.nan),
        }

        # Add grade information if available
        if 'grade' in climb_row and not pd.isna(climb_row['grade']):
            features['grade'] = climb_row['grade']
        elif 'display_difficulty' in climb_row and not pd.isna(climb_row['display_difficulty']):
            # Create a grade based on display_difficulty if not available
            difficulty = climb_row['display_difficulty']
            v_grade = max(0, int(difficulty / 3))
            features['grade'] = f"V{v_grade}"

        if 'grade_value' in climb_row and not pd.isna(climb_row['grade_value']):
            features['grade_value'] = climb_row['grade_value']
        elif 'display_difficulty' in climb_row and not pd.isna(climb_row['display_difficulty']):
            # Create a grade_value based on display_difficulty if not available
            features['grade_value'] = climb_row['display_difficulty'] / 3.0

        # Add hold-related features
        features['num_holds'] = len(climb_holds)
        features['num_start_holds'] = sum(climb_holds['is_start']) if 'is_start' in climb_holds.columns else 0
        features['num_finish_holds'] = sum(climb_holds['is_finish']) if 'is_finish' in climb_holds.columns else 0

        # Create binary grid representation (1 if hold present, 0 if not)
        grid = np.zeros((grid_size, grid_size))

        if 'x' in climb_holds.columns and 'y' in climb_holds.columns:
            for _, hold in climb_holds.iterrows():
                # Normalize coordinates to grid
                norm_x = (hold['x'] - board_dimensions['min_x']) / width
                norm_y = (hold['y'] - board_dimensions['min_y']) / height

                # Convert to grid indices
                grid_x = min(int(norm_x * grid_size), grid_size - 1)
                grid_y = min(int(norm_y * grid_size), grid_size - 1)

                # Mark hold position in grid
                grid[grid_y, grid_x] = 1

            # Add grid cells as features
            for i in range(grid_size):
                for j in range(grid_size):
                    features[f'grid_{i}_{j}'] = grid[i, j]

            # Calculate hold density in regions
            features['density_bottom'] = np.sum(grid[grid_size//2:, :]) / (grid_size * grid_size/2)
            features['density_top'] = np.sum(grid[:grid_size//2, :]) / (grid_size * grid_size/2)
            features['density_left'] = np.sum(grid[:, :grid_size//2]) / (grid_size * grid_size/2)
            features['density_right'] = np.sum(grid[:, grid_size//2:]) / (grid_size * grid_size/2)

            # Calculate distances between holds
            if len(climb_holds) > 1:
                hold_coords = climb_holds[['x', 'y']].values

                # Calculate all pairwise distances
                distances = []
                for i in range(len(hold_coords)):
                    for j in range(i+1, len(hold_coords)):
                        dist = np.sqrt(((hold_coords[i] - hold_coords[j])**2).sum())
                        distances.append(dist)

                features['avg_distance'] = np.mean(distances)
                features['max_distance'] = np.max(distances)
                features['min_distance'] = np.min(distances)
                features['std_distance'] = np.std(distances)
            else:
                features['avg_distance'] = 0
                features['max_distance'] = 0
                features['min_distance'] = 0
                features['std_distance'] = 0

        ml_features.append(features)

    # Create DataFrame from features
    ml_df = pd.DataFrame(ml_features)

    if ml_df.empty:
        print("Warning: No features could be created.")
        return ml_df

    # Save processed data
    ml_df.to_csv(os.path.join(output_dir, 'ml_features.csv'), index=False)

    # Save a simple version with just the essential features
    essential_cols = ['climb_id', 'name', 'grade', 'grade_value', 'display_difficulty', 'angle',
                     'is_benchmark', 'num_holds', 'num_start_holds', 'num_finish_holds']

    # Add optional columns if they exist
    for col in ['density_bottom', 'density_top', 'density_left', 'density_right',
               'avg_distance', 'max_distance', 'min_distance', 'std_distance']:
        if col in ml_df.columns:
            essential_cols.append(col)

    # Check which essential columns exist
    existing_cols = [col for col in essential_cols if col in ml_df.columns]

    if existing_cols:
        ml_df[existing_cols].to_csv(os.path.join(output_dir, 'essential_features.csv'), index=False)

    print(f"Processed data saved to {output_dir}")
    print(f"Created {len(ml_df)} samples with {len(ml_df.columns)} features")

    return ml_df

def create_visualization_data(climbs_df, holds_df, output_dir='dataset'):
    """
    Create data specifically for visualization purposes.

    Args:
        climbs_df: DataFrame containing climbs data
        holds_df: DataFrame containing holds data
        output_dir: Directory to save the data
    """
    print("Creating visualization data...")

    if climbs_df.empty:
        print("Error: No climb data available. Cannot create visualization data.")
        return

    # Check if we have grade data
    if 'grade' not in climbs_df.columns and 'grade_value' not in climbs_df.columns and 'display_difficulty' not in climbs_df.columns:
        print("Warning: No grade data available. Cannot create visualization data by grade.")
        return

    # Use grade column if available, otherwise create it from other fields
    if 'grade' not in climbs_df.columns:
        if 'grade_value' in climbs_df.columns:
            climbs_df['grade'] = climbs_df['grade_value'].apply(lambda x: f"V{int(x)}" if not pd.isna(x) else None)
        elif 'display_difficulty' in climbs_df.columns:
            climbs_df['grade'] = climbs_df['display_difficulty'].apply(
                lambda x: f"V{int(x/3)}" if not pd.isna(x) else None
            )

    # Group climbs by grade
    vis_data = {}

    # Get unique grades that aren't null
    grades = climbs_df['grade'].dropna().unique()

    if len(grades) == 0:
        print("No valid grades found for visualization. Creating generic categories.")
        # Create difficulty categories based on display_difficulty
        if 'display_difficulty' in climbs_df.columns:
            categories = {
                'Easy': (0, 15),
                'Medium': (16, 24),
                'Hard': (25, float('inf'))
            }

            for category, (min_diff, max_diff) in categories.items():
                category_climbs = climbs_df[
                    (climbs_df['display_difficulty'] >= min_diff) &
                    (climbs_df['display_difficulty'] < max_diff)
                ]

                if len(category_climbs) > 0:
                    samples = category_climbs.sample(min(5, len(category_climbs)))
                    vis_data[category] = []

                    for _, climb in samples.iterrows():
                        process_climb_for_visualization(climb, holds_df, vis_data[category])

        # If no suitable categorization is possible, just use a single category
        if len(vis_data) == 0:
            vis_data['All Climbs'] = []
            samples = climbs_df.sample(min(10, len(climbs_df)))

            for _, climb in samples.iterrows():
                process_climb_for_visualization(climb, holds_df, vis_data['All Climbs'])
    else:
        for grade in grades:
            grade_climbs = climbs_df[climbs_df['grade'] == grade]
            # If we have too many climbs, just take a sample
            if len(grade_climbs) > 5:
                samples = grade_climbs.sample(5)
            else:
                samples = grade_climbs

            vis_data[grade] = []

            for _, climb in samples.iterrows():
                process_climb_for_visualization(climb, holds_df, vis_data[grade])

    # Save visualization data
    with open(os.path.join(output_dir, 'visualization_data.json'), 'w') as f:
        json.dump(vis_data, f, indent=2)

    print(f"Visualization data saved to {os.path.join(output_dir, 'visualization_data.json')}")

def process_climb_for_visualization(climb, holds_df, vis_data_list):
    """Helper function to process a climb for visualization data"""
    # Get holds for this climb
    climb_id = climb['id']
    climb_holds = holds_df[holds_df['climb_id'] == climb_id]

    if len(climb_holds) == 0:
        return

    # Create hold list
    hold_list = []
    for _, hold in climb_holds.iterrows():
        hold_data = {
            'x': float(hold['x']),
            'y': float(hold['y']),
            'is_start': bool(hold['is_start']) if 'is_start' in hold else False,
            'is_finish': bool(hold['is_finish']) if 'is_finish' in hold else False
        }
        hold_list.append(hold_data)

    # Create climb data
    climb_data = {
        'id': climb_id,
        'name': climb['name'],
        'holds': hold_list,
        'angle': climb.get('angle', None),
        'difficulty': climb.get('display_difficulty', None)
    }

    vis_data_list.append(climb_data)

def main():
    parser = argparse.ArgumentParser(description='Collect and process Kilter Board data')
    parser.add_argument('--board', type=str, default='kilter', help='Board name (default: kilter)')
    parser.add_argument('--database', type=str, default='kilter.db', help='Path to database file')
    parser.add_argument('--output', type=str, default='dataset', help='Output directory for processed data')
    parser.add_argument('--skip-sync', action='store_true', help='Skip database synchronization')
    parser.add_argument('--limit', type=int, default=20000, help='Maximum number of climbs to process')

    args = parser.parse_args()

    # Setup BoardLib and get database
    if not args.skip_sync:
        db_path = setup_boardlib(args.board, args.database)
    else:
        db_path = args.database

    # Extract data from database
    climbs_df, holds_df, board_dimensions = extract_climbs_data(db_path)

    # Process data for machine learning
    ml_df = process_data_for_ml(climbs_df, holds_df, board_dimensions, args.output)

    # Create visualization data
    create_visualization_data(climbs_df, holds_df, args.output)

    if ml_df.empty:
        print("\nWarning: No usable data was processed. Generating synthetic data instead.")
        print("Running synthetic data generator...")

        # Generate synthetic data as a fallback
        from generate_synthetic_data import generate_synthetic_data
        try:
            generate_synthetic_data(500, args.output)
            print("Synthetic data generation complete! You can proceed with visualization and modeling.")
        except ImportError:
            print("Could not import generate_synthetic_data. Please create this script first.")
            print("See instructions for creating synthetic data in the documentation.")
    else:
        print("\nData collection and processing complete!")
        print(f"Number of climbs processed: {len(ml_df)}")
        print(f"Data saved to {args.output}")
        print("\nNext steps:")
        print("1. Run the visualization script to explore the data")
        print("2. Train the grade prediction model")

if __name__ == "__main__":
    main()