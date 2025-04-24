"""
Kilter Board Data Visualization Script

This script creates visualizations to analyze the Kilter Board dataset, helping to understand
patterns in route difficulty, hold placement, and other relevant features for grade prediction.

Usage:
    python visualize_kilter_data.py --data_dir dataset/ --output_dir visualizations/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data(data_dir):
    """Load the processed Kilter Board data."""
    print("Loading data...")

    ml_features_path = os.path.join(data_dir, 'ml_features.csv')
    essential_features_path = os.path.join(data_dir, 'essential_features.csv')
    visualization_data_path = os.path.join(data_dir, 'visualization_data.json')

    # Load ML features data
    if os.path.exists(ml_features_path):
        ml_df = pd.read_csv(ml_features_path)
        print(f"Loaded {len(ml_df)} climbs with ML features")
    else:
        ml_df = None
        print(f"Warning: ML features file not found at {ml_features_path}")

    # Load essential features data
    if os.path.exists(essential_features_path):
        essential_df = pd.read_csv(essential_features_path)
        print(f"Loaded {len(essential_df)} climbs with essential features")
    else:
        essential_df = ml_df if ml_df is not None else None
        print(f"Warning: Essential features file not found at {essential_features_path}")

    # Load visualization data
    if os.path.exists(visualization_data_path):
        with open(visualization_data_path, 'r') as f:
            vis_data = json.load(f)
        print(f"Loaded visualization data with {len(vis_data)} grades")
    else:
        vis_data = None
        print(f"Warning: Visualization data file not found at {visualization_data_path}")

    return ml_df, essential_df, vis_data

def visualize_grade_distribution(df, output_dir):
    """Create visualizations of the grade distribution."""
    print("Creating grade distribution visualizations...")

    plt.figure(figsize=(12, 6))

    # Define a custom sorting function for V-grades
    def v_grade_sort_key(grade):
        if pd.isna(grade) or not isinstance(grade, str):
            return -1  # Put missing values at the beginning
        if grade.startswith('V'):
            # Extract the number after 'V'
            try:
                # Handle both 'V5' and 'V10' formats
                grade_num = grade[1:]
                # Handle potential '+' or '-' modifiers
                if '+' in grade_num:
                    grade_num = grade_num.replace('+', '')
                if '-' in grade_num:
                    grade_num = grade_num.replace('-', '')
                return int(grade_num)
            except ValueError:
                return -1
        return -1

    # Sort grades and count occurrences
    if 'grade' in df.columns:
        # Get grade counts
        grade_counts = df['grade'].value_counts()

        # Create a DataFrame for easier sorting
        grade_df = pd.DataFrame({
            'grade': grade_counts.index,
            'count': grade_counts.values
        })

        # Sort by the custom key
        grade_df['sort_key'] = grade_df['grade'].apply(v_grade_sort_key)
        grade_df = grade_df.sort_values('sort_key')

        # Filter out any non-V grades or invalid ones
        grade_df = grade_df[grade_df['sort_key'] >= 0]

        # Create bar plot with sorted grades
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='grade', y='count', data=grade_df)
        plt.title('Distribution of Climb Grades', fontsize=14)
        plt.xlabel('Grade', fontsize=12)
        plt.ylabel('Number of Climbs', fontsize=12)
        plt.xticks(rotation=45)

        # Add count labels
        for i, row in enumerate(grade_df.itertuples()):
            ax.text(i, row.count + (max(grade_df['count']) * 0.01),
                    str(row.count), ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_distribution.png'), dpi=300)
        plt.close()

        # Create pie chart for grade groups
        plt.figure(figsize=(10, 10))

        # Group grades into categories
        def categorize_grade(grade):
            if pd.isna(grade) or not isinstance(grade, str):
                return 'Unknown'
            try:
                grade_str = grade.strip()
                if not grade_str.startswith('V'):
                    return 'Unknown'

                # Extract the grade number
                grade_num = int(grade_str[1:].split('+')[0])

                if grade_num <= 2:
                    return 'Beginner (V0-V2)'
                elif grade_num <= 5:
                    return 'Intermediate (V3-V5)'
                elif grade_num <= 8:
                    return 'Advanced (V6-V8)'
                else:
                    return 'Elite (V9+)'
            except:
                return 'Unknown'

        df['grade_category'] = df['grade'].apply(categorize_grade)
        category_counts = df['grade_category'].value_counts()

        # Create pie chart
        plt.pie(category_counts.values, labels=category_counts.index,
                autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette('viridis', len(category_counts)))
        plt.title('Distribution of Climb Difficulty Categories', fontsize=14)
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_categories_pie.png'), dpi=300)
        plt.close()

    # Create histogram of numerical grade values
    if 'grade_value' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['grade_value'].dropna(), bins=30, kde=True)
        plt.title('Distribution of Grade Values', fontsize=14)
        plt.xlabel('Grade Value', fontsize=12)
        plt.ylabel('Number of Climbs', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_value_distribution.png'), dpi=300)
        plt.close()

    # Create histogram of display difficulty
    if 'display_difficulty' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['display_difficulty'].dropna(), bins=30, kde=True)
        plt.title('Distribution of Display Difficulty', fontsize=14)
        plt.xlabel('Display Difficulty', fontsize=12)
        plt.ylabel('Number of Climbs', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'display_difficulty_distribution.png'), dpi=300)
        plt.close()

def visualize_feature_correlations(df, output_dir):
    """Visualize correlations between features and grades."""
    print("Creating feature correlation visualizations...")

    # Select numerical features
    numerical_cols = []
    for col in ['grade_value', 'display_difficulty', 'angle', 'is_benchmark',
                'num_holds', 'num_start_holds', 'num_finish_holds']:
        if col in df.columns:
            numerical_cols.append(col)

    # Add density and distance features if available
    for col in df.columns:
        if 'density_' in col or '_distance' in col:
            numerical_cols.append(col)

    # Filter out any missing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis.")
        return

    # Create correlation matrix
    corr_matrix = df[numerical_cols].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_matrix.png'), dpi=300)
    plt.close()

    # Correlation with grade value
    if 'grade_value' in numerical_cols:
        correlations = []
        for col in numerical_cols:
            if col != 'grade_value':
                try:
                    corr, _ = pearsonr(df['grade_value'].fillna(0), df[col].fillna(0))
                    correlations.append({'feature': col, 'correlation': corr})
                except:
                    print(f"Skipping correlation for {col} due to error")

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='correlation', y='feature', data=corr_df)
        plt.title('Feature Correlation with Grade Value', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_correlations.png'), dpi=300)
        plt.close()

    # Correlation with display difficulty
    if 'display_difficulty' in numerical_cols and 'display_difficulty' != 'grade_value':
        correlations = []
        for col in numerical_cols:
            if col != 'display_difficulty':
                try:
                    corr, _ = pearsonr(df['display_difficulty'].fillna(0), df[col].fillna(0))
                    correlations.append({'feature': col, 'correlation': corr})
                except:
                    print(f"Skipping correlation for {col} due to error")

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='correlation', y='feature', data=corr_df)
        plt.title('Feature Correlation with Display Difficulty', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'difficulty_correlations.png'), dpi=300)
        plt.close()

def visualize_hold_placement(vis_data, output_dir):
    """Visualize hold placement patterns by grade."""
    print("Creating hold placement visualizations...")

    # Create directory for hold visualizations
    hold_vis_dir = os.path.join(output_dir, 'hold_placement')
    os.makedirs(hold_vis_dir, exist_ok=True)

    if vis_data is None:
        print("Skipping hold placement visualizations - no visualization data available")
        return

    # Visualize sample routes for each grade
    for grade, climbs in vis_data.items():
        for i, climb in enumerate(climbs):
            if 'holds' not in climb or not climb['holds']:
                continue

            plt.figure(figsize=(8, 8))

            # Draw board outline
            plt.plot([0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0], 'k-', linewidth=2)

            # Plot holds
            for hold in climb['holds']:
                x, y = hold['x'], hold['y']

                if 'is_start' in hold and hold['is_start']:
                    plt.scatter(x, y, c='green', s=100, alpha=0.8, edgecolors='black')
                elif 'is_finish' in hold and hold['is_finish']:
                    plt.scatter(x, y, c='red', s=100, alpha=0.8, edgecolors='black')
                else:
                    plt.scatter(x, y, c='blue', s=60, alpha=0.7, edgecolors='black')

            plt.title(f"Grade {grade}: {climb.get('name', 'Unnamed')}", fontsize=14)
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            plt.xlim(-100, 1100)
            plt.ylim(-100, 1100)

            # Add legend
            plt.scatter([], [], c='green', s=100, label='Start Hold')
            plt.scatter([], [], c='red', s=100, label='Finish Hold')
            plt.scatter([], [], c='blue', s=60, label='Regular Hold')
            plt.legend(loc='upper right')

            plt.grid(linestyle='--', alpha=0.3)
            plt.savefig(os.path.join(hold_vis_dir, f"{grade}_climb_{i+1}.png"), dpi=300)
            plt.close()

    # Create hold density heatmaps by grade category
    if vis_data:
        print("Creating hold density heatmaps...")

        # Sort grades to make sure V-grades are in proper numerical order
        grade_keys = sorted(vis_data.keys(), key=lambda x:
                            int(x[1:].split('+')[0]) if x.startswith('V') and x[1:].split('+')[0].isdigit() else 999)

        # Group grades into categories
        grade_categories = {}

        for grade in grade_keys:
            if grade.startswith('V'):
                try:
                    grade_num = int(grade[1:].split('+')[0])

                    if grade_num <= 2:
                        category = 'Beginner'
                    elif grade_num <= 5:
                        category = 'Intermediate'
                    elif grade_num <= 8:
                        category = 'Advanced'
                    else:
                        category = 'Elite'

                    if category not in grade_categories:
                        grade_categories[category] = []

                    grade_categories[category].append(grade)
                except ValueError:
                    # Handle non-standard grade formats
                    pass
            else:
                # Handle non-V grades (might be categories already)
                if grade not in grade_categories:
                    grade_categories[grade] = [grade]

        # Create a heatmap for each category
        for category, grades in grade_categories.items():
            # Collect all holds for grades in this category
            all_holds = []

            for grade in grades:
                if grade in vis_data:
                    for climb in vis_data[grade]:
                        if 'holds' in climb:
                            for hold in climb['holds']:
                                all_holds.append((hold['x'], hold['y']))

            if all_holds:
                plt.figure(figsize=(10, 10))

                # Create 2D histogram
                heatmap, xedges, yedges = np.histogram2d(
                    [h[0] for h in all_holds],
                    [h[1] for h in all_holds],
                    bins=20,
                    range=[[0, 1000], [0, 1000]]
                )

                # Apply Gaussian smoothing for better visualization
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=1.0)

                # Plot heatmap
                plt.imshow(heatmap.T, origin='lower', extent=[0, 1000, 0, 1000],
                           cmap='hot', interpolation='nearest')
                plt.colorbar(label='Hold Density')

                plt.title(f'Hold Density Heatmap - {category} Routes', fontsize=14)
                plt.xlabel('X Coordinate', fontsize=12)
                plt.ylabel('Y Coordinate', fontsize=12)

                plt.grid(False)
                plt.savefig(os.path.join(output_dir, f'hold_density_{category.lower()}.png'), dpi=300)
                plt.close()

def visualize_feature_distributions(df, output_dir):
    """Visualize distributions of key features by grade."""
    print("Creating feature distribution visualizations...")

    # Check if we have grade data
    if 'grade' not in df.columns:
        print("Warning: No grade column found. Using grade_value instead.")
        if 'grade_value' in df.columns:
            # Create grade from grade_value
            df['grade'] = df['grade_value'].apply(lambda x: f"V{int(x)}" if not pd.isna(x) else None)
        else:
            print("No grade data available for feature distribution visualization.")
            return

    feature_groups = [
        {'name': 'Hold Count', 'cols': ['num_holds']},
        {'name': 'Hold Types', 'cols': ['num_start_holds', 'num_finish_holds']},
        {'name': 'Hold Density', 'cols': ['density_top', 'density_bottom', 'density_left', 'density_right']},
        {'name': 'Hold Distances', 'cols': ['avg_distance', 'max_distance', 'min_distance', 'std_distance']}
    ]

    for group in feature_groups:
        # Check if all columns are available
        available_cols = [col for col in group['cols'] if col in df.columns]

        if not available_cols:
            print(f"Skipping {group['name']} visualizations - missing columns")
            continue

        # Create box plots for each feature by grade
        for col in available_cols:
            plt.figure(figsize=(14, 8))

            # Filter grades with at least 5 samples for better visualization
            grade_counts = df['grade'].value_counts()
            grades_to_plot = grade_counts[grade_counts >= 5].index

            # Define a custom sorting function for V-grades
            def v_grade_sort_key(grade):
                if pd.isna(grade) or not isinstance(grade, str):
                    return 999  # Put missing values at the end
                if grade.startswith('V'):
                    # Extract the number after 'V'
                    try:
                        # Handle both 'V5' and 'V10' formats
                        grade_num = grade[1:]
                        # Handle potential '+' or '-' modifiers
                        if '+' in grade_num:
                            grade_num = grade_num.replace('+', '')
                        if '-' in grade_num:
                            grade_num = grade_num.replace('-', '')
                        return int(grade_num)
                    except ValueError:
                        return 999
                return 999

            # Sort grade values for better visualization
            grade_order = sorted(grades_to_plot, key=v_grade_sort_key)

            # Create filtered dataframe
            plot_df = df[df['grade'].isin(grades_to_plot)]

            # Create box plot
            sns.boxplot(x='grade', y=col, data=plot_df, order=grade_order)
            plt.title(f'{col} by Grade', fontsize=14)
            plt.xlabel('Grade', fontsize=12)
            plt.ylabel(col.replace('_', ' ').title(), fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_by_grade.png'), dpi=300)
            plt.close()

        # If there are multiple features in the group, create violin plots for comparison
        if len(available_cols) > 1 and 'grade_value' in df.columns:
            plt.figure(figsize=(12, 8))

            # Reshape data for plotting
            plot_data = []
            for col in available_cols:
                temp_df = df[['grade_value', col]].copy()
                temp_df['feature'] = col.replace('_', ' ').title()
                temp_df = temp_df.rename(columns={col: 'value'})
                plot_data.append(temp_df)

            plot_df = pd.concat(plot_data)

            # Create violin plot
            sns.violinplot(x='grade_value', y='value', hue='feature', data=plot_df,
                          split=True, inner='quart', palette='Set2')
            plt.title(f'{group["name"]} Distribution by Grade Value', fontsize=14)
            plt.xlabel('Grade Value', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(title='Feature')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{group["name"].lower().replace(" ", "_")}_violins.png'), dpi=300)
            plt.close()

def visualize_pca_analysis(df, output_dir):
    """Perform PCA analysis on grid features and visualize results."""
    print("Creating PCA visualizations...")

    # Check if we have grid features
    grid_cols = [col for col in df.columns if col.startswith('grid_')]

    if not grid_cols:
        print("Skipping PCA analysis - no grid features available")
        return

    # Check if we have grade value data
    if 'grade_value' not in df.columns:
        print("Skipping PCA analysis - no grade_value column available")
        return

    # Extract grid features and grade value
    X = df[grid_cols].values
    y = df['grade_value'].values

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Separate by grade category
    grade_categories = {
        'Beginner (V0-V2)': (0, 2),
        'Intermediate (V3-V5)': (3, 5),
        'Advanced (V6-V8)': (6, 8),
        'Elite (V9+)': (9, 16)
    }

    for category, (min_grade, max_grade) in grade_categories.items():
        mask = (y >= min_grade) & (y <= max_grade)
        if np.any(mask):  # Only plot if there are points in this category
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       alpha=0.5, label=category)

    plt.title('PCA of Hold Positions by Grade Category', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300)
    plt.close()

    # Explained variance ratio
    plt.figure(figsize=(10, 6))

    n_components = min(10, len(grid_cols))
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X)

    plt.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_)
    plt.plot(range(1, n_components + 1), np.cumsum(pca_full.explained_variance_ratio_),
            'ro-', linewidth=2)

    plt.title('Explained Variance by Principal Components', fontsize=14)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.xticks(range(1, n_components + 1))
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_variance.png'), dpi=300)
    plt.close()

def visualize_benchmark_comparison(df, output_dir):
    """Compare benchmark and non-benchmark routes."""
    print("Creating benchmark comparison visualizations...")

    # Check if we have benchmark info
    if 'is_benchmark' not in df.columns:
        print("Skipping benchmark comparison - no is_benchmark column available")
        return

    # Create comparison for hold count and difficulty
    comparison_features = []

    if 'num_holds' in df.columns:
        comparison_features.append('num_holds')

    for col in ['grade_value', 'display_difficulty']:
        if col in df.columns:
            comparison_features.append(col)

    if 'avg_distance' in df.columns:
        comparison_features.append('avg_distance')

    if not comparison_features:
        print("No suitable features for benchmark comparison")
        return

    # Create boxplots comparing benchmark and non-benchmark routes
    for feature in comparison_features:
        plt.figure(figsize=(10, 6))

        # Filter out NaN values
        plot_df = df.dropna(subset=[feature])

        # Create categorical benchmark column
        plot_df['Benchmark'] = plot_df['is_benchmark'].apply(lambda x: 'Benchmark' if x == 1 else 'Standard')

        # Create box plot
        sns.boxplot(x='Benchmark', y=feature, data=plot_df)
        plt.title(f'{feature.replace("_", " ").title()} Comparison: Benchmark vs. Standard Routes', fontsize=14)
        plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'benchmark_comparison_{feature}.png'), dpi=300)
        plt.close()

    # Create grouped bar chart for grade distribution
    if 'grade' in df.columns:
        plt.figure(figsize=(14, 8))

        # Calculate percentage of benchmarks by grade
        benchmark_by_grade = df.groupby('grade')['is_benchmark'].mean() * 100

        # Define a custom sorting function for V-grades
        def v_grade_sort_key(grade):
            if pd.isna(grade) or not isinstance(grade, str):
                return 999  # Put missing values at the end
            if grade.startswith('V'):
                # Extract the number after 'V'
                try:
                    # Handle both 'V5' and 'V10' formats
                    grade_num = grade[1:]
                    # Handle potential '+' or '-' modifiers
                    if '+' in grade_num:
                        grade_num = grade_num.replace('+', '')
                    if '-' in grade_num:
                        grade_num = grade_num.replace('-', '')
                    return int(grade_num)
                except ValueError:
                    return 999
            return 999

        # Sort grades for better visualization
        sorted_index = sorted(benchmark_by_grade.index, key=v_grade_sort_key)
        benchmark_by_grade = benchmark_by_grade.loc[sorted_index]

        # Plot bar chart
        sns.barplot(x=benchmark_by_grade.index, y=benchmark_by_grade.values)
        plt.title('Percentage of Benchmark Routes by Grade', fontsize=14)
        plt.xlabel('Grade', fontsize=12)
        plt.ylabel('Percentage of Benchmarks', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_percentage_by_grade.png'), dpi=300)
        plt.close()

def create_summary_report(output_dir, ml_df):
    """Create a summary report of the data analysis."""
    print("Creating summary report...")

    report_path = os.path.join(output_dir, 'data_analysis_summary.md')

    with open(report_path, 'w') as f:
        f.write("# Kilter Board Data Analysis Summary\n\n")

        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total number of climbs: {len(ml_df)}\n")

        if 'grade' in ml_df.columns:
            # Define a custom sorting function for V-grades
            def v_grade_sort_key(grade):
                if pd.isna(grade) or not isinstance(grade, str):
                    return 999  # Put missing values at the end
                if grade.startswith('V'):
                    # Extract the number after 'V'
                    try:
                        # Handle both 'V5' and 'V10' formats
                        grade_num = grade[1:]
                        # Handle potential '+' or '-' modifiers
                        if '+' in grade_num:
                            grade_num = grade_num.replace('+', '')
                        if '-' in grade_num:
                            grade_num = grade_num.replace('-', '')
                        return int(grade_num)
                    except ValueError:
                        return 999
                return 999

            grade_counts = ml_df['grade'].value_counts()
            # Sort grades
            sorted_grades = sorted([(grade, count) for grade, count in grade_counts.items()],
                                  key=lambda x: v_grade_sort_key(x[0]))

            f.write(f"- Number of unique grades: {len(grade_counts)}\n")
            f.write("- Grade distribution:\n")
            for grade, count in sorted_grades:
                f.write(f"  - {grade}: {count} climbs ({count/len(ml_df)*100:.1f}%)\n")

        if 'is_benchmark' in ml_df.columns:
            benchmark_count = ml_df['is_benchmark'].sum()
            f.write(f"- Benchmark routes: {benchmark_count} ({benchmark_count/len(ml_df)*100:.1f}%)\n")

        # Feature overview
        f.write("\n## Feature Overview\n\n")

        for col in ['num_holds', 'num_start_holds', 'num_finish_holds']:
            if col in ml_df.columns:
                f.write(f"- {col.replace('_', ' ').title()}: Avg = {ml_df[col].mean():.1f}, Min = {ml_df[col].min()}, Max = {ml_df[col].max()}\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Correlation analysis
        if 'grade_value' in ml_df.columns:
            correlations = []
            for col in ml_df.columns:
                if col != 'grade_value' and ml_df[col].dtype in [np.int64, np.float64]:
                    try:
                        corr, _ = pearsonr(ml_df['grade_value'].fillna(0), ml_df[col].fillna(0))
                        correlations.append((col, corr))
                    except:
                        pass

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            f.write("### Strongest correlations with grade value:\n")
            for col, corr in correlations[:5]:
                f.write(f"- {col.replace('_', ' ').title()}: {corr:.3f}\n")

        # Visualization references
        f.write("\n## Visualizations\n\n")
        f.write("Key visualizations created:\n\n")
        f.write("1. Grade distribution\n")
        f.write("2. Feature correlations\n")
        f.write("3. Hold placement patterns\n")
        f.write("4. Feature distributions by grade\n")
        f.write("5. PCA analysis of hold positions\n")
        f.write("6. Benchmark vs. standard route comparisons\n")

        # Next steps
        f.write("\n## Next Steps\n\n")
        f.write("1. Train machine learning models to predict grades\n")
        f.write("2. Evaluate model performance\n")
        f.write("3. Analyze feature importance\n")
        f.write("4. Test model on new routes\n")

    print(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Kilter Board data')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Directory containing the processed data')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output_dir)

    # Load data
    ml_df, essential_df, vis_data = load_data(args.data_dir)

    # Use whichever dataset is available
    df = essential_df if essential_df is not None else ml_df

    if df is None:
        print("Error: No data available for visualization.")
        return

    # Create visualizations
    visualize_grade_distribution(df, output_dir)
    visualize_feature_correlations(df, output_dir)
    visualize_hold_placement(vis_data, output_dir)
    visualize_feature_distributions(df, output_dir)
    visualize_pca_analysis(df, output_dir)
    visualize_benchmark_comparison(df, output_dir)

    # Create summary report
    create_summary_report(output_dir, df)

    print(f"All visualizations saved to {output_dir}")
    print("Visualization complete!")

if __name__ == "__main__":
    main()