import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import unicodedata
import re
from datetime import timedelta, datetime
from rapidfuzz import fuzz
from sklearn.impute import KNNImputer
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from difflib import SequenceMatcher

class DatabaseManager:
    """Handles all database operations and data loading"""
    
    def __init__(self):
        load_dotenv()
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'user': os.getenv("DB_USER"), 
            'database': os.getenv("DB_NAME"),
            'password': os.getenv("DB_PASSWORD")
        }
    
    def load_events_data(self) -> pd.DataFrame:
        """Load and perform initial cleaning of events data"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM events")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        df_events = pd.DataFrame(rows, columns=columns)
        cursor.close()
        conn.close()
        
        return self._clean_events_data(df_events)
    
    def load_stats_data(self) -> pd.DataFrame:
        """Load and clean fighter stats data"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM stats")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        df_stats = pd.DataFrame(rows, columns=columns)
        cursor.close()
        conn.close()
        
        return self._clean_stats_data(df_stats)
    
    def _clean_events_data(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess events data"""
        # Convert to datetime
        df_events['event_date'] = pd.to_datetime(df_events['event_date'], format="%Y-%m-%d")
        
        # Drop duplicates and invalid rows
        df_events = df_events.drop_duplicates()
        df_events = df_events[~((df_events['fighter_red'].isna()) & (df_events['fighter_blue'].isna()))]
        df_events = df_events[~df_events[['winner']].isnull().all(axis=1)]
        
        # Fill stances with most common stance
        df_fighters_red = df_events[['fighter_red', 'stance_red']].rename(
            columns={'fighter_red': 'fighter', 'stance_red': 'stance'}
        )
        df_fighters_blue = df_events[['fighter_blue', 'stance_blue']].rename(
            columns={'fighter_blue': 'fighter', 'stance_blue': 'stance'}
        )
        
        df_fighters = pd.concat([df_fighters_red, df_fighters_blue], ignore_index=True)
        df_fighters = df_fighters.drop_duplicates(subset=['fighter'], keep='first')
        
        stance_counts = df_fighters['stance'].value_counts(dropna=True)
        top_stance = stance_counts.index[0]
        
        df_events['stance_red'] = df_events['stance_red'].fillna(top_stance)
        df_events['stance_blue'] = df_events['stance_blue'].fillna(top_stance)
        
        # Convert numeric columns
        for col in df_events.select_dtypes(include=['number']).columns:
            df_events[col] = df_events[col].astype('int64')
        
        # Convert winner to boolean (1 = red wins, 0 = blue wins)
        df_events['winner'] = (df_events['winner'] == df_events['fighter_red']).astype(int)
        
        # Normalize weight classes
        df_events["weight_class"] = df_events["weight_class"].apply(self._normalize_weight_class)
        
        # Drop ID column and reset index
        df_events.drop(columns=['id'], axis=1, inplace=True, errors='ignore')
        df_events = df_events.reset_index(drop=True)
        
        return df_events
    
    def _clean_stats_data(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess stats data"""
        # Remove duplicates, keeping row with least NaN values
        df_stats = df_stats.assign(nan_count=df_stats.isnull().sum(axis=1)) \
                    .sort_values(['name', 'nan_count']) \
                    .drop_duplicates('name', keep='first') \
                    .drop('nan_count', axis=1)
        
        # Keep necessary columns
        keep_cols = ['name', 'octagon_debut', 'height', 'weight', 'reach',
                    'leg_reach', 'sig_strikes_landed_per_minute', 'sig_strikes_absorbed_per_minute', 
                    'takedowns_avg', 'submission_avg', 'knockdown_avg', 'fight_time_avg']
        df_stats = df_stats[keep_cols]
        
        # Handle zero values and fill NaNs
        df_stats[['height', 'weight']] = df_stats[['height', 'weight']].replace(0, np.nan)
        
        fill_zero_cols = ['sig_strikes_landed_per_minute', 'sig_strikes_absorbed_per_minute',
                        'takedowns_avg', 'submission_avg', 'knockdown_avg']
        df_stats[fill_zero_cols] = df_stats[fill_zero_cols].fillna(0)
        
        df_stats[['fight_time_avg']] = df_stats[['fight_time_avg']].fillna('00:00')
        
        # Convert time to seconds
        df_stats['fight_time_avg'] = pd.to_timedelta('00:' + df_stats['fight_time_avg']).dt.total_seconds().astype(int)
        
        return df_stats
    
    def _normalize_weight_class(self, val):
        """Normalize weight class names"""
        if not isinstance(val, str):
            return "Open Weight"
            
        val = val.strip().lower()
        val = re.sub(r"\s+", " ", val)
        val = val.replace("womens", "women's")
        val = val.replace("women ", "women's ")
        
        mapping = {
            "lightweight": "Lightweight", "welterweight": "Welterweight",
            "middleweight": "Middleweight", "featherweight": "Featherweight",
            "bantamweight": "Bantamweight", "heavyweight": "Heavyweight",
            "light heavyweight": "Light Heavyweight", "flyweight": "Flyweight",
            "women's strawweight": "Women's Strawweight", "women's flyweight": "Women's Flyweight",
            "women's bantamweight": "Women's Bantamweight", "open weight": "Open Weight",
            "catch weight": "Catch Weight"
        }
        
        if val in mapping:
            return mapping[val]
        
        # Fuzzy keyword matching
        if "heavyweight" in val and "light" not in val:
            return "Heavyweight"
        elif "lightweight" in val and "feather" not in val:
            return "Lightweight"
        elif "middleweight" in val:
            return "Middleweight"
        elif "featherweight" in val:
            return "Featherweight"
        elif "bantamweight" in val:
            return "Women's Bantamweight" if "women" in val else "Bantamweight"
        elif "flyweight" in val:
            return "Women's Flyweight" if "women" in val else "Flyweight"
        elif "strawweight" in val:
            return "Women's Strawweight"
        
        return "Open Weight"

class DataPreprocessor:
    """Handles data cleaning, normalization, and preprocessing"""
    
    def __init__(self):
        self.imputers = {}
    
    def normalize_name(self, name):
        """Normalize fighter names for consistency"""
        if pd.isna(name):
            return ""
        
        name = str(name)
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        name = name.lower()
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name)
        
        words = name.split()
        return ' '.join(words).strip()
    
    def normalize_fighter_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all fighter names in the dataframe"""
        df = df.copy()
        df[["fighter_red", "fighter_blue"]] = df[["fighter_red", "fighter_blue"]].map(self.normalize_name)
        return df
    
    def load_or_create_imputer(self, imputer_path: str, data: pd.DataFrame, columns: List[str]):
        """Load existing imputer or create new one"""
        try:
            with open(imputer_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Imputer not found at {imputer_path}")

    
    def impute_missing_values(self, df: pd.DataFrame, imputer_path: str) -> pd.DataFrame:
        """Impute missing values using KNN imputer"""
        df = df.copy()
        missing_cols = df.columns[df.isnull().any()]
        
        if len(missing_cols) > 0:
            imputer = self.load_or_create_imputer(imputer_path, df, missing_cols)
            df[missing_cols] = imputer.transform(df[missing_cols])
        
        return df

class FighterMatcher:
    """Handles fuzzy matching of fighter names"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.unique_fighters = self._get_unique_fighters()
        
    def _get_unique_fighters(self) -> List[str]:
        """Extract unique fighter names from both red and blue columns"""
        red_fighters = self.df['fighter_red'].dropna().unique()
        blue_fighters = self.df['fighter_blue'].dropna().unique()
        all_fighters = list(set(list(red_fighters) + list(blue_fighters)))
        return sorted(all_fighters)
    
    def create_fuzzy_mapping(self, event_names, stats_names, threshold=85):
        """Create mapping using fuzzy matching"""
        mapping = {}
        
        for event_name in event_names:
            if pd.isna(event_name):
                continue
                
            best_match = None
            best_score = 0
            
            for stats_name in stats_names:
                if pd.isna(stats_name):
                    continue
                    
                ratio = fuzz.ratio(event_name, stats_name)
                token_sort_ratio = fuzz.token_sort_ratio(event_name, stats_name)
                token_set_ratio = fuzz.token_set_ratio(event_name, stats_name)
                score = max(ratio, token_sort_ratio, token_set_ratio)
                
                if score >= threshold and score > best_score:
                    best_match = stats_name
                    best_score = score
            
            mapping[event_name] = best_match if best_match else event_name
        
        return mapping
    
    def interactive_match(self, input_name: str) -> Optional[str]:
        """Interactive matching with user prompts"""
        matches = self._find_best_matches(input_name, threshold=0.3, top_k=3)
        
        if not matches:
            print(f"✗ No matches found for '{input_name}'")
            return None
        
        best_match, best_score = matches[0]
        
        if best_score == 1.0:
            print(f"✓ Perfect match found: '{best_match}'")
            return best_match
        
        print(f"\nSuggestions for '{input_name}':")
        for i, (name, score) in enumerate(matches, 1):
            print(f"{i}. {name.title()} (Similarity: {score:.2%})")
        
        print(f"{len(matches) + 1}. None of the above")
        
        while True:
            try:
                choice = input(f"Select option (1-{len(matches) + 1}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(matches):
                    print("No fighter selected.")
                    return None
                elif 0 <= choice_idx < len(matches):
                    selected_name = matches[choice_idx][0]
                    print(f"✓ Selected: '{selected_name.title()}'")
                    return selected_name
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def _find_best_matches(self, input_name: str, threshold: float = 0.6, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find best matching fighter names"""
        similarities = []
        
        for fighter in self.unique_fighters:
            similarity = self._calculate_similarity(input_name, fighter)
            if similarity >= threshold:
                similarities.append((fighter, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        norm_name1 = self._normalize_name(name1)
        norm_name2 = self._normalize_name(name2)
        
        if not norm_name1 or not norm_name2:
            return 0.0
        
        if norm_name1 == norm_name2:
            return 1.0
        
        seq_similarity = SequenceMatcher(None, norm_name1, norm_name2).ratio()
        
        words1 = set(norm_name1.split())
        words2 = set(norm_name2.split())
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_similarity = intersection / union if union > 0 else 0
            similarity = 0.7 * seq_similarity + 0.3 * jaccard_similarity
        else:
            similarity = seq_similarity
        
        return similarity
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching"""
        if pd.isna(name):
            return ""
        
        normalized = str(name).lower().strip()
        normalized = re.sub(r'\b(jr|sr|iii|ii|iv)\b\.?', '', normalized)
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def get_fighter_input(self, corner_color: str, corner_emoji: str) -> Optional[str]:
        """Get fighter input with retry functionality"""
        while True:
            print(f"\nEnter the {corner_color} corner fighter name:")
            fighter_input = input(f"{corner_color.title()} fighter: ").strip()
            
            if not fighter_input:
                retry = input("No fighter name entered. Try again? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return None
                continue
            
            print(f"\n{corner_emoji} Searching for {corner_color.title()} fighter: '{fighter_input}'")
            matched_fighter = self.interactive_match(fighter_input)
            
            if matched_fighter:
                return matched_fighter
            else:
                print(f"\n❌ Could not match '{fighter_input}'")
                retry = input(f"Would you like to try a different name for the {corner_color.title()} fighter? (y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return None
    
    def run_fighter_matching(self) -> pd.DataFrame:
        """Run the complete interactive fighter matching process"""
        print("=== UFC Fighter Matching System ===")
        print(f"Available fighters: {len(self.unique_fighters)} unique fighters in database")
        
        while True:
            # Get red fighter
            matched_red = self.get_fighter_input("RED", "🔴")
            if not matched_red:
                print("Red fighter is required to continue.")
                restart = input("\nWould you like to start over? (y/n): ").strip().lower()
                if restart not in ['y', 'yes']:
                    print("Exiting fighter matching.")
                    return pd.DataFrame()
                continue
            
            # Get blue fighter
            matched_blue = self.get_fighter_input("BLUE", "🔵")
            if not matched_blue:
                print("Blue fighter is required to continue.")
                restart = input("\nWould you like to start over? (y/n): ").strip().lower()
                if restart not in ['y', 'yes']:
                    print("Exiting fighter matching.")
                    return pd.DataFrame()
                continue
            
            # Both fighters matched successfully
            break
        
        # Save matched names to the instance
        self.matched_names = [matched_red, matched_blue]
        
        # Filter dataframe
        names = [matched_red, matched_blue]
        print(f"\n=== Final Selection ===")
        print(f"🔴 Red Fighter: {matched_red.title()}")
        print(f"🔵 Blue Fighter: {matched_blue.title()}")
        
        df_filtered = self.df[
            (self.df['fighter_red'].isin(names)) |
            (self.df['fighter_blue'].isin(names))
        ]
        
        print(f"\n✅ Found {len(df_filtered)} matching fight records in database.")
        
        if len(df_filtered) > 0:
            print(f"Dataframe shape: {df_filtered.shape}")
        else:
            print("No fight records found for these fighters.")
        
        return df_filtered
    
    def get_matched_names(self) -> List[str]:
        """Get the matched fighter names as a list"""
        if hasattr(self, 'matched_names'):
            return self.matched_names.copy()
        else:
            return []

class FeatureEngineer:
    """Handles all feature engineering operations"""
    
    def __init__(self):
        pass
    
    def create_fighter_record_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create win/loss tracking features"""
        df_copy = df.copy()
        df_copy['event_date'] = pd.to_datetime(df_copy['event_date'])
        df_copy = df_copy.sort_values('event_date').reset_index(drop=True)
        
        # Initialize tracking dictionaries
        fighter_wins = {}
        fighter_losses = {}
        fighter_total_fights = {}
        
        # Initialize columns
        record_cols = ['wins_before_red', 'losses_before_red', 'total_fights_before_red',
                    'wins_before_blue', 'losses_before_blue', 'total_fights_before_blue']
        for col in record_cols:
            df_copy[col] = 0
        
        # Process each fight chronologically
        for idx, row in df_copy.iterrows():
            red_fighter = row['fighter_red']
            blue_fighter = row['fighter_blue']
            winner = row['winner']
            
            # Get current records BEFORE this fight
            for fighter, prefix in [(red_fighter, 'red'), (blue_fighter, 'blue')]:
                df_copy.at[idx, f'wins_before_{prefix}'] = fighter_wins.get(fighter, 0)
                df_copy.at[idx, f'losses_before_{prefix}'] = fighter_losses.get(fighter, 0)
                df_copy.at[idx, f'total_fights_before_{prefix}'] = fighter_total_fights.get(fighter, 0)
                
                # Initialize if new fighter
                if fighter not in fighter_wins:
                    fighter_wins[fighter] = 0
                    fighter_losses[fighter] = 0
                    fighter_total_fights[fighter] = 0
            
            # Update records AFTER processing this fight
            if winner == 1:  # Red wins
                fighter_wins[red_fighter] += 1
                fighter_losses[blue_fighter] += 1
            else:  # Blue wins
                fighter_wins[blue_fighter] += 1
                fighter_losses[red_fighter] += 1
            
            fighter_total_fights[red_fighter] += 1
            fighter_total_fights[blue_fighter] += 1
        
        return df_copy
    
    def create_win_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create win percentage features"""
        df_processed = df.copy()
        
        for color in ['red', 'blue']:
            df_processed[f'win_pct_before_{color}'] = np.where(
                df_processed[f'total_fights_before_{color}'] > 0,
                (df_processed[f'wins_before_{color}'] / df_processed[f'total_fights_before_{color}']).round(3),
                0
            )
        
        return df_processed
    
    def create_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create win/loss streak features"""
        df_copy = df.copy()
        df_copy['event_date'] = pd.to_datetime(df_copy['event_date'])
        df_copy = df_copy.sort_values('event_date').reset_index(drop=True)
        
        fighter_streaks = {}
        
        # Initialize columns
        streak_cols = ['win_streak_red', 'win_streak_blue', 'lose_streak_red', 'lose_streak_blue']
        for col in streak_cols:
            df_copy[col] = 0
        
        for idx, row in df_copy.iterrows():
            red_fighter = row['fighter_red']
            blue_fighter = row['fighter_blue']
            winner = row['winner']
            
            # Initialize fighter tracking
            for fighter in [red_fighter, blue_fighter]:
                if fighter not in fighter_streaks:
                    fighter_streaks[fighter] = {
                        'current_win_streak': 0,
                        'current_lose_streak': 0
                    }
            
            # Get current streaks BEFORE this fight
            df_copy.at[idx, 'win_streak_red'] = fighter_streaks[red_fighter]['current_win_streak']
            df_copy.at[idx, 'lose_streak_red'] = fighter_streaks[red_fighter]['current_lose_streak']
            df_copy.at[idx, 'win_streak_blue'] = fighter_streaks[blue_fighter]['current_win_streak']
            df_copy.at[idx, 'lose_streak_blue'] = fighter_streaks[blue_fighter]['current_lose_streak']
            
            # Update streaks AFTER processing this fight
            if winner == 1:  # Red wins
                fighter_streaks[red_fighter]['current_win_streak'] += 1
                fighter_streaks[red_fighter]['current_lose_streak'] = 0
                fighter_streaks[blue_fighter]['current_lose_streak'] += 1
                fighter_streaks[blue_fighter]['current_win_streak'] = 0
            else:  # Blue wins
                fighter_streaks[blue_fighter]['current_win_streak'] += 1
                fighter_streaks[blue_fighter]['current_lose_streak'] = 0
                fighter_streaks[red_fighter]['current_lose_streak'] += 1
                fighter_streaks[red_fighter]['current_win_streak'] = 0
        
        # Add derived features
        df_copy['on_win_streak_red'] = (df_copy['win_streak_red'] >= 1).astype(int)
        df_copy['on_win_streak_blue'] = (df_copy['win_streak_blue'] >= 1).astype(int)
        df_copy['long_win_streak_red'] = (df_copy['win_streak_red'] >= 3).astype(int)
        df_copy['long_win_streak_blue'] = (df_copy['win_streak_blue'] >= 3).astype(int)
        
        return df_copy
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month  
        df['day_of_week'] = df['event_date'].dt.dayofweek
        return df
    
    def create_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create difference features between red and blue fighters"""
        df_copy = df.copy()
        
        # Find numeric _blue columns
        blue_cols = [col for col in df_copy.columns 
                    if col.endswith('_blue') and pd.api.types.is_numeric_dtype(df_copy[col])]
        
        # Create difference columns
        diff_data = {}
        cols_to_drop = []
        
        for blue_col in blue_cols:
            red_col = blue_col.replace('_blue', '_red')
            if red_col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[red_col]):
                diff_col = blue_col.replace('_blue', '_diff')
                diff_data[diff_col] = df_copy[blue_col] - df_copy[red_col]
                cols_to_drop.extend([blue_col, red_col])
        
        # Create difference DataFrame and merge
        if diff_data:
            diff_df = pd.DataFrame(diff_data, index=df_copy.index)
            df_copy = df_copy.drop(cols_to_drop, axis=1)
            df_copy = pd.concat([df_copy, diff_df], axis=1)
        
        return df_copy.copy()

class ModelPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self):
        self.supported_models = {
            'LogisticRegression': self._predict_sklearn,
            'SGDClassifier': self._predict_sklearn,
            'RandomForestClassifier': self._predict_sklearn,
            'DecisionTreeClassifier': self._predict_sklearn,
            'GradientBoostingClassifier': self._predict_sklearn,
            'KNeighborsClassifier': self._predict_sklearn,
            'GaussianNB': self._predict_sklearn,
            'XGBClassifier': self._predict_sklearn,
            'AdaBoostClassifier': self._predict_sklearn,
            'SVC': self._predict_sklearn,
            'Neural Network': self._predict_pytorch
        }
    
    def load_model(self, model_path: str, model_type: str = 'auto'):
        """Load model from file"""
        model_path = Path(model_path)
        
        if model_type == 'auto':
            model_type = self._detect_model_type(model_path)
        
        if model_type == 'Neural Network':
            return self._load_pytorch_model(model_path)
        else:
            return self._load_sklearn_model(model_path)
    
    def _detect_model_type(self, model_path: Path) -> str:
        """Auto-detect model type"""
        if model_path.suffix in ['.pth', '.pt']:
            return 'Neural Network'
        elif model_path.suffix in ['.pkl', '.pickle']:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return type(model).__name__
            except:
                return 'Unknown'
        elif model_path.suffix == '.joblib':
            try:
                model = joblib.load(model_path)
                return type(model).__name__
            except:
                return 'Unknown'
        return 'Unknown'
    
    def _load_sklearn_model(self, model_path: Path):
        """Load sklearn/xgboost models"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except:
            return joblib.load(model_path)
    
    def _load_pytorch_model(self, model_path: Path, input_dim: int = None):
        """
        Load PyTorch model from file.
        Handles full model or state_dict using the Deep architecture.
        """
        try:
            # Try to load full model first
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, nn.Module):
                model.eval()
                return model
        except:
            pass

        # Load as state_dict with Deep architecture
        if input_dim is None:
            raise ValueError("For PyTorch state_dict, you must provide input_dim (number of features).")

        # Create Deep model architecture
        class Deep(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 1),
                )
            
            def forward(self, x):
                return self.network(x)

        model = Deep(input_dim)
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {e}")
    
    def predict(self, model, features: Dict, model_type: str = 'auto') -> Dict:
        """Make prediction with any supported model"""
        if model_type == 'auto':
            model_type = type(model).__name__
        
        if isinstance(model, nn.Module):
            model_type = 'Neural Network'
        
        if model_type in self.supported_models:
            return self.supported_models[model_type](model, features)
        else:
            return self._predict_sklearn(model, features)
    
    def _predict_sklearn(self, model, features: Dict) -> Dict:
        """Handle sklearn-compatible models"""
        feature_names = list(features.keys())
        feature_values = list(features.values())
        X = pd.DataFrame([feature_values], columns=feature_names)
        
        prediction = model.predict(X)[0]
        
        try:
            probabilities = model.predict_proba(X)[0]
            if len(probabilities) > 1:
                fighter_red_prob = probabilities[1]
                fighter_blue_prob = probabilities[0]
            else:
                fighter_red_prob = probabilities[0]
                fighter_blue_prob = 1 - probabilities[0]
            confidence = max(probabilities)
        except AttributeError:
            try:
                decision_score = model.decision_function(X)[0]
                fighter_red_prob = 1 / (1 + np.exp(-decision_score))
                fighter_blue_prob = 1 - fighter_red_prob
                confidence = max(fighter_red_prob, fighter_blue_prob)
            except AttributeError:
                fighter_red_prob = float(prediction)
                fighter_blue_prob = 1.0 - float(prediction)
                confidence = 1.0
        
        return {
            'prediction': prediction,
            'fighter_red_win_prob': fighter_red_prob,
            'fighter_blue_win_prob': fighter_blue_prob,
            'confidence': confidence,
            'model_name': type(model).__name__
        }
    
    def _predict_pytorch(self, model, features: Dict) -> Dict:
        """Handle PyTorch neural network models"""
        feature_values = list(features.values())
        X = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(X)
            
            if outputs.shape[1] == 2:
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                fighter_red_prob = probabilities[1]
                fighter_blue_prob = probabilities[0]
                prediction = int(fighter_red_prob > 0.5)
            elif outputs.shape[1] == 1:
                fighter_red_prob = torch.sigmoid(outputs).numpy()[0][0]
                fighter_blue_prob = 1 - fighter_red_prob
                prediction = int(fighter_red_prob > 0.5)
            else:
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                fighter_red_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                fighter_blue_prob = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
                prediction = int(fighter_red_prob > 0.5)
        
        confidence = max(fighter_red_prob, fighter_blue_prob)
        
        return {
            'prediction': prediction,
            'fighter_red_win_prob': fighter_red_prob,
            'fighter_blue_win_prob': fighter_blue_prob,
            'confidence': confidence,
            'model_name': 'Neural Network (PyTorch)'
        }

class UFCPredictionPipeline:
    """Main pipeline class that orchestrates the entire prediction process"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = ModelPredictor()
        
        # Storage for processed data
        self.events_data = None
        self.stats_data = None
        self.merged_data = None
        self.processed_data = None
        
        # Storage for encoders and scalers
        self.stance_encoder = None
        self.scaler = None
        self.top_features = None
    
    def load_data(self):
        """Load and perform initial cleaning of all data"""
        print("Loading events data...")
        self.events_data = self.db_manager.load_events_data()
        
        print("Loading stats data...")  
        self.stats_data = self.db_manager.load_stats_data()
        
        # Normalize names
        self.events_data = self.preprocessor.normalize_fighter_names(self.events_data)
        self.stats_data['name'] = self.stats_data['name'].map(self.preprocessor.normalize_name)
        
        print(f"Events data shape: {self.events_data.shape}")
        print(f"Stats data shape: {self.stats_data.shape}")
    
    def merge_data(self):
        """Merge events and stats data using fuzzy matching"""
        print("Performing fuzzy matching and merging data...")
        
        # Get unique fighter names
        blue_fighters = self.events_data['fighter_blue'].dropna().unique()
        red_fighters = self.events_data['fighter_red'].dropna().unique()
        all_event_fighters = set(blue_fighters) | set(red_fighters)
        stats_names = self.stats_data['name'].dropna().unique()
        
        # Create matcher and perform fuzzy mapping
        matcher = FighterMatcher(self.events_data)
        fuzzy_mapping = matcher.create_fuzzy_mapping(all_event_fighters, stats_names, threshold=85)
        
        # Apply fuzzy mapping
        self.events_data['fighter_blue_mapped'] = self.events_data['fighter_blue'].map(fuzzy_mapping)
        self.events_data['fighter_red_mapped'] = self.events_data['fighter_red'].map(fuzzy_mapping)
        
        # Fill NaN values with original names
        self.events_data['fighter_blue_mapped'] = self.events_data['fighter_blue_mapped'].fillna(self.events_data['fighter_blue'])
        self.events_data['fighter_red_mapped'] = self.events_data['fighter_red_mapped'].fillna(self.events_data['fighter_red'])
        
        # Merge stats for blue fighters
        self.merged_data = self.events_data.merge(
            self.stats_data,
            how='left',
            left_on='fighter_blue_mapped',
            right_on='name',
            suffixes=('', '_drop')
        )
        
        # Rename blue stats columns
        cols_to_rename_blue = [col for col in self.stats_data.columns if col != 'name']
        self.merged_data.rename(columns={col: f"{col}_blue" for col in cols_to_rename_blue}, inplace=True)
        self.merged_data.drop(columns=['name', 'fighter_blue_mapped'], inplace=True)
        
        # Merge stats for red fighters
        self.merged_data = self.merged_data.merge(
            self.stats_data,
            how='left',
            left_on='fighter_red_mapped',
            right_on='name',
            suffixes=('', '_drop')
        )
        
        # Rename red stats columns
        cols_to_rename_red = [col for col in self.stats_data.columns if col != 'name']
        self.merged_data.rename(columns={col: f"{col}_red" for col in cols_to_rename_red}, inplace=True)
        self.merged_data.drop(columns=['name', 'fighter_red_mapped'], inplace=True)
        
        # Fill missing debut dates
        self._fill_debut_dates()
        
        # Impute missing values
        self.merged_data = self.preprocessor.impute_missing_values(
            self.merged_data, "models/knn_imputer_feature_engineering.pkl"
        )
        
        # Sort by date and reset index
        self.merged_data['event_date'] = pd.to_datetime(self.merged_data['event_date'], errors='coerce')
        self.merged_data.sort_values(by=['event_date'], inplace=True)
        self.merged_data.reset_index(drop=True, inplace=True)
        
        print(f"Merged data shape: {self.merged_data.shape}")
    
    def _fill_debut_dates(self):
        """Fill missing debut dates with minimum event date for each fighter"""
        self.merged_data['octagon_debut_blue'] = pd.to_datetime(self.merged_data['octagon_debut_blue'], errors='coerce')
        self.merged_data['octagon_debut_red'] = pd.to_datetime(self.merged_data['octagon_debut_red'], errors='coerce')
        
        # Create fighter debuts lookup
        fighters_long = pd.concat([
            self.events_data[['fighter_red', 'event_date']].rename(columns={'fighter_red': 'fighter'}),
            self.events_data[['fighter_blue', 'event_date']].rename(columns={'fighter_blue': 'fighter'})
        ], ignore_index=True)
        
        fighter_debuts = fighters_long.groupby('fighter')['event_date'].min().reset_index()
        fighter_debuts.rename(columns={'event_date': 'octagon_debut'}, inplace=True)
        
        # Fill missing red debuts
        self.merged_data = self.merged_data.merge(
            fighter_debuts, left_on='fighter_red', right_on='fighter', how='left'
        )
        self.merged_data.loc[self.merged_data['octagon_debut_red'].isna(), 'octagon_debut_red'] = \
            self.merged_data.loc[self.merged_data['octagon_debut_red'].isna(), 'octagon_debut']
        self.merged_data.drop(columns=['fighter', 'octagon_debut'], inplace=True)
        
        # Fill missing blue debuts
        self.merged_data = self.merged_data.merge(
            fighter_debuts, left_on='fighter_blue', right_on='fighter', how='left'
        )
        self.merged_data.loc[self.merged_data['octagon_debut_blue'].isna(), 'octagon_debut_blue'] = \
            self.merged_data.loc[self.merged_data['octagon_debut_blue'].isna(), 'octagon_debut']
        self.merged_data.drop(columns=['fighter', 'octagon_debut'], inplace=True)
    
    def select_fighters_interactive(self) -> pd.DataFrame:
        """Interactive fighter selection and data filtering"""
        print("=== UFC Fighter Selection ===")
        
        matcher = FighterMatcher(self.merged_data)
        filtered_data = matcher.run_fighter_matching()
        matched_names = matcher.get_matched_names()
        
        if len(filtered_data) == 0:
            print("No data found for selected fighters.")
            return pd.DataFrame()
        
        # Add a new row for prediction
        if len(matched_names) == 2:
            filtered_data = self._add_prediction_row(filtered_data, matched_names)
        
        return filtered_data
    
    def _add_prediction_row(self, df: pd.DataFrame, fighter_names: List[str]) -> pd.DataFrame:
        """Add a new row for the upcoming fight prediction"""
        red_fighter, blue_fighter = fighter_names[0], fighter_names[1]
        
        # Define columns for each fighter
        red_columns = [
            'sig_strikes_landed_per_minute_red', 'sig_strikes_absorbed_per_minute_red', 
            'takedowns_avg_red', 'submission_avg_red', 'knockdown_avg_red', 'fight_time_avg_red',
            'stance_red', 'octagon_debut_red', 'height_red', 'weight_red', 'reach_red', 'leg_reach_red'
        ]
        
        blue_columns = [
            'sig_strikes_landed_per_minute_blue', 'sig_strikes_absorbed_per_minute_blue', 
            'takedowns_avg_blue', 'submission_avg_blue', 'knockdown_avg_blue', 'fight_time_avg_blue',
            'stance_blue', 'octagon_debut_blue', 'height_blue', 'weight_blue', 'reach_blue', 'leg_reach_blue'
        ]
        
        # Get last rows for each fighter
        last_red_row = df[df['fighter_red'] == red_fighter].iloc[-1] if len(df[df['fighter_red'] == red_fighter]) > 0 else None
        last_blue_row = df[df['fighter_blue'] == blue_fighter].iloc[-1] if len(df[df['fighter_blue'] == blue_fighter]) > 0 else None
        
        # Create new row data
        new_row_data = {
            'event_date': pd.to_datetime('today').date(),
            'fighter_red': red_fighter,
            'fighter_blue': blue_fighter,
        }
        
        # Copy stats from last fights
        if last_red_row is not None:
            for col in red_columns:
                if col in df.columns:
                    new_row_data[col] = last_red_row[col]
        
        if last_blue_row is not None:
            for col in blue_columns:
                if col in df.columns:
                    new_row_data[col] = last_blue_row[col]
            if 'weight_class' in df.columns:
                new_row_data['weight_class'] = last_blue_row['weight_class']
        
        # Add the new row
        df = pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce').dt.date
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Engineering features...")
        
        # Basic record features
        df = self.feature_engineer.create_fighter_record_features(df)
        df = self.feature_engineer.create_win_ratio_features(df)
        df = self.feature_engineer.create_streak_features(df)
        
        # Time features
        df = self.feature_engineer.add_time_features(df)
        
        print("Feature engineering complete!")
        return df
    
    def prepare_for_prediction(self, df: pd.DataFrame) -> Dict:
        """Prepare the final row for prediction"""
        print("Preparing data for prediction...")
        
        # Take the last row (the prediction row)
        prediction_row = df.tail(1).copy()
        
        # Load stance encoder
        try:
            with open("models/encoder_stance.pkl", "rb") as f:
                self.stance_encoder = pickle.load(f)
        except FileNotFoundError:
            print("Stance encoder not found, creating new one...")
            self.stance_encoder = OneHotEncoder(sparse_output=False, drop='first')
            self.stance_encoder.fit(df[['stance_red']])
        
        # Encode stances
        prediction_row = self._encode_stances(prediction_row)
        
        # Drop unnecessary columns
        columns_to_drop = [
            'event_date', 'event_name', 'round', 'time', 'weight_class', 'win_method', 
            'fighter_blue', 'fighter_red', 'octagon_debut_blue', 'octagon_debut_red',
            'knockdowns_red', 'knockdowns_blue', 'sig_attempts_red', 'sig_attempts_blue',
            'sig_strikes_red', 'sig_strikes_blue', 'total_strikes_attempts_red',
            'total_strikes_attempts_blue', 'total_strikes_red', 'total_strikes_blue',
            'sub_attempts_red', 'sub_attempts_blue', 'takedowns_red', 'takedowns_blue',
            'takedown_attempts_red', 'takedown_attempts_blue', 'control_time_red',
            'control_time_blue', 'head_strikes_red', 'head_strikes_blue',
            'head_attempts_red', 'head_attempts_blue', 'body_strikes_red',
            'body_strikes_blue', 'body_attempts_red', 'body_attempts_blue',
            'leg_strikes_red', 'leg_strikes_blue', 'leg_attempts_red',
            'leg_attempts_blue', 'distance_red', 'distance_blue',
            'distance_attempts_red', 'distance_attempts_blue', 'clinch_strikes_red',
            'clinch_strikes_blue', 'clinch_attempts_red', 'clinch_attempts_blue',
            'ground_strikes_red', 'ground_strikes_blue', 'ground_attempts_red',
            'ground_attempts_blue'
        ]
        
        prediction_row = prediction_row.drop(columns=columns_to_drop, errors='ignore')
        
        # Create difference features
        prediction_row = self.feature_engineer.create_difference_features(prediction_row)
        
        # Load top features and scaler
        self._load_feature_selection_and_scaler()
        
        # Select top features and scale
        if self.top_features is not None:
            # Ensure all required features exist
            missing_features = set(self.top_features) - set(prediction_row.columns)
            if missing_features:
                #print(f"Warning: Missing features: {missing_features}")
                for feature in missing_features:
                    prediction_row[feature] = 0.0
            
            prediction_row_selected = prediction_row[self.top_features]
        else:
            prediction_row_selected = prediction_row
        
        # Scale the features
        if self.scaler is not None:
            scaled_features = self.scaler.transform(prediction_row_selected)
            feature_dict = dict(zip(prediction_row_selected.columns, scaled_features[0]))
        else:
            feature_dict = prediction_row_selected.iloc[0].to_dict()
        
        print(f"Prepared {len(feature_dict)} features for prediction")
        return feature_dict
    
    def _encode_stances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode fighter stances"""
        # Transform red stance
        stance_red_encoded = self.stance_encoder.transform(df[['stance_red']])
        stance_red_names = [name.replace('stance_red_', '') + '_red'
                        for name in self.stance_encoder.get_feature_names_out(['stance_red'])]
        
        # Transform blue stance
        stance_blue_encoded = self.stance_encoder.transform(
            df[['stance_blue']].rename(columns={'stance_blue': 'stance_red'})
        )
        stance_blue_names = [name.replace('stance_red_', '') + '_blue'
                            for name in self.stance_encoder.get_feature_names_out(['stance_red'])]
        
        # Combine encoded features
        all_encoded = np.concatenate([stance_red_encoded, stance_blue_encoded], axis=1)
        all_feature_names = stance_red_names + stance_blue_names
        
        # Create DataFrame and merge
        encoded_df = pd.DataFrame(all_encoded, columns=all_feature_names, index=df.index)
        df = pd.concat([df.drop(['stance_red', 'stance_blue'], axis=1), encoded_df], axis=1)
        
        return df
    
    def _load_feature_selection_and_scaler(self):
        """Load feature selection and scaler"""
        try:
            # Load top features
            top_features_df = pd.read_csv('data/notebooks/top_features.csv')
            self.top_features = top_features_df['feature'].tolist()
            print(f"Loaded {len(self.top_features)} top features")
        except FileNotFoundError:
            print("Top features file not found, using all features")
            self.top_features = None
        
        try:
            # Load scaler
            with open("models/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("Loaded feature scaler")
        except FileNotFoundError:
            print("Scaler not found, using unscaled features")
            self.scaler = None
    
    def predict_single_model(self, features: Dict, model_path: str, model_type: str = 'auto') -> Dict:
        """Make prediction with a single model"""
        if model_path.endswith('.pth') or model_type == 'Neural Network':
            input_dim = len(features)  # number of input features
            model = self.predictor._load_pytorch_model(model_path, input_dim=input_dim)
        else:
            model = self.predictor.load_model(model_path, model_type)
        result = self.predictor.predict(model, features, model_type)
        return result
    
    def predict_ensemble(
        self, features: Dict, model_paths: List[Tuple[str, str]],
        red_name: str = None, blue_name: str = None) -> List[Dict]:
        """Make predictions with multiple models"""
        print("Making ensemble predictions...")
        results = []
        
        # Convert names to Title Case if available
        if red_name:
            red_name = red_name.title()
        if blue_name:
            blue_name = blue_name.title()
        
        for model_path, model_type in model_paths:
            try:
                result = self.predict_single_model(features, model_path, model_type)
                results.append(result)
                
                # Decide winner name
                if result['prediction'] == 1:  # Red wins
                    winner = red_name if red_name else "Fighter RED"
                else:  # Blue wins
                    winner = blue_name if blue_name else "Fighter BLUE"
                
                print(f"{result['model_name']} Prediction:")
                print(f"Winner: {winner}")
                print(f"Fighter Red probability: {result['fighter_red_win_prob']:.3f}")
                print(f"Fighter Blue probability: {result['fighter_blue_win_prob']:.3f}")
                print(f"Confidence: {result['confidence']:.3f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error with model {model_path}: {str(e)}")
        
        # Calculate ensemble prediction
        if results:
            avg_red_prob = np.mean([r['fighter_red_win_prob'] for r in results])
            avg_blue_prob = 1 - avg_red_prob
            ensemble_prediction = int(avg_red_prob > 0.5)
            
            if ensemble_prediction == 1:
                winner = red_name if red_name else "Fighter RED"
            else:
                winner = blue_name if blue_name else "Fighter BLUE"
            
            print("ENSEMBLE PREDICTION:")
            print(f"Winner: {winner}")
            print(f"Fighter Red probability: {avg_red_prob:.3f}")
            print(f"Fighter Blue probability: {avg_blue_prob:.3f}")
            print(f"Models used: {len(results)}")
        
        return results

    
    def run_complete_pipeline(self, model_paths: List[Tuple[str, str]] = None):
        """Run the complete prediction pipeline"""
        print("="*60)
        print("UFC FIGHT PREDICTION PIPELINE")
        print("="*60)
        
        # Default model paths
        if model_paths is None:
            model_paths = [
                ('models/adaboostclassifier.pkl', 'auto'),
                ('models/decisiontreeclassifier.pkl', 'auto'),
                ('models/gaussiannb.pkl', 'auto'),
                ('models/gradientboostingclassifier.pkl', 'auto'),
                ('models/kneighborsclassifier.pkl', 'auto'),
                ('models/logisticregression.pkl', 'auto'),
                ('models/PyTorch_state_dict.pth', 'Neural Network'),
                ('models/randomforestclassifier.pkl', 'auto'),
                ('models/sgdclassifier.pkl', 'auto'),
                ('models/svc.pkl', 'auto'),
                ('models/xgbclassifier.pkl', 'auto')
            ]
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Merge data
            self.merge_data()
            
            # Step 3: Interactive fighter selection
            filtered_data = self.select_fighters_interactive()
            if len(filtered_data) == 0:
                print("No valid data for prediction. Exiting.")
                return
            
            # Step 4: Feature engineering
            processed_data = self.engineer_features(filtered_data)
            
            # Step 5: Prepare for prediction
            features = self.prepare_for_prediction(processed_data)
            
            # Step 6: Make predictions
            last_row = processed_data.tail(1).iloc[0]
            red_name = last_row["fighter_red"]
            blue_name = last_row["fighter_blue"]

            results = self.predict_ensemble(features, model_paths, red_name, blue_name)
            
            print("\nPipeline completed successfully!")
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Usage example
if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = UFCPredictionPipeline()
    results = pipeline.run_complete_pipeline()
    
    # You can also run individual steps:
    # pipeline.load_data()
    # pipeline.merge_data()
    # filtered_data = pipeline.select_fighters_interactive()
    # processed_data = pipeline.engineer_features(filtered_data)
    # features = pipeline.prepare_for_prediction(processed_data)
    # single_result = pipeline.predict_single_model(features, 'models/sgdclassifier.pkl')