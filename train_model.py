#!/usr/bin/env python3
"""
FilmQuant ML - Box Office Prediction Model Training

This script trains machine learning models to predict box office performance
based on film metadata including cast, crew, budget, and marketing data.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FilmQuantMLTrainer:
    def __init__(self, data_path="data/filmquant-ml-historical-data-2023-final.csv"):
        """Initialize the trainer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def load_and_explore_data(self):
        """Load dataset and perform initial exploration"""
        print("🎬 LOADING FILMQUANT ML DATASET")
        print("=" * 50)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} films with {len(self.df.columns)} features")
        
        # Basic info about target variable
        target_col = 'actual_box_office_domestic_usd'
        if target_col in self.df.columns:
            target_data = pd.to_numeric(self.df[target_col], errors='coerce').dropna()
            print(f"\nTarget variable: {target_col}")
            print(f"Films with box office data: {len(target_data)}/42")
            if len(target_data) > 0:
                print(f"Range: ${target_data.min():.1f}M - ${target_data.max():.1f}M")
                print(f"Median: ${target_data.median():.1f}M")
        
        # Show data types and missing values
        print(f"\nData Overview:")
        print(f"Missing values per column:")
        missing_data = self.df.isnull().sum()
        for col in missing_data[missing_data > 0].index:
            print(f"  {col}: {missing_data[col]} missing")
            
        return self.df
        
    def engineer_features(self):
        """Create and engineer features for ML training"""
        print("\n🔧 FEATURE ENGINEERING")
        print("=" * 30)
        
        # Create feature dataframe
        features_df = self.df.copy()
        
        # 1. CATEGORICAL ENCODING
        print("Encoding categorical features...")
        
        # Encode genres (one-hot encoding for top genres)
        genre_cols = ['genre1_id', 'genre2_id', 'genre3_id', 'genre4_id', 'genre5_id']
        all_genres = []
        for col in genre_cols:
            all_genres.extend(features_df[col].dropna().unique())
        
        # Get top genres
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(10).index.tolist()
        
        # Create binary features for top genres
        for genre in top_genres:
            genre_feature = f"genre_{genre.lower().replace(' ', '_').replace('/', '_')}"
            features_df[genre_feature] = 0
            for col in genre_cols:
                features_df.loc[features_df[col] == genre, genre_feature] = 1
        
        print(f"  Created {len(top_genres)} genre features")
        
        # 2. STAR POWER FEATURES
        print("Creating star power features...")
        
        # Load people mappings to create star power scores
        try:
            with open('data/mappings/people.json', 'r') as f:
                people_data = json.load(f)
            
            # Create A-list actor mapping (simplified)
            a_list_actors = [
                'Leonardo DiCaprio', 'Tom Cruise', 'Chris Pratt', 'Margot Robbie',
                'Ryan Gosling', 'Matt Damon', 'Keanu Reeves', 'Harrison Ford',
                'Jack Black', 'Emma Stone', 'Jennifer Lawrence', 'Scarlett Johansson'
            ]
            
            # Calculate star power score
            actor_cols = ['actor1_id', 'actor2_id', 'actor3_id', 'actor4_id', 'actor5_id', 'actor6_id']
            features_df['star_power_score'] = 0
            
            for _, row in features_df.iterrows():
                score = 0
                for col in actor_cols:
                    if pd.notna(row[col]):
                        actor_name = str(row[col]).strip()
                        if actor_name in a_list_actors:
                            # A-list actors get higher scores
                            if col == 'actor1_id':  # Lead actor
                                score += 3
                            else:  # Supporting actors
                                score += 1
                features_df.loc[features_df.index == row.name, 'star_power_score'] = score
            
            print(f"  Star power scores range: {features_df['star_power_score'].min()}-{features_df['star_power_score'].max()}")
            
        except Exception as e:
            print(f"  Warning: Could not create star power features: {e}")
            features_df['star_power_score'] = 0
        
        # 3. BUDGET CATEGORIES
        print("Creating budget category features...")
        budget_series = pd.to_numeric(features_df['budget_usd'], errors='coerce')
        features_df['budget_category_low'] = (budget_series <= 50).astype(int)
        features_df['budget_category_mid'] = ((budget_series > 50) & (budget_series <= 150)).astype(int)
        features_df['budget_category_high'] = (budget_series > 150).astype(int)
        
        # 4. RELEASE DATE FEATURES
        print("Creating release date features...")
        try:
            dates = pd.to_datetime(features_df['release_date'])
            features_df['release_month'] = dates.dt.month
            features_df['release_quarter'] = dates.dt.quarter
            features_df['is_summer_release'] = ((dates.dt.month >= 5) & (dates.dt.month <= 8)).astype(int)
            features_df['is_holiday_release'] = ((dates.dt.month == 11) | (dates.dt.month == 12)).astype(int)
            print("  Release timing features created")
        except Exception as e:
            print(f"  Warning: Could not parse release dates: {e}")
        
        # 5. STUDIO POWER
        print("Creating studio features...")
        major_studios = ['Walt Disney Pictures', 'Universal Pictures', 'Warner Bros. Pictures', 
                        'Marvel Studios (Walt Disney Studios)', 'Paramount Pictures']
        features_df['is_major_studio'] = features_df['studio_id'].isin(major_studios).astype(int)
        
        # 6. RUNTIME FEATURES
        runtime_series = pd.to_numeric(features_df['runtime_minutes'], errors='coerce')
        features_df['is_long_runtime'] = (runtime_series > 150).astype(int)
        
        print(f"Feature engineering complete. Total features: {len(features_df.columns)}")
        
        return features_df
        
    def prepare_training_data(self, features_df):
        """Prepare features and target for ML training"""
        print("\n📊 PREPARING TRAINING DATA")
        print("=" * 35)
        
        # Define target variable
        target_col = 'actual_box_office_domestic_usd'
        
        # Check target availability
        target_series = pd.to_numeric(features_df[target_col], errors='coerce')
        valid_target_mask = target_series.notna()
        
        print(f"Films with box office data: {valid_target_mask.sum()}/{len(features_df)}")
        
        if valid_target_mask.sum() < 5:
            print("❌ ERROR: Not enough films with box office data for training!")
            print("   Need at least 5 films with target values")
            return None, None
        
        # Filter to films with target data
        train_data = features_df[valid_target_mask].copy()
        y = target_series[valid_target_mask].values
        
        # Select features for training
        feature_columns = [
            'budget_usd', 'runtime_minutes', 'star_power_score',
            'budget_category_low', 'budget_category_mid', 'budget_category_high',
            'release_month', 'release_quarter', 'is_summer_release', 'is_holiday_release',
            'is_major_studio', 'is_long_runtime'
        ]
        
        # Add genre features
        genre_features = [col for col in train_data.columns if col.startswith('genre_')]
        feature_columns.extend(genre_features)
        
        # Add trailer views if available
        if 'kinocheck_trailer_views' in train_data.columns:
            trailer_views = pd.to_numeric(train_data['kinocheck_trailer_views'], errors='coerce')
            if trailer_views.notna().sum() > 0:
                feature_columns.append('kinocheck_trailer_views')
                print("  Including trailer views as feature")
        
        # Filter to existing columns
        feature_columns = [col for col in feature_columns if col in train_data.columns]
        
        # Create feature matrix
        X = train_data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)  # Fill missing with 0 for now
        
        print(f"Training features: {len(feature_columns)}")
        print(f"Training samples: {len(X)}")
        print(f"Feature names: {feature_columns}")
        
        return X, y
        
    def train_models(self, X, y):
        """Train multiple ML models and compare performance"""
        print("\n🤖 TRAINING ML MODELS")
        print("=" * 30)
        
        # Split data
        if len(X) < 10:
            # Small dataset - use simple split
            test_size = 0.2
        else:
            test_size = 0.25
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        print(f"Train set: {len(X_train)} films")
        print(f"Test set: {len(X_test)} films")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models to try
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for linear models, original for tree-based
                if 'Regression' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                print(f"  MAE: ${mae:.1f}M")
                print(f"  RMSE: ${rmse:.1f}M") 
                print(f"  R²: {r2:.3f}")
                
                # Store model
                self.models[name] = model
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Find best model
        if results:
            best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   MAE: ${results[best_model_name]['mae']:.1f}M")
            print(f"   R²: {results[best_model_name]['r2']:.3f}")
            
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name
        
        return results
        
    def analyze_feature_importance(self, model_results):
        """Analyze which features are most important for predictions"""
        print("\n📈 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        # Get feature importance from tree-based models
        for name, result in model_results.items():
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.X_train.columns
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n{name} - Top 10 Features:")
                for _, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
                
                self.feature_importance[name] = importance_df
        
    def save_model(self, output_dir="models"):
        """Save the trained model and metadata"""
        print(f"\n💾 SAVING MODEL")
        print("=" * 20)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model using pickle
        import pickle
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"filmquant_ml_model_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scalers.get('standard'),
            'feature_names': list(self.X_train.columns),
            'training_date': timestamp,
            'training_films': len(self.X_train) + len(self.X_test)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved: {model_path}")
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.json")
            with open(importance_path, 'w') as f:
                json.dump({name: df.to_dict('records') 
                          for name, df in self.feature_importance.items()}, f, indent=2)
            print(f"✅ Feature importance saved: {importance_path}")
        
        return model_path
        
    def run_full_training_pipeline(self):
        """Run the complete ML training pipeline"""
        print("🚀 FILMQUANT ML TRAINING PIPELINE")
        print("=" * 50)
        print(f"Starting training at {datetime.now()}")
        print()
        
        try:
            # 1. Load and explore data
            self.load_and_explore_data()
            
            # 2. Engineer features
            features_df = self.engineer_features()
            
            # 3. Prepare training data
            X, y = self.prepare_training_data(features_df)
            
            if X is None:
                print("❌ Training failed: Insufficient data")
                return None
            
            # 4. Train models
            results = self.train_models(X, y)
            
            # 5. Analyze feature importance
            self.analyze_feature_importance(results)
            
            # 6. Save model
            model_path = self.save_model()
            
            print(f"\n🎉 TRAINING COMPLETE!")
            print(f"✅ Best model: {self.best_model_name}")
            print(f"✅ Model saved: {model_path}")
            print(f"✅ Ready for predictions!")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # Run training
    trainer = FilmQuantMLTrainer()
    model_path = trainer.run_full_training_pipeline()
    
    if model_path:
        print(f"\n🎬 FilmQuant ML model training successful!")
        print(f"Model ready for box office predictions!")
    else:
        print(f"\n❌ Training failed. Check data and try again.")
