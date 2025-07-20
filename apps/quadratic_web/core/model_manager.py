import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

from config.config import Config

class ModelManager:
    """Model saving and loading system for trained QuadraticPredictor models"""
    
    def __init__(self):
        self.save_path = Path(Config.MODEL_SAVE_PATH)
        self.backup_path = Path(Config.MODEL_BACKUP_PATH)
        self.metadata_file = self.save_path / Config.MODEL_METADATA_FILE
        
        # Create directories
        self.save_path.mkdir(exist_ok=True)
        self.backup_path.mkdir(exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load central metadata file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.metadata = {'models': [], 'next_id': 1}
        else:
            self.metadata = {'models': [], 'next_id': 1}
    
    def _save_metadata(self):
        """Save central metadata file with backup"""
        try:
            # Create backup if metadata exists
            if self.metadata_file.exists():
                backup_file = self.backup_path / f"metadata_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(self.metadata_file, backup_file)
            
            # Save current metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save metadata: {e}")
    
    def save_model(self, predictor, scenario_key: str, model_name: str, 
               dataset_info: Dict, performance_metrics: Dict = None) -> str:
        """Save a trained QuadraticPredictor model"""
        
        if not predictor.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if len(model_name.strip()) == 0:
            raise ValueError("Model name cannot be empty")
            
        if len(model_name) > Config.MODEL_NAME_MAX_LENGTH:
            raise ValueError(f"Model name too long (max {Config.MODEL_NAME_MAX_LENGTH} characters)")
        
        # Check if we're at the model limit
        if len(self.metadata['models']) >= Config.MAX_SAVED_MODELS:
            raise ValueError(f"Maximum number of models ({Config.MAX_SAVED_MODELS}) reached")
        
        # Generate unique model ID
        model_id = f"model_{self.metadata['next_id']:03d}"
        model_dir = self.save_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        try:
            # ✅ Use the passed performance_metrics instead of predictor.performance_stats
            actual_performance = performance_metrics or predictor.performance_stats
            
            # Save model weights and network parameters
            model_data = {
                'scenario_key': scenario_key,
                'network_parameters': [param.tolist() for param in predictor.network.get_all_parameters()],
                'network_architecture': predictor.scenario.network_architecture,
                'activations': predictor.scenario.activations,
                'training_history': predictor.training_history,
                'performance_stats': actual_performance,  # ✅ Use correct performance data
                'is_trained': True
            }
            
            with open(model_dir / 'model_data.json', 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            # Save data processor scalers
            scaler_key_input = f"{predictor.scenario.name}_input"
            scaler_key_target = f"{predictor.scenario.name}_target"
            
            scalers_data = {
                'input_scaler': predictor.data_processor.scalers.get(scaler_key_input),
                'target_scaler': predictor.data_processor.scalers.get(scaler_key_target)
            }
            
            with open(model_dir / 'scalers.pkl', 'wb') as f:
                pickle.dump(scalers_data, f)
            
            # ✅ SINGLE model_info definition with correct performance metrics
            model_info = {
                'model_id': model_id,
                'model_name': model_name.strip(),
                'scenario_key': scenario_key,
                'scenario_name': predictor.scenario.name,
                'created_date': datetime.now().isoformat(),
                'dataset_size': dataset_info.get('total_equations', 0),
                'dataset_stats': dataset_info.get('stats', {}),
                'performance_metrics': actual_performance,  # ✅ Use correct performance data
                'description': f"Model trained on {dataset_info.get('total_equations', 0)} equations",
                'model_size_bytes': self._calculate_model_size(model_dir),
                'version': '1.0'
            }
            
            # Check model size limit
            if model_info['model_size_bytes'] > Config.MAX_MODEL_SIZE:
                shutil.rmtree(model_dir)
                raise ValueError(f"Model too large ({model_info['model_size_bytes']} bytes)")
            
            with open(model_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # Update central metadata
            self.metadata['models'].append(model_info)
            self.metadata['next_id'] += 1
            self._save_metadata()
            
            return model_id
            
        except Exception as e:
            # Cleanup on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise RuntimeError(f"Failed to save model: {e}")

    def save_models_batch(self, predictor, scenario_key: str, model_name: str, 
                     dataset_info: Dict, performance_metrics: Dict = None, 
                     model_prefix: str = None) -> str:
        """Save a model as part of a batch operation with folder organization"""
        
        if not predictor.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if len(model_name.strip()) == 0:
            raise ValueError("Model name cannot be empty")
            
        if len(model_name) > Config.MODEL_NAME_MAX_LENGTH:
            raise ValueError(f"Model name too long (max {Config.MODEL_NAME_MAX_LENGTH} characters)")
        
        # Check if we're at the model limit
        if len(self.metadata['models']) >= Config.MAX_SAVED_MODELS:
            raise ValueError(f"Maximum number of models ({Config.MAX_SAVED_MODELS}) reached")
        
        # Generate unique model ID
        model_id = f"model_{self.metadata['next_id']:03d}"
        
        # Create folder structure for batch saves
        if model_prefix:
            # Create prefix folder if it doesn't exist
            prefix_folder = self.save_path / model_prefix
            prefix_folder.mkdir(exist_ok=True)
            model_dir = prefix_folder / model_id
        else:
            model_dir = self.save_path / model_id
            
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Use the passed performance_metrics instead of predictor.performance_stats
            actual_performance = performance_metrics or predictor.performance_stats
            
            # Save model weights and network parameters
            model_data = {
                'scenario_key': scenario_key,
                'network_parameters': [param.tolist() for param in predictor.network.get_all_parameters()],
                'network_architecture': predictor.scenario.network_architecture,
                'activations': predictor.scenario.activations,
                'training_history': predictor.training_history,
                'performance_stats': actual_performance,
                'is_trained': True,
                'batch_info': {
                    'is_batch_save': True,
                    'model_prefix': model_prefix,
                    'batch_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(model_dir / 'model_data.json', 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            # Save data processor scalers
            scaler_key_input = f"{predictor.scenario.name}_input"
            scaler_key_target = f"{predictor.scenario.name}_target"
            
            scalers_data = {
                'input_scaler': predictor.data_processor.scalers.get(scaler_key_input),
                'target_scaler': predictor.data_processor.scalers.get(scaler_key_target)
            }
            
            with open(model_dir / 'scalers.pkl', 'wb') as f:
                pickle.dump(scalers_data, f)
            
            # Create model_info with batch information
            model_info = {
                'model_id': model_id,
                'model_name': model_name.strip(),
                'scenario_key': scenario_key,
                'scenario_name': predictor.scenario.name,
                'created_date': datetime.now().isoformat(),
                'dataset_size': dataset_info.get('total_equations', 0),
                'dataset_stats': dataset_info.get('stats', {}),
                'performance_metrics': actual_performance,
                'description': f"Batch model trained on {dataset_info.get('total_equations', 0)} equations",
                'model_size_bytes': self._calculate_model_size(model_dir),
                'version': '1.0',
                'batch_info': {
                    'is_batch_save': True,
                    'model_prefix': model_prefix,
                    'folder_path': str(model_dir.relative_to(self.save_path))
                }
            }
            
            # Check model size limit
            if model_info['model_size_bytes'] > Config.MAX_MODEL_SIZE:
                shutil.rmtree(model_dir)
                raise ValueError(f"Model too large ({model_info['model_size_bytes']} bytes)")
            
            # Save model info
            with open(model_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            # Update central metadata
            self.metadata['models'].append(model_info)
            self.metadata['next_id'] += 1
            self._save_metadata()
            
            return model_id
            
        except Exception as e:
            # Cleanup on failure
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise RuntimeError(f"Failed to save batch model: {e}")

    
    def load_model(self, model_id: str, data_processor, scenarios) -> Optional[Any]:
        """Load a saved model and return QuadraticPredictor instance"""
        
        # First, try to find the model directory
        model_dir = None
        
        # Check if it's in the root save path (single models)
        root_model_dir = self.save_path / model_id
        if root_model_dir.exists():
            model_dir = root_model_dir
        else:
            # Search in prefix folders for batch models
            for item in self.save_path.iterdir():
                if item.is_dir() and not item.name.startswith('model_'):
                    # This might be a prefix folder
                    potential_model_dir = item / model_id
                    if potential_model_dir.exists():
                        model_dir = potential_model_dir
                        break
        
        if not model_dir or not model_dir.exists():
            print(f"❌ Model directory not found for model_id: {model_id}")
            return None

        try:
            # Load model info
            model_info_file = model_dir / 'model_info.json'
            if not model_info_file.exists():
                print(f"❌ model_info.json not found in {model_dir}")
                return None
                
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)

            # Load model data
            model_data_file = model_dir / 'model_data.json'
            if not model_data_file.exists():
                print(f"❌ model_data.json not found in {model_dir}")
                return None
                
            with open(model_data_file, 'r') as f:
                model_data = json.load(f)

            # Load scalers
            scalers_file = model_dir / 'scalers.pkl'
            if not scalers_file.exists():
                print(f"❌ scalers.pkl not found in {model_dir}")
                return None
                
            with open(scalers_file, 'rb') as f:
                scalers_data = pickle.load(f)

            # Get scenario
            scenario_key = model_data['scenario_key']
            if scenario_key not in scenarios:
                print(f"❌ Scenario '{scenario_key}' not found in available scenarios")
                return None

            scenario = scenarios[scenario_key]

            # Import QuadraticPredictor here to avoid circular imports
            from core.predictor import QuadraticPredictor

            # Create new predictor instance
            predictor = QuadraticPredictor(scenario, data_processor)
            predictor.create_network()

            # Restore network parameters
            parameters = [np.array(param) for param in model_data['network_parameters']]
            predictor.network.set_all_parameters(parameters)

            # Restore data processor scalers
            scaler_key_input = f"{scenario.name}_input"
            scaler_key_target = f"{scenario.name}_target"

            if scalers_data['input_scaler']:
                data_processor.scalers[scaler_key_input] = scalers_data['input_scaler']

            if scalers_data['target_scaler']:
                data_processor.scalers[scaler_key_target] = scalers_data['target_scaler']

            # Restore training state
            predictor.training_history = model_data['training_history']
            predictor.performance_stats = model_data['performance_stats']
            predictor.is_trained = True

            print(f"✅ Successfully loaded model: {model_id} from {model_dir}")
            return predictor

        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}")
    
    def get_saved_models(self) -> List[Dict]:
        """Get list of all saved models sorted by creation date with folder organization"""
        models = self.metadata['models'].copy()
        
        # Add folder organization information
        for model in models:
            batch_info = model.get('batch_info', {})
            if batch_info.get('is_batch_save'):
                model['display_name'] = f"📁 {batch_info.get('model_prefix', 'batch')}/{model['model_name']}"
                model['is_batch_model'] = True
                model['folder_prefix'] = batch_info.get('model_prefix')
            else:
                model['display_name'] = model['model_name']
                model['is_batch_model'] = False
                model['folder_prefix'] = None
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_date'], reverse=True)
        
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a saved model"""
        model_dir = self.save_path / model_id
        
        if not model_dir.exists():
            return False
        
        try:
            # Move to backup before deletion
            backup_dir = self.backup_path / f"deleted_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(model_dir), str(backup_dir))
            
            # Remove from metadata
            self.metadata['models'] = [
                m for m in self.metadata['models'] 
                if m['model_id'] != model_id
            ]
            self._save_metadata()
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        models = [m for m in self.metadata['models'] if m['model_id'] == model_id]
        return models[0] if models else None
    
    def cleanup_old_backups(self, max_age_days: int = 7) -> int:
        """Clean up old backup files"""
        if not self.backup_path.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        for item in self.backup_path.iterdir():
            if item.stat().st_mtime < cutoff_time:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted_count += 1
                except Exception:
                    continue
        
        return deleted_count
    
    def _calculate_model_size(self, model_dir: Path) -> int:
        """Calculate total size of model directory"""
        total_size = 0
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def validate_model_integrity(self, model_id: str) -> Dict[str, bool]:
        """Validate model file integrity"""
        model_dir = self.save_path / model_id
        
        checks = {
            'directory_exists': model_dir.exists(),
            'model_data_exists': (model_dir / 'model_data.json').exists(),
            'model_info_exists': (model_dir / 'model_info.json').exists(),
            'scalers_exist': (model_dir / 'scalers.pkl').exists(),
            'valid_json': True
        }
        
        if checks['directory_exists']:
            try:
                # Try to load JSON files
                with open(model_dir / 'model_data.json', 'r') as f:
                    json.load(f)
                with open(model_dir / 'model_info.json', 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                checks['valid_json'] = False
        
        return checks
