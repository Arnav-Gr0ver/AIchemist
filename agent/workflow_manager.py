from typing import Dict, List, Any

class WorkflowManager:
    """Manages the workflow for replication studies"""
    
    def generate_workflow(self, filtered_content: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate workflow steps based on the filtered paper content"""
        workflow = []
        
        # Step 1: Sandbox Setup
        workflow.append({
            "id": 1,
            "name": "Sandbox Environment Setup",
            "type": "sandbox_setup",
            "description": "Set up the computational environment with necessary dependencies",
            "details": self._generate_dependency_list(filtered_content),
            "status": "pending"
        })
        
        # Step 2: Web Retrieval (if necessary)
        retrieval_needed = self._check_if_retrieval_needed(filtered_content)
        if retrieval_needed:
            workflow.append({
                "id": 2,
                "name": "Web Retrieval",
                "type": "retrieval",
                "description": "Retrieve models and datasets from online sources",
                "details": self._generate_retrieval_info(filtered_content),
                "status": "pending"
            })
        
        # Step 3: Architecture Setup
        workflow.append({
            "id": 3 if retrieval_needed else 2,
            "name": "Architecture Setup",
            "type": "architecture_setup",
            "description": "Set up the model architecture based on the paper",
            "details": self._generate_architecture_info(filtered_content),
            "status": "pending"
        })
        
        # Step 4: Training Setup
        workflow.append({
            "id": 4 if retrieval_needed else 3,
            "name": "Training Setup",
            "type": "training_setup",
            "description": "Prepare training scripts and configurations",
            "details": self._generate_training_setup_info(filtered_content),
            "status": "pending"
        })
        
        # Step 5: Training and Logging
        workflow.append({
            "id": 5 if retrieval_needed else 4,
            "name": "Training and Logging",
            "type": "training",
            "description": "Execute training process and record logs",
            "details": self._generate_training_info(filtered_content),
            "status": "pending"
        })
        
        # Step 6: Evaluation
        workflow.append({
            "id": 6 if retrieval_needed else 5,
            "name": "Evaluation",
            "type": "evaluation",
            "description": "Evaluate the trained model according to metrics in the paper",
            "details": self._generate_evaluation_info(filtered_content),
            "status": "pending"
        })
        
        # Step 7: Ablation Studies (if mentioned in the paper)
        if self._check_if_ablation_needed(filtered_content):
            workflow.append({
                "id": 7 if retrieval_needed else 6,
                "name": "Ablation Studies",
                "type": "ablation",
                "description": "Perform ablation studies as described in the paper",
                "details": self._generate_ablation_info(filtered_content),
                "status": "pending"
            })
        
        return workflow
    
    def check_feasibility(self, workflow_steps: List[Dict[str, Any]], sandbox) -> Dict[str, Any]:
        """Check if the replication is feasible in the given sandbox"""
        sandbox_capabilities = sandbox.get_capabilities()
        
        # Check compute requirements
        training_step = next((step for step in workflow_steps if step["type"] == "training"), None)
        if training_step:
            required_compute = self._estimate_compute_requirements(training_step["details"])
            if required_compute["gpu_memory"] > sandbox_capabilities["gpu_memory"]:
                return {
                    "feasible": False,
                    "reason": f"Insufficient GPU memory. Required: {required_compute['gpu_memory']}GB, Available: {sandbox_capabilities['gpu_memory']}GB",
                    "alternatives": ["Reduce batch size", "Use model parallelism", "Simplify model architecture"]
                }
            
            if required_compute["disk_space"] > sandbox_capabilities["disk_space"]:
                return {
                    "feasible": False,
                    "reason": f"Insufficient disk space. Required: {required_compute['disk_space']}GB, Available: {sandbox_capabilities['disk_space']}GB",
                    "alternatives": ["Use smaller dataset", "Implement data streaming"]
                }
        
        # Check dependency compatibility
        setup_step = next((step for step in workflow_steps if step["type"] == "sandbox_setup"), None)
        if setup_step:
            incompatible_deps = self._check_dependency_compatibility(setup_step["details"], sandbox_capabilities)
            if incompatible_deps:
                return {
                    "feasible": False,
                    "reason": f"Incompatible dependencies: {', '.join(incompatible_deps)}",
                    "alternatives": ["Use different versions", "Use alternative libraries"]
                }
        
        return {
            "feasible": True,
            "estimated_time": self._estimate_completion_time(workflow_steps, sandbox_capabilities),
            "notes": "The replication study appears to be feasible with the available resources"
        }
    
    def _generate_dependency_list(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate a list of required dependencies based on paper content"""
        # This is a simplified implementation
        # A real system would use NLP to extract dependencies from the text
        dependencies = {
            "python_packages": [],
            "system_requirements": [],
            "data_requirements": []
        }
        
        # Look for common ML frameworks in methodology and implementation sections
        frameworks = ["tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas"]
        
        for section in ["methodology", "implementation", "training"]:
            if section in filtered_content:
                content = filtered_content[section].lower()
                for framework in frameworks:
                    if framework in content:
                        dependencies["python_packages"].append(framework)
        
        # Add common dependencies
        dependencies["python_packages"].extend(["matplotlib", "jupyter"])
        dependencies["system_requirements"].append("CUDA compatible GPU")
        
        return dependencies
    
    def _check_if_retrieval_needed(self, filtered_content: Dict[str, str]) -> bool:
        """Check if web retrieval is needed for models or datasets"""
        # Look for mentions of datasets or pretrained models
        retrieval_keywords = ["dataset", "data set", "corpus", "pretrained", "pre-trained", "checkpoint"]
        
        for section_content in filtered_content.values():
            for keyword in retrieval_keywords:
                if keyword.lower() in section_content.lower():
                    return True
        
        return False
    
    def _generate_retrieval_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about what needs to be retrieved"""
        # This would use NLP to extract dataset names, model sources, etc.
        retrieval_info = {
            "datasets": [],
            "pretrained_models": [],
            "other_resources": []
        }
        
        # Simple keyword matching
        dataset_keywords = ["dataset", "data set", "corpus"]
        model_keywords = ["pretrained", "pre-trained", "checkpoint", "weights"]
        
        for section, content in filtered_content.items():
            content_lower = content.lower()
            
            # Check for datasets
            for keyword in dataset_keywords:
                if keyword in content_lower:
                    # Simple regex to find potential dataset names
                    # This is a simplified approach
                    potential_datasets = re.findall(r'(?i)(?:' + keyword + r')\s+(?:called|named)?\s*["\']?([A-Za-z0-9_\-]+)["\']?', content)
                    retrieval_info["datasets"].extend(potential_datasets)
            
            # Check for pretrained models
            for keyword in model_keywords:
                if keyword in content_lower:
                    # Simple regex to find potential model names
                    potential_models = re.findall(r'(?i)(?:' + keyword + r')\s+(?:called|named)?\s*["\']?([A-Za-z0-9_\-]+)["\']?', content)
                    retrieval_info["pretrained_models"].extend(potential_models)
        
        # Remove duplicates
        retrieval_info["datasets"] = list(set(retrieval_info["datasets"]))
        retrieval_info["pretrained_models"] = list(set(retrieval_info["pretrained_models"]))
        
        return retrieval_info
    
    def _generate_architecture_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about the model architecture"""
        architecture_info = {
            "model_type": "unknown",
            "layers": [],
            "parameters": {},
            "implementation_notes": []
        }
        
        # Look for architecture information in relevant sections
        if "architecture" in filtered_content:
            content = filtered_content["architecture"]
            
            # Try to identify model type
            model_types = ["CNN", "RNN", "LSTM", "Transformer", "GAN", "VAE", "MLP"]
            for model_type in model_types:
                if model_type in content or model_type.lower() in content.lower():
                    architecture_info["model_type"] = model_type
                    break
            
            # Look for layer information
            # This would require more sophisticated NLP in a real system
            layer_keywords = ["layer", "conv", "lstm", "attention", "dense", "fully connected", "pooling"]
            for keyword in layer_keywords:
                if keyword in content.lower():
                    architecture_info["implementation_notes"].append(f"Paper mentions {keyword} layers")
        
        return architecture_info
    
    def _generate_training_setup_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about training setup"""
        training_setup = {
            "optimizer": "unknown",
            "learning_rate": "unknown",
            "batch_size": "unknown",
            "epochs": "unknown",
            "loss_function": "unknown",
            "regularization": [],
            "hyperparameters": {}
        }
        
        # Extract training information from the paper
        training_sections = ["training", "implementation", "methodology"]
        combined_content = ""
        for section in training_sections:
            if section in filtered_content:
                combined_content += filtered_content[section]
        
        # Simple pattern matching for common training parameters
        # In a real system, more sophisticated NLP would be used
        if re.search(r'(?i)(?:optimizer|optimization)[\s:]+([A-Za-z]+)', combined_content):
            match = re.search(r'(?i)(?:optimizer|optimization)[\s:]+([A-Za-z]+)', combined_content)
            training_setup["optimizer"] = match.group(1)
            
        if re.search(r'(?i)learning\s+rate[\s:]+(\d+\.\d+)', combined_content):
            match = re.search(r'(?i)learning\s+rate[\s:]+(\d+\.\d+)', combined_content)
            training_setup["learning_rate"] = float(match.group(1))
            
        if re.search(r'(?i)batch\s+size[\s:]+(\d+)', combined_content):
            match = re.search(r'(?i)batch\s+size[\s:]+(\d+)', combined_content)
            training_setup["batch_size"] = int(match.group(1))
            
        if re.search(r'(?i)(?:epochs|iterations)[\s:]+(\d+)', combined_content):
            match = re.search(r'(?i)(?:epochs|iterations)[\s:]+(\d+)', combined_content)
            training_setup["epochs"] = int(match.group(1))
            
        if re.search(r'(?i)loss\s+function[\s:]+([A-Za-z_]+)', combined_content):
            match = re.search(r'(?i)loss\s+function[\s:]+([A-Za-z_]+)', combined_content)
            training_setup["loss_function"] = match.group(1)
        
        return training_setup
    
    def _generate_training_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about the training process"""
        training_info = {
            "dataset_split": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "logging": ["loss", "accuracy"],
            "checkpointing": True,
            "early_stopping": False,
            "distributed_training": False
        }
        
        # In a real system, this would use NLP to extract training details
        # For now, we'll use some default values and simple pattern matching
        
        combined_content = ""
        for section in ["training", "implementation", "methodology"]:
            if section in filtered_content:
                combined_content += filtered_content[section]
        
        # Check for early stopping
        if "early stop" in combined_content.lower():
            training_info["early_stopping"] = True
        
        # Check for distributed training
        if any(term in combined_content.lower() for term in ["distributed", "multi-gpu", "parallel"]):
            training_info["distributed_training"] = True
        
        return training_info
    
    def _generate_evaluation_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about the evaluation process"""
        evaluation_info = {
            "metrics": [],
            "baseline_comparisons": [],
            "evaluation_datasets": []
        }
        
        # Look for evaluation information in relevant sections
        eval_sections = ["evaluation", "results"]
        combined_content = ""
        for section in eval_sections:
            if section in filtered_content:
                combined_content += filtered_content[section]
        
        # Look for common evaluation metrics
        common_metrics = ["accuracy", "precision", "recall", "f1", "AUC", "ROC", "BLEU", "ROUGE", "MSE", "MAE"]
        for metric in common_metrics:
            if metric.lower() in combined_content.lower():
                evaluation_info["metrics"].append(metric)
        
        # If no metrics found, add some default ones
        if not evaluation_info["metrics"]:
            evaluation_info["metrics"] = ["accuracy", "loss"]
        
        return evaluation_info
    
    def _check_if_ablation_needed(self, filtered_content: Dict[str, str]) -> bool:
        """Check if ablation studies are mentioned in the paper"""
        ablation_keywords = ["ablation", "ablation study", "ablation experiment"]
        
        for section_content in filtered_content.values():
            for keyword in ablation_keywords:
                if keyword.lower() in section_content.lower():
                    return True
        
        return False
    
    def _generate_ablation_info(self, filtered_content: Dict[str, str]) -> Dict[str, Any]:
        """Generate information about ablation studies"""
        ablation_info = {
            "components_to_ablate": [],
            "ablation_experiments": []
        }
        
        # In a real system, this would use NLP to extract specific ablation details
        # For now, we'll include a generic ablation plan
        ablation_info["components_to_ablate"] = ["model components", "hyperparameters", "data augmentation"]
        ablation_info["ablation_experiments"] = [
            {"name": "Remove component X", "description": "Train without component X"},
            {"name": "Modify hyperparameter Y", "description": "Train with different values of Y"}
        ]
        
        return ablation_info
    
    def _estimate_compute_requirements(self, training_details: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the compute requirements for training"""
        # This would be a heuristic based on model size, dataset size, etc.
        # For simplicity, we'll return some default values
        return {
            "gpu_memory": 8.0,  # GB
            "cpu_cores": 4,
            "ram": 16.0,  # GB
            "disk_space": 20.0  # GB
        }
    
    def _check_dependency_compatibility(self, dependencies: Dict[str, Any], 
                                      sandbox_capabilities: Dict[str, Any]) -> List[str]:
        """Check if dependencies are compatible with the sandbox"""
        # This would check version compatibility, etc.
        # For simplicity, we'll assume all dependencies are compatible
        return []
    
    def _estimate_completion_time(self, workflow_steps: List[Dict[str, Any]], 
                               sandbox_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the time to complete the replication study"""
        # This would use heuristics based on model complexity, dataset size, etc.
        # For simplicity, we'll return a default estimate
        return {
            "total_hours": 24,
            "breakdown": {
                "setup": 1,
                "training": 20,
                "evaluation": 2,
                "analysis": 1
            }
        }