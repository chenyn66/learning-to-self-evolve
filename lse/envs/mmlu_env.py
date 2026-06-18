from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import datasets

from lse.envs.base import BaseEnv
from lse.paths import mmlu_data_root

TRAIN_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
    'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

DEV_SUBJECTS = ['business', 'psychology', 'biology', 'chemistry', 'economics', 'math', 'physics', 'computer_science', 'engineering']

GPQA_SUBJECTS = ["electronic_science_and_technology", "philosophy", "traditional_chinese_medicine", "applied_economics", "mathematics", 
                "physics", "clinical_medicine", "computer_science_and_technology", "information_and_communication_engineering", 
                "control_science_and_engineering", "theoretical_economics", "law", "history", "basic_medicine", "education", 
                "materials_science_and_engineering", "electrical_engineering", "power_engineering_and_engineering_thermophysics", 
                "military_science", "biology", "business_administration", "language_and_literature", "public_health_and_preventive_medicine", 
                "chemistry", "hydraulic_engineering", "chemical_engineering_and_technology", "pharmacy", "geography", "art_studies", 
                "architecture", "forestry_engineering", "public_administration", "oceanography", "journalism_and_communication", 
                "nuclear_science_and_technology", "weapon_science_and_technology", "naval_architecture_and_ocean_engineering", 
                "environmental_science_and_engineering", "transportation_engineering", "geology", "musicology", "stomatology", 
                "mechanical_engineering", "aeronautical_and_astronautical_science_and_technology", "civil_engineering", "mechanics", 
                "petroleum_and_natural_gas_engineering", "sociology", "food_science_and_engineering", "agricultural_engineering", 
                "surveying_and_mapping_science_and_technology", "metallurgical_engineering", "library_information_and_archival_management", 
                "mining_engineering", "astronomy", "atmospheric_science", "optical_engineering", "animal_husbandry", "geophysics", 
                "crop_science", "forestry", "textile_science_and_engineering", "physical_education"]

GPQA_SUBFIELD_SUBJECTS = ["circuits_and_systems", "electronic_science_and_technology", "philosophical_aesthetics", "philosophy", "traditional_chinese_medicine", 
"traditional_chinese_pharmacy", "finance", "public_finance", "applied_economics", "quantitative_economics", "combinatorial_mathematics", 
"ordinary_differential_equations", "mathematical_analysis", "advanced_algebra", "mathematics", "probability_and_statistics", 
"polynomials_and_series_expansions", "geometry_and_topology", "functions_of_complex_variables", "fundamental_mathematics", "number_theory", 
"particle_and_nuclear_physics", "electrodynamics", "quantum_mechanics", "atomic_and_molecular_physics", "fluid_physics", "solid_state_physics", 
"physics", "thermodynamics_and_statistical_physics", "polymer_physics", "thermodynamics", "acoustics", "internal_medicine", "obstetrics_and_gynecology", 
"clinical_medicine", "neurology", "computer_science_and_technology", "databases", "signal_and_information_processing", 
"information_and_communication_engineering", "antenna_and_radio_communication", "communication_principles", "control_science_and_engineering", 
"theoretical_economics", "law", "criminal_law", "civil_and_commercial_law", "world_history", "historical_geography", "archaeology_and_museology", 
"basic_medicine", "human_anatomy_and_histology_embryology", "pathology_and_pathophysiology", "education", "materials_physics_and_chemistry", 
"power_electronics_and_electrical_drives", "electrical_theory_and_new_technologies", "electrical_engineering", "power_engineering_and_engineering_thermophysics", 
"thermal_energy_engineering", "engineering_thermophysics", "heat_transfer", "military_science", "botany", "biology", "genetics", "zoology", "physiology", 
"microbiology", "ecology", "business_administration", "language_and_literature", "public_health_and_preventive_medicine", "organic_chemistry", 
"physical_chemistry", "analytical_chemistry", "electrochemistry", "chemistry", "inorganic_chemistry", "hydraulics_and_hydrology", 
"mass_transport_and_separation_process_in_chemical_engineering", "fluid_flow_and_heat_transfer_in_chemical_engineering", 
"elements_of_chemical_reaction_engineering", "pharmacy", "geography", "art_studies", "fine_arts", "film_studies", "architecture", 
"forestry_engineering", "public_administration", "oceanography", "journalism_and_communication", "nuclear_science_and_technology", 
"weapon_science_and_technology", "naval_architecture_and_ocean_engineering", "environmental_science_and_engineering", "transportation_engineering", 
"geology", "mineralogy_petrology_and_economic_geology", "musicology", "music_history_education_and_technology", "stomatology", "manufacturing_automation", 
"aeronautical_and_astronautical_science_and_technology", "geotechnical_engineering", "civil_engineering", "fundamentals_of_dynamics_and_control", 
"theoretical_fluid_mechanics", "theoretical_mechanics", "rigid_body_mechanics", "petroleum_and_natural_gas_engineering", "sociology", 
"food_science_and_engineering", "agricultural_engineering", "surveying_and_mapping_science_and_technology", "iron_and_steel_metallurgy", 
"metallurgical_engineering", "library_information_and_archival_management", "mining_engineering", "astronomy", "astronomical_observation_and_technology", 
"atmospheric_science", "theoretical_optics", "optical_engineering", "applied_optics", "animal_husbandry", "geophysics", "crop_science", "forestry", 
"textile_science_and_engineering", "physical_education"]

class MMLUEnv(BaseEnv):
    """Single-problem environment for MMLU-Redux."""

    def __init__(self, args, item: Dict[str, Any], subject: str):
        self.args = copy.deepcopy(args)
        self._original_item = copy.deepcopy(item)
        self._subject = subject
        self._problem: Optional[Dict[str, Any]] = None
        self._summary: List[Dict[str, Any]] = []

    def reset(self) -> None:
        item = self._original_item
        
        self._problem = {
            "task_id": f"mmlu_{self._subject}_{id(item)}", # Use id() or some hash as unique ID
            "train": [],
            "test": [], # Not strictly used as we pass full object, but good for structure
            "question": item["question"],
            "choices": item["choices"],
            "meta": {
                "subject": self._subject,
                "gold_answer": self._get_gold_answer(item),
                "error_type": item.get("error_type", ""),
                "potential_reason": item.get("potential_reason", "")
            },
        }
        self._summary = []

    def _get_gold_answer(self, item):
        # Priority: correct_answer > answer
        # Convert to A, B, C, D
        
        def to_letter(val):
            if isinstance(val, int):
                return ["A", "B", "C", "D"][val]
            if isinstance(val, str):
                if val.isdigit():
                    idx = int(val)
                    if 0 <= idx <= 3:
                         return ["A", "B", "C", "D"][idx]
                if val.upper() in ["A", "B", "C", "D"]:
                    return val.upper()
            return None

        # Check correct_answer column first (for Redux corrections)
        correct_ans = item.get("correct_answer")
        if correct_ans:
            letter = to_letter(correct_ans)
            if letter:
                return letter

        # Fallback to original answer index
        ans = item.get("answer")
        if ans is not None:
            return to_letter(ans)
            
        return None

    def get_problem(self) -> Dict[str, Any]:
        assert self._problem is not None, "Environment not reset. Call reset() first."
        return self._problem

    def evaluate(self, predictions: List[str]) -> None:
        assert self._problem is not None, "Environment not reset. Call reset() first."
        pred = predictions[0] if predictions else ""
        
        # Simple exact match
        gold = self._problem["meta"]["gold_answer"]
        if not gold:
            # If gold is missing (e.g. bad question with no answer), treating as wrong or skip?
            # For now treat as wrong unless we handle "no_correct_answer" specifically.
            ok = False
        else:
            ok = (pred.strip().upper() == gold)

        self._summary = [
            {
                "task_id": self._problem["task_id"],
                "observations": [],
                "question": self._problem["question"],
                "choices": self._problem["choices"],
                "train_count": 0,
                "test_count": 1,
                "pred_outputs": pred,
                "gold_outputs": gold,
                "accuracy": 1 if ok else 0,
                "subject": self._subject,
                "error": "" if ok else f"Mismatch: pred {pred} != gold {gold}"
            }
        ]

    def get_summary(self) -> List[Dict[str, Any]]:
        return self._summary

    def clone_with_same_problem(self) -> "MMLUEnv":
        cloned = MMLUEnv(self.args, item=self._original_item, subject=self._subject)
        cloned._problem = copy.deepcopy(self._problem) if self._problem is not None else None
        cloned._summary = []
        return cloned


class BatchMMLU:
    """Batched wrapper holding one MMLUEnv per simulation.
    
    Enforces that all problems in the batch come from the same subject (if desired),
    or mixed if we want mixed batch. 
    Bird implementation fixes db_id per batch. We can fix subject per batch.
    """

    def __init__(self, args, exclude_subjects: Optional[List[str]] = None, subject: Optional[str] = None, exclude_indices: Optional[Set[int]] = None, **kwargs):
        self.args = copy.deepcopy(args)
        for k, v in kwargs.items():
            if hasattr(self.args, k):
                setattr(self.args, k, v)
            elif hasattr(self.args.task, k):
                setattr(self.args.task, k, v)

        self.n_sims = self.args.n_sims
        self._rng = random.Random(getattr(self.args, "seed", 0))
        self.resample_problem = bool(self.args.task.resample_problem)

        # Determine subject(s)
        self._fixed_subject = getattr(self.args.task, "subject", None)
        if subject:
            self._fixed_subject = subject

        if self.args.task.split == "train":
            available_subjects = TRAIN_SUBJECTS
        elif self.args.task.split == "dev":
            available_subjects = DEV_SUBJECTS
        elif self.args.task.split == "gpqa":
            available_subjects = GPQA_SUBJECTS
        elif self.args.task.split == "gpqa-subfield":
            available_subjects = GPQA_SUBFIELD_SUBJECTS
        else:
            raise ValueError(f"Invalid split: {self.args.task.split}")

        if exclude_subjects:
            available_subjects = [s for s in available_subjects if s not in exclude_subjects]

        # If no specific subject requested, pick one at random for this batch instance
        # (Mimicking Bird's behavior of picking one DB)
        if not self._fixed_subject or self._fixed_subject.lower() == "all":
             self._fixed_subject = self._rng.choice(available_subjects)

        # Load dataset for this subject
        # We load "test" split as MMLU-Redux only has test split usually (or we treat it as such)
        try:
            data_dir = getattr(self.args.task, "data_dir", None)
            data_root = (
                Path(str(data_dir)).expanduser().resolve()
                if data_dir
                else mmlu_data_root()
            )
            if self.args.task.split == "train":
                self._dataset = datasets.load_dataset("edinburgh-dawg/mmlu-redux-2.0", self._fixed_subject, split="test")
            elif self.args.task.split == "gpqa":
                self._dataset = datasets.load_from_disk(str(data_root / "supergpqa-redux"))
                self._dataset = self._dataset[self._fixed_subject]
            elif self.args.task.split == "dev":
                self._dataset = datasets.load_from_disk(str(data_root / "mmlu-pro-redux"))
                self._dataset = self._dataset[self._fixed_subject]
            elif self.args.task.split == "gpqa-subfield":
                self._dataset = datasets.load_from_disk(str(data_root / "supergpqa-redux-subfield"))
                self._dataset = self._dataset[self._fixed_subject]
            else:
                raise ValueError(f"Invalid split: {self.args.task.split}")

            self._all_items = [item for item in self._dataset if item['error_type'] != 'no_correct_answer'] # filter out no_correct_answer
        except Exception as e:
            raise RuntimeError(f"Error loading MMLU subject {self._fixed_subject}: {e}") from e


        

        self.envs: List[MMLUEnv] = []
        self._summary: List[Dict[str, Any]] = []
        
        # Support holding out specific indices if needed (not fully implemented but structural placeholder)
        self._exclude_indices = set(exclude_indices) if exclude_indices else set()

        self._pool = [item for i, item in enumerate(self._all_items) if i not in self._exclude_indices]

    def reset(self) -> None:
        
        
        if not self._pool:
            raise ValueError(f"Empty pool for MMLU sampling: {self._pool}")

        if self.n_sims > len(self._pool):
            # Sampling with replacement if not enough
            sampled_items = self._rng.choices(self._pool, k=self.n_sims)
        else:
            if self.resample_problem:
                sampled_items = self._rng.sample(self._pool, k=self.n_sims)
            elif not hasattr(self, 'problems'):
                sampled_items = self._rng.sample(self._pool, k=self.n_sims)
            else:
                sampled_items = self.problems
        self.problems = sampled_items
        self.envs = [MMLUEnv(self.args, item=it, subject=self._fixed_subject) for it in self.problems]
        for env in self.envs:
            env.reset()

    def get_batch(self) -> List[Dict[str, Any]]:
        return [env.get_problem() for env in self.envs]

    def evaluate(self, predictions_batch: List[List[str]]) -> None:
        assert len(predictions_batch) == len(self.envs), "Mismatched batch sizes"
        
        # MMLU eval is fast (string match), no need for threads really, but keeping structure
        for env, preds in zip(self.envs, predictions_batch):
            env.evaluate(preds)
            
        self._summary = [s for env in self.envs for s in env.get_summary()]

    def get_summary(self) -> List[Dict[str, Any]]:
        return self._summary

    def clone_with_same_problems(self) -> "BatchMMLU":
        cloned: BatchMMLU = object.__new__(BatchMMLU)
        cloned.args = self.args
        cloned.n_sims = self.n_sims
        cloned._rng = self._rng
        cloned._fixed_subject = self._fixed_subject
        cloned._dataset = self._dataset
        cloned._all_items = self._all_items
        cloned.resample_problem = self.resample_problem
        cloned.problems = self.problems
        cloned.envs = [env.clone_with_same_problem() for env in self.envs]
        cloned._summary = []
        cloned._exclude_indices = self._exclude_indices
        return cloned

    @classmethod
    def create_test_envs(cls, args, test_info) -> "BatchMMLU":
        # test_info contains 'subject' and 'holdout_indices'
        test_envs = BatchMMLU(args, subject=test_info["subject"])
        test_envs.resample_problem = False
        test_envs.n_sims = len(test_info["holdout_indices"])
        
        # Filter to only holdout indices
        pool = test_envs._all_items
        indices = test_info["holdout_indices"]
        test_envs.problems = [pool[i] for i in indices if i < len(pool)]
        
        # Re-init envs
        test_envs.envs = [MMLUEnv(args, item=it, subject=test_info["subject"]) for it in test_envs.problems]
        for env in test_envs.envs:
            env.reset()
            
        return test_envs

