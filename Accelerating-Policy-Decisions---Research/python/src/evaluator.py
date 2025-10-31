"""
Policy Evaluation Module for Climate Policy Analysis

This module provides comprehensive evaluation metrics and benchmarks for 
climate policy document analysis, including retrieval accuracy, response quality,
and policy effectiveness assessment.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# ML and evaluation imports (will be available when requirements are installed)
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rouge_score import rouge_scorer
except ImportError:
    print("Some dependencies not installed. Run: pip install -r requirements.txt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Comprehensive evaluation framework for climate policy analysis systems.
    
    Provides metrics for:
    - Retrieval accuracy (Precision@K, NDCG@K, MRR)
    - Response quality (ROUGE, BERT Score, coherence)
    - Policy effectiveness (implementation success, impact assessment)
    - User satisfaction (relevance, usability, trust)
    """
    
    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize the policy evaluator.
        
        Args:
            reference_data_path: Path to reference/ground truth data
        """
        self.reference_data_path = reference_data_path
        self.evaluation_history = []
        
        # Initialize scorers
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                        use_stemmer=True)
        except Exception:
            logger.warning("ROUGE scorer not available")
            self.rouge_scorer = None
            
        # Evaluation benchmarks based on paper results
        self.benchmarks = {
            "precision_at_10": 0.861,  # Our system performance from paper
            "ndcg_at_10": 0.923,
            "mrr": 0.798,
            "response_time_seconds": 2.02,
            "user_satisfaction": 9.1,
            "f1_score": 0.847
        }
        
        # Policy effectiveness criteria
        self.policy_criteria = {
            "clarity": "Policy language clarity and comprehensibility",
            "feasibility": "Implementation feasibility and practicality",
            "ambition": "Ambition level relative to climate targets",
            "coverage": "Sectoral and geographic coverage",
            "timeline": "Implementation timeline appropriateness",
            "monitoring": "Monitoring and verification mechanisms"
        }
    
    def evaluate_retrieval_performance(self, 
                                     retrieved_docs: List[Dict[str, Any]], 
                                     relevant_docs: List[str],
                                     k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate document retrieval performance.
        
        Args:
            retrieved_docs: List of retrieved documents with scores
            relevant_docs: List of relevant document IDs
            k_values: K values for Precision@K and NDCG@K evaluation
            
        Returns:
            Dictionary of retrieval metrics
        """
        logger.info("Evaluating retrieval performance")
        
        if k_values is None:
            k_values = [5, 10, 20]
        
        metrics = {}
        
        # Extract retrieved document IDs
        retrieved_ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(retrieved_docs)]
        relevant_set = set(relevant_docs)
        
        # Calculate metrics for each K
        for k in k_values:
            retrieved_k = retrieved_ids[:k]
            relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in relevant_set]
            
            # Precision@K
            precision_k = len(relevant_retrieved) / k if k > 0 else 0
            metrics[f"precision_at_{k}"] = precision_k
            
            # Recall@K  
            recall_k = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0
            metrics[f"recall_at_{k}"] = recall_k
            
            # F1@K
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0
            metrics[f"f1_at_{k}"] = f1_k
            
            # NDCG@K (simplified version)
            dcg = 0
            for i, doc_id in enumerate(retrieved_k):
                if doc_id in relevant_set:
                    dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
                    
            # Ideal DCG (all relevant docs at top)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))
            
            ndcg_k = dcg / idcg if idcg > 0 else 0
            metrics[f"ndcg_at_{k}"] = ndcg_k
            
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1 / (i + 1)
                break
        metrics["mrr"] = mrr
        
        # Mean Average Precision (MAP)
        ap = 0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
                
        map_score = ap / len(relevant_set) if relevant_set else 0
        metrics["map"] = map_score
        
        logger.info(f"Retrieval evaluation complete. MAP: {map_score:.3f}")
        return metrics
    
    def evaluate_response_quality(self, 
                                generated_responses: List[str],
                                reference_responses: List[str]) -> Dict[str, float]:
        """
        Evaluate generated response quality against references.
        
        Args:
            generated_responses: List of generated responses
            reference_responses: List of reference responses
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Evaluating response quality")
        
        if len(generated_responses) != len(reference_responses):
            raise ValueError("Number of generated and reference responses must match")
            
        metrics = {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "coherence_score": [],
            "relevance_score": []
        }
        
        # Calculate ROUGE scores if available
        if self.rouge_scorer:
            for gen, ref in zip(generated_responses, reference_responses):
                scores = self.rouge_scorer.score(ref, gen)
                metrics["rouge1_f"].append(scores['rouge1'].fmeasure)
                metrics["rouge2_f"].append(scores['rouge2'].fmeasure) 
                metrics["rougeL_f"].append(scores['rougeL'].fmeasure)
        
        # Calculate coherence and relevance (simplified heuristics)
        for gen, ref in zip(generated_responses, reference_responses):
            # Coherence: based on sentence count and average sentence length
            sentences = gen.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            coherence = min(1.0, avg_sentence_length / 15)  # Normalize to 0-1
            metrics["coherence_score"].append(coherence)
            
            # Relevance: simplified cosine similarity using TF-IDF
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([gen, ref])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                metrics["relevance_score"].append(similarity)
            except Exception:
                metrics["relevance_score"].append(0.5)  # Default neutral score
        
        # Calculate averages
        final_metrics = {}
        for metric, values in metrics.items():
            if values:  # Only if we have values
                final_metrics[f"avg_{metric}"] = np.mean(values)
                final_metrics[f"std_{metric}"] = np.std(values)
        
        logger.info(f"Response quality evaluation complete. Avg ROUGE-L: {final_metrics.get('avg_rougeL_f', 0):.3f}")
        return final_metrics
    
    def evaluate_policy_effectiveness(self, 
                                    policy_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate policy effectiveness based on multiple criteria.
        
        Args:
            policy_documents: List of policy documents with annotations
            
        Returns:
            Policy effectiveness evaluation
        """
        logger.info(f"Evaluating effectiveness of {len(policy_documents)} policies")
        
        evaluation = {
            "overall_score": 0,
            "criteria_scores": {},
            "policy_scores": [],
            "recommendations": []
        }
        
        criteria_scores = {criterion: [] for criterion in self.policy_criteria}
        
        for i, doc in enumerate(policy_documents):
            policy_score = {"document_id": doc.get("id", f"policy_{i}"), "scores": {}}
            
            # Extract relevant features for evaluation (available for future use)
            _ = doc.get("climate_features", {})
            _ = doc.get("metadata", {})
            _ = doc.get("entities", {})
            
            # Evaluate each criterion
            for criterion in self.policy_criteria:
                score = self._evaluate_policy_criterion(doc, criterion)
                policy_score["scores"][criterion] = score
                criteria_scores[criterion].append(score)
            
            # Calculate overall score for this policy
            policy_score["overall"] = np.mean(list(policy_score["scores"].values()))
            evaluation["policy_scores"].append(policy_score)
        
        # Calculate average scores across all policies
        for criterion, scores in criteria_scores.items():
            evaluation["criteria_scores"][criterion] = {
                "mean": np.mean(scores) if scores else 0,
                "std": np.std(scores) if scores else 0,
                "median": np.median(scores) if scores else 0
            }
        
        # Overall effectiveness score
        all_scores = [score["overall"] for score in evaluation["policy_scores"]]
        evaluation["overall_score"] = np.mean(all_scores) if all_scores else 0
        
        # Generate recommendations
        evaluation["recommendations"] = self._generate_policy_recommendations(evaluation)
        
        logger.info(f"Policy effectiveness evaluation complete. Overall score: {evaluation['overall_score']:.3f}")
        return evaluation
    
    def _evaluate_policy_criterion(self, doc: Dict[str, Any], criterion: str) -> float:
        """
        Evaluate a single policy criterion for a document.
        
        Args:
            doc: Policy document
            criterion: Evaluation criterion
            
        Returns:
            Score for the criterion (0-1 scale)
        """
        metadata = doc.get("metadata", {})
        climate_features = doc.get("climate_features", {})
        entities = doc.get("entities", {})
        
        if criterion == "clarity":
            # Based on readability scores and document structure
            reading_ease = metadata.get("flesch_reading_ease", 50)
            return min(1.0, reading_ease / 100)
            
        elif criterion == "feasibility":
            # Based on implementation timeline and concrete measures
            policy_instruments = climate_features.get("policy_instruments", [])
            return min(1.0, len(policy_instruments) / 5)  # Normalize by expected max
            
        elif criterion == "ambition":
            # Based on numeric targets and urgency indicators
            targets = climate_features.get("numeric_targets", [])
            urgency = climate_features.get("urgency_score", 0)
            return min(1.0, (len(targets) + urgency) / 10)
            
        elif criterion == "coverage":
            # Based on sectoral mentions and geographic scope
            geographic_scope = climate_features.get("geographic_scope", [])
            return min(1.0, len(geographic_scope) / 4)
            
        elif criterion == "timeline":
            # Based on presence of dates and implementation schedules
            dates = entities.get("dates", [])
            return min(1.0, len(dates) / 5)
            
        elif criterion == "monitoring":
            # Based on mentions of monitoring, reporting, verification
            monitoring_terms = ["monitor", "report", "verify", "track", "measure"]
            text = doc.get("cleaned_text", "").lower()
            mentions = sum(text.count(term) for term in monitoring_terms)
            return min(1.0, mentions / 10)
            
        return 0.5  # Default neutral score
    
    def _generate_policy_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on evaluation results.
        
        Args:
            evaluation: Policy evaluation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        criteria_scores = evaluation["criteria_scores"]
        
        # Identify weak areas
        for criterion, scores in criteria_scores.items():
            if scores["mean"] < 0.5:  # Below threshold
                recommendations.append(
                    f"Improve {criterion}: Current average score {scores['mean']:.2f} "
                    f"indicates need for enhancement in {self.policy_criteria[criterion]}"
                )
        
        # Overall recommendations based on score distribution
        overall_score = evaluation["overall_score"]
        if overall_score < 0.4:
            recommendations.append("Critical: Overall policy effectiveness is low. Consider comprehensive policy redesign.")
        elif overall_score < 0.6:
            recommendations.append("Moderate: Policy framework needs significant improvements in multiple areas.")
        elif overall_score < 0.8:
            recommendations.append("Good: Policy framework is solid but has room for targeted improvements.")
        else:
            recommendations.append("Excellent: Policy framework demonstrates high effectiveness across criteria.")
            
        return recommendations
    
    def evaluate_user_satisfaction(self, 
                                 user_feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate user satisfaction based on feedback data.
        
        Args:
            user_feedback: List of user feedback records
            
        Returns:
            User satisfaction metrics
        """
        logger.info(f"Evaluating user satisfaction from {len(user_feedback)} feedback records")
        
        metrics = {
            "overall_satisfaction": [],
            "accuracy_rating": [],
            "relevance_rating": [],
            "interpretability_rating": [],
            "response_time_satisfaction": [],
            "trust_score": []
        }
        
        for feedback in user_feedback:
            # Extract ratings (assuming 1-10 scale)
            metrics["overall_satisfaction"].append(feedback.get("overall_rating", 5))
            metrics["accuracy_rating"].append(feedback.get("accuracy", 5))
            metrics["relevance_rating"].append(feedback.get("relevance", 5))
            metrics["interpretability_rating"].append(feedback.get("interpretability", 5))
            metrics["response_time_satisfaction"].append(feedback.get("response_time_rating", 5))
            metrics["trust_score"].append(feedback.get("trust_rating", 5))
        
        # Calculate statistics
        satisfaction_stats = {}
        for metric, values in metrics.items():
            if values:
                satisfaction_stats[f"avg_{metric}"] = np.mean(values)
                satisfaction_stats[f"std_{metric}"] = np.std(values)
                satisfaction_stats[f"median_{metric}"] = np.median(values)
        
        # Calculate composite satisfaction score
        key_metrics = ["overall_satisfaction", "accuracy_rating", "relevance_rating"]
        composite_scores = []
        for i in range(len(user_feedback)):
            score = np.mean([metrics[metric][i] for metric in key_metrics])
            composite_scores.append(score)
            
        satisfaction_stats["composite_satisfaction"] = np.mean(composite_scores) if composite_scores else 0
        
        logger.info(f"User satisfaction evaluation complete. Composite score: {satisfaction_stats['composite_satisfaction']:.2f}")
        return satisfaction_stats
    
    def run_comprehensive_evaluation(self, 
                                   evaluation_data: Dict[str, Any],
                                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all metrics.
        
        Args:
            evaluation_data: Dictionary containing all evaluation inputs
            output_path: Optional path to save results
            
        Returns:
            Complete evaluation report
        """
        logger.info("Running comprehensive evaluation")
        
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "benchmarks": self.benchmarks,
            "results": {}
        }
        
        # Retrieval evaluation
        if "retrieval_data" in evaluation_data:
            retrieval_data = evaluation_data["retrieval_data"]
            report["results"]["retrieval"] = self.evaluate_retrieval_performance(
                retrieval_data["retrieved_docs"],
                retrieval_data["relevant_docs"]
            )
        
        # Response quality evaluation
        if "response_data" in evaluation_data:
            response_data = evaluation_data["response_data"]
            report["results"]["response_quality"] = self.evaluate_response_quality(
                response_data["generated_responses"],
                response_data["reference_responses"]
            )
        
        # Policy effectiveness evaluation
        if "policy_data" in evaluation_data:
            policy_data = evaluation_data["policy_data"]
            report["results"]["policy_effectiveness"] = self.evaluate_policy_effectiveness(
                policy_data["policy_documents"]
            )
        
        # User satisfaction evaluation
        if "user_feedback" in evaluation_data:
            report["results"]["user_satisfaction"] = self.evaluate_user_satisfaction(
                evaluation_data["user_feedback"]
            )
        
        # Performance comparison with benchmarks
        report["benchmark_comparison"] = self._compare_with_benchmarks(report["results"])
        
        # Overall assessment
        report["overall_assessment"] = self._generate_overall_assessment(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to: {output_path}")
        
        logger.info("Comprehensive evaluation complete")
        return report
    
    def _compare_with_benchmarks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results with established benchmarks."""
        comparison = {}
        
        if "retrieval" in results:
            retrieval = results["retrieval"]
            comparison["retrieval"] = {
                "precision_at_10": {
                    "achieved": retrieval.get("precision_at_10", 0),
                    "benchmark": self.benchmarks["precision_at_10"],
                    "ratio": retrieval.get("precision_at_10", 0) / self.benchmarks["precision_at_10"]
                },
                "ndcg_at_10": {
                    "achieved": retrieval.get("ndcg_at_10", 0),
                    "benchmark": self.benchmarks["ndcg_at_10"], 
                    "ratio": retrieval.get("ndcg_at_10", 0) / self.benchmarks["ndcg_at_10"]
                }
            }
        
        return comparison
    
    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of system performance."""
        assessment = {
            "strengths": [],
            "weaknesses": [],
            "priority_improvements": [],
            "overall_rating": "Not Available"
        }
        
        results = report.get("results", {})
        
        # Analyze each component
        if "retrieval" in results:
            precision = results["retrieval"].get("precision_at_10", 0)
            if precision > 0.8:
                assessment["strengths"].append("High retrieval precision")
            elif precision < 0.5:
                assessment["weaknesses"].append("Low retrieval precision")
                assessment["priority_improvements"].append("Improve document ranking algorithm")
        
        if "response_quality" in results:
            rouge_score = results["response_quality"].get("avg_rougeL_f", 0)
            if rouge_score > 0.7:
                assessment["strengths"].append("High response quality")
            elif rouge_score < 0.4:
                assessment["weaknesses"].append("Poor response quality")
                assessment["priority_improvements"].append("Enhance response generation model")
        
        # Overall rating
        strength_count = len(assessment["strengths"])
        weakness_count = len(assessment["weaknesses"])
        
        if strength_count > weakness_count:
            assessment["overall_rating"] = "Good"
        elif weakness_count > strength_count:
            assessment["overall_rating"] = "Needs Improvement"
        else:
            assessment["overall_rating"] = "Satisfactory"
            
        return assessment


# Utility functions for evaluation
def create_sample_evaluation_data() -> Dict[str, Any]:
    """Create sample evaluation data for testing."""
    return {
        "retrieval_data": {
            "retrieved_docs": [
                {"id": "doc_1", "score": 0.95},
                {"id": "doc_2", "score": 0.87},
                {"id": "doc_3", "score": 0.72}
            ],
            "relevant_docs": ["doc_1", "doc_3", "doc_5"]
        },
        "response_data": {
            "generated_responses": [
                "Climate policy requires immediate action on carbon emissions.",
                "Renewable energy investment is crucial for sustainable development."
            ],
            "reference_responses": [
                "Urgent climate policy measures must address carbon emission reduction.",
                "Investment in renewable energy sources is essential for sustainability."
            ]
        },
        "user_feedback": [
            {"overall_rating": 9, "accuracy": 8, "relevance": 9, "interpretability": 8},
            {"overall_rating": 8, "accuracy": 9, "relevance": 8, "interpretability": 9}
        ]
    }


if __name__ == "__main__":
    # Example usage
    evaluator = PolicyEvaluator()
    
    # Create sample data
    sample_data = create_sample_evaluation_data()
    
    # Run evaluation
    report = evaluator.run_comprehensive_evaluation(sample_data)
    
    print("Evaluation complete!")
    print(f"Overall assessment: {report['overall_assessment']['overall_rating']}")
    
    if report["overall_assessment"]["strengths"]:
        print("Strengths:", ", ".join(report["overall_assessment"]["strengths"]))
    if report["overall_assessment"]["priority_improvements"]:
        print("Priority improvements:", ", ".join(report["overall_assessment"]["priority_improvements"]))