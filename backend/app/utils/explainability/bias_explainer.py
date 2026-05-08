"""
Explainability Engine for ML Fairness.

Explains WHY bias occurs in a machine learning model by:
- Identifying feature contributions to bias
- Analyzing disparities between sensitive groups
- Detecting proxy/discriminatory features
- Generating human-readable explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class BiasExplainer:
    """
    Main explainability engine for identifying and explaining bias in ML models.
    """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attributes: pd.DataFrame,
        target_column: str,
    ):
        """
        Initialize BiasExplainer.

        Args:
            model: Trained ML model with predict() method
            X: Feature matrix (DataFrame)
            y: Target labels (Series)
            sensitive_attributes: DataFrame with sensitive attributes
            target_column: Name of target column in original dataset
        """
        self.model = model
        self.X = X.copy()
        self.y = y.copy()
        self.sensitive_attributes = sensitive_attributes.copy()
        self.target_column = target_column
        self.feature_names = list(X.columns)
        self.n_samples = len(X)

    def explain_bias(self) -> Dict[str, Any]:
        """
        Generate complete bias explanation with all components.

        Returns:
            Dict containing:
            - bias_detected: bool
            - top_bias_contributors: List of Feature importance with reasons
            - group_analysis: Per-group statistics and observations
            - proxy_features: Detected discriminatory proxies
            - summary: Human-readable summary
        """
        # Convert data to safe numpy array for prediction
        X_safe = self._prepare_data_for_prediction(self.X)

        # Get predictions
        try:
            y_pred = self.model.predict(X_safe)
        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {e}")

        # 1. Feature Importance Analysis
        feature_importance = self._analyze_feature_importance()

        # 2. Group-wise Analysis
        group_analysis = self._analyze_groups(y_pred)

        # 3. Proxy Feature Detection
        proxy_features = self._detect_proxy_features()

        # 4. Bias Contribution Heuristic
        bias_contributors = self._compute_bias_contribution(
            feature_importance, proxy_features
        )

        # 5. Human-readable explanations
        bias_detected = len(bias_contributors) > 0
        summary = self._generate_summary(
            bias_contributors, group_analysis, proxy_features
        )

        return {
            "bias_detected": bias_detected,
            "top_bias_contributors": bias_contributors,
            "group_analysis": group_analysis,
            "proxy_features": proxy_features,
            "summary": summary,
        }

    # ============ DATA PREPARATION ============

    def _prepare_data_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for model prediction with safe data cleaning.

        Ensures:
        - NaN values filled appropriately
        - Numeric columns safe for processing
        - Returns DataFrame to preserve column names for ColumnTransformer

        Args:
            X: Feature matrix

        Returns:
            Clean DataFrame ready for sklearn pipeline prediction
        """
        X_safe = X.copy()

        # Only fill NaN values, keep original dtypes for pipeline to process
        for col in X_safe.columns:
            if pd.api.types.is_numeric_dtype(X_safe[col]):
                # For numeric: replace inf with NaN, then fill NaN with 0
                X_safe[col] = X_safe[col].replace([np.inf, -np.inf], np.nan)
                X_safe[col] = X_safe[col].fillna(0.0)
            else:
                # For categorical: fill NaN with "UNKNOWN" string
                X_safe[col] = X_safe[col].fillna("UNKNOWN")
                X_safe[col] = X_safe[col].astype(str)

        return X_safe

    # ============ COMPONENT 1: FEATURE IMPORTANCE ============

    def _analyze_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance using model-specific method or fallback.

        Supports:
        - Tree-based models (feature_importances_)
        - Linear models (coef_)
        - Fallback: permutation importance

        Returns:
            Dict mapping feature_name -> importance_score (normalized 0-1)
        """
        importances = {}

        # Try tree-based importance
        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
        # Try linear model coefficients
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # For binary classification
            importances = dict(zip(self.feature_names, np.abs(coef)))
        # Fallback to permutation importance
        else:
            importances = self._compute_permutation_importance()

        # Normalize to 0-1 range
        if importances:
            max_imp = max(importances.values())
            if max_imp > 0:
                importances = {k: v / max_imp for k, v in importances.items()}

        return importances

    def _compute_permutation_importance(self) -> Dict[str, float]:
        """
        Compute permutation importance as fallback.

        Returns:
            Dict mapping feature_name -> importance_score
        """
        try:
            # Use safe data for prediction
            X_safe = self._prepare_data_for_prediction(self.X)
            y_pred = self.model.predict(X_safe)
            result = permutation_importance(
                self.model, X_safe, y_pred, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = dict(zip(self.feature_names, result.importances_mean))
            return importances
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            # Return uniform distribution if all else fails
            return {feat: 1.0 / len(self.feature_names) for feat in self.feature_names}

    # ============ COMPONENT 2: GROUP-WISE ANALYSIS ============

    def _analyze_groups(self, y_pred: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze fairness metrics per sensitive group.

        For each unique value in sensitive_attributes, compute:
        - selection_rate
        - true_positive_rate
        - false_positive_rate
        - positive_prediction_distribution

        Returns:
            List of dicts with group analysis
        """
        group_analysis = []

        # Iterate through each sensitive attribute
        for attr_col in self.sensitive_attributes.columns:
            unique_groups = self.sensitive_attributes[attr_col].unique()

            for group_value in unique_groups:
                group_mask = self.sensitive_attributes[attr_col] == group_value
                group_y_true = self.y[group_mask]
                group_y_pred = y_pred[group_mask]
                group_size = group_mask.sum()

                if group_size == 0:
                    continue

                # Compute metrics
                sel_rate = float(np.mean(group_y_pred == 1))

                # True positive rate
                if (group_y_true == 1).sum() > 0:
                    tpr = float(
                        np.sum((group_y_pred == 1) & (group_y_true == 1))
                        / np.sum(group_y_true == 1)
                    )
                else:
                    tpr = 0.0

                # False positive rate
                if (group_y_true == 0).sum() > 0:
                    fpr = float(
                        np.sum((group_y_pred == 1) & (group_y_true == 0))
                        / np.sum(group_y_true == 0)
                    )
                else:
                    fpr = 0.0

                # Generate observation
                observation = self._generate_group_observation(
                    group_value, sel_rate, tpr, fpr
                )

                group_analysis.append(
                    {
                        "sensitive_attribute": attr_col,
                        "group": str(group_value),
                        "group_size": int(group_size),
                        "selection_rate": round(sel_rate, 4),
                        "true_positive_rate": round(tpr, 4),
                        "false_positive_rate": round(fpr, 4),
                        "positive_prediction_count": int(np.sum(group_y_pred == 1)),
                        "observations": observation,
                    }
                )

        return group_analysis

    def _generate_group_observation(
        self, group_value: str, sel_rate: float, tpr: float, fpr: float
    ) -> str:
        """
        Generate human-readable observation for a group.

        Args:
            group_value: Value of the sensitive attribute
            sel_rate: Selection rate for this group
            tpr: True positive rate for this group
            fpr: False positive rate for this group

        Returns:
            String observation
        """
        observations = []

        if sel_rate < 0.3:
            observations.append(
                f"{group_value} receives fewer positive predictions (rate: {sel_rate:.1%})"
            )
        elif sel_rate > 0.7:
            observations.append(
                f"{group_value} receives more positive predictions (rate: {sel_rate:.1%})"
            )

        if tpr < 0.4:
            observations.append(
                f"True positive rate is low for {group_value} ({tpr:.1%})"
            )
        elif tpr > 0.8:
            observations.append(
                f"True positive rate is high for {group_value} ({tpr:.1%})"
            )

        if fpr < 0.2:
            observations.append(f"Low false positive rate for {group_value}")
        elif fpr > 0.5:
            observations.append(f"High false positive rate for {group_value}")

        if not observations:
            observations.append(
                f"Group {group_value} shows standard prediction patterns"
            )

        return "; ".join(observations)

    # ============ COMPONENT 3: PROXY FEATURE DETECTION ============

    def _detect_proxy_features(self) -> List[Dict[str, Any]]:
        """
        Detect features highly correlated with sensitive attributes.

        Uses:
        - Pearson correlation for numeric features
        - Cramér's V for categorical features

        Threshold: correlation > 0.7

        Returns:
            List of proxy features with correlation info
        """
        proxy_features = []

        for attr_col in self.sensitive_attributes.columns:
            sensitive_attr = self.sensitive_attributes[attr_col]

            for feature_col in self.X.columns:
                feature_data = self.X[feature_col]

                # Both numeric
                if pd.api.types.is_numeric_dtype(
                    sensitive_attr
                ) and pd.api.types.is_numeric_dtype(feature_data):
                    correlation = self._pearson_correlation(
                        sensitive_attr, feature_data
                    )
                    if correlation > 0.7:
                        proxy_features.append(
                            {
                                "feature": feature_col,
                                "sensitive_attribute": attr_col,
                                "correlated_with": str(sensitive_attr.name),
                                "correlation_score": round(correlation, 4),
                                "correlation_type": "Pearson",
                                "reason": f"{feature_col} may indirectly encode {attr_col} information",
                            }
                        )

                # Both categorical or at least one is categorical
                elif not pd.api.types.is_numeric_dtype(
                    sensitive_attr
                ) or not pd.api.types.is_numeric_dtype(feature_data):
                    # Encode both as numeric for correlation
                    enc_attr = self._encode_categorical(sensitive_attr)
                    enc_feat = self._encode_categorical(feature_data)

                    correlation = self._pearson_correlation(
                        pd.Series(enc_attr, index=sensitive_attr.index),
                        pd.Series(enc_feat, index=feature_data.index),
                    )

                    if correlation > 0.7:
                        proxy_features.append(
                            {
                                "feature": feature_col,
                                "sensitive_attribute": attr_col,
                                "correlated_with": str(sensitive_attr.name),
                                "correlation_score": round(correlation, 4),
                                "correlation_type": "Encoded-Correlation",
                                "reason": f"{feature_col} may indirectly encode {attr_col} information",
                            }
                        )

        return proxy_features

    def _pearson_correlation(self, s1: pd.Series, s2: pd.Series) -> float:
        """
        Safely compute Pearson correlation.

        Args:
            s1, s2: Pandas Series

        Returns:
            Correlation coefficient (-1 to 1), or 0 if computation fails
        """
        try:
            # Remove NaN values
            mask = ~(s1.isna() | s2.isna())
            if mask.sum() < 2:
                return 0.0
            return float(np.abs(np.corrcoef(s1[mask], s2[mask])[0, 1]))
        except Exception:
            return 0.0

    def _encode_categorical(self, series: pd.Series) -> np.ndarray:
        """
        Encode categorical series as numeric codes.

        Args:
            series: Pandas Series

        Returns:
            Numeric array
        """
        try:
            encoder = LabelEncoder()
            # Handle NaN by replacing with 'UNKNOWN'
            filled_series = series.fillna("UNKNOWN").astype(str)
            return encoder.fit_transform(filled_series)
        except Exception:
            # If encoding fails, return zeros
            return np.zeros(len(series))

    # ============ COMPONENT 4: BIAS CONTRIBUTION HEURISTIC ============

    def _compute_bias_contribution(
        self, feature_importance: Dict[str, float], proxy_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank features by bias contribution.

        bias_contribution = feature_importance * correlation_with_sensitive_attribute

        Returns:
            List of top bias contributors (sorted by contribution score)
        """
        bias_scores = {}

        # Start with feature importance
        for feat, imp in feature_importance.items():
            bias_scores[feat] = imp

        # Boost scores for features that are potential proxies
        proxy_feature_names = {pf["feature"] for pf in proxy_features}
        for feat in proxy_feature_names:
            if feat in bias_scores:
                # Increase contribution score for proxy features
                bias_scores[feat] *= 1.5

        # Filter and sort
        contributors = [
            {
                "feature": feat,
                "importance": round(score, 4),
                "reason": self._generate_feature_reason(
                    feat, feature_importance, proxy_features
                ),
            }
            for feat, score in bias_scores.items()
            if score > 0.1  # Filter weak contributors
        ]

        # Sort by importance descending
        contributors.sort(key=lambda x: x["importance"], reverse=True)
        return contributors[:10]  # Top 10 contributors

    def _generate_feature_reason(
        self,
        feature: str,
        feature_importance: Dict[str, float],
        proxy_features: List[Dict[str, Any]],
    ) -> str:
        """
        Generate human-readable reason for feature's bias contribution.

        Args:
            feature: Feature name
            feature_importance: Dict of all feature importances
            proxy_features: List of detected proxy features

        Returns:
            Explanation string
        """
        reasons = []

        # Check importance
        imp = feature_importance.get(feature, 0)
        if imp > 0.3:
            reasons.append(f"strongly influences predictions")
        elif imp > 0.15:
            reasons.append(f"moderately influences predictions")

        # Check if it's a proxy
        is_proxy = any(pf["feature"] == feature for pf in proxy_features)
        if is_proxy:
            proxies = [pf for pf in proxy_features if pf["feature"] == feature]
            for pf in proxies:
                reasons.append(
                    f"correlated with {pf['sensitive_attribute']} ({pf['correlation_score']:.2f})"
                )

        if not reasons:
            reasons.append("shows differential impact across groups")

        return feature + " " + " and ".join(reasons)

    # ============ COMPONENT 5: SHAP SUPPORT (OPTIONAL) ============

    def compute_shap_values(self) -> Optional[Dict[str, Any]]:
        """
        Compute SHAP values if available.

        Returns:
            Dict with SHAP analysis, or None if SHAP not available
        """
        if not HAS_SHAP:
            return None

        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X)

            # Handle different output formats
            if isinstance(shap_values, list):
                # For binary classification, use class 1
                shap_values = shap_values[1]

            # Compute global importance
            global_importance = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(self.feature_names, global_importance))

            # Normalize
            max_shap = max(shap_importance.values())
            if max_shap > 0:
                shap_importance = {k: v / max_shap for k, v in shap_importance.items()}

            return {
                "method": "SHAP-TreeExplainer",
                "feature_importance": shap_importance,
            }
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None

    # ============ COMPONENT 6: SUMMARY GENERATION ============

    def _generate_summary(
        self,
        bias_contributors: List[Dict[str, Any]],
        group_analysis: List[Dict[str, Any]],
        proxy_features: List[Dict[str, Any]],
    ) -> str:
        """
        Generate human-readable summary of bias findings.

        Args:
            bias_contributors: Top contributing features
            group_analysis: Group-wise fairness metrics
            proxy_features: Detected proxy features

        Returns:
            Summary paragraph
        """
        summary_parts = []

        # Introduce bias finding
        if not bias_contributors:
            return "No significant bias drivers identified in this model."

        # Add top contributor
        top_contrib = bias_contributors[0]
        summary_parts.append(
            f"Bias is primarily influenced by {top_contrib['feature']}, which {top_contrib['reason']}."
        )

        # Add group disparity observation
        if group_analysis:
            group_disparities = self._identify_group_disparities(group_analysis)
            if group_disparities:
                summary_parts.append(group_disparities)

        # Add proxy feature warning
        if proxy_features:
            proxy_count = len(proxy_features)
            summary_parts.append(
                f"Additionally, {proxy_count} feature(s) may indirectly encode sensitive information, "
                f"acting as discriminatory proxies."
            )

        # Add recommendation
        if len(bias_contributors) > 1:
            other_features = [c["feature"] for c in bias_contributors[1:3]]
            summary_parts.append(
                f"Other contributing factors include {', '.join(other_features)}."
            )

        return " ".join(summary_parts)

    def _identify_group_disparities(self, group_analysis: List[Dict[str, Any]]) -> str:
        """
        Identify major disparities between groups.

        Returns:
            String describing disparities
        """
        if len(group_analysis) < 2:
            return ""

        # Find max and min selection rates
        sel_rates = {
            ga["group"]: ga["selection_rate"]
            for ga in group_analysis
            if ga.get("sensitive_attribute")
            == group_analysis[0].get("sensitive_attribute")
        }

        if len(sel_rates) < 2:
            return ""

        max_group = max(sel_rates, key=sel_rates.get)
        min_group = min(sel_rates, key=sel_rates.get)
        disparity = sel_rates[max_group] - sel_rates[min_group]

        if disparity > 0.2:  # >20% difference
            return (
                f"Significant disparity detected: "
                f"{max_group} receives positive predictions at {sel_rates[max_group]:.1%} "
                f"while {min_group} receives {sel_rates[min_group]:.1%}."
            )
        return ""
