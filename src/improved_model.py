from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, fbeta_score, precision_score, recall_score,
    f1_score, precision_recall_curve, classification_report
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.logger import Logger  # adjust import path to match your project structure


class ImprovedModelTrainer:
    def __init__(self, log_file: str, random_state: int = 42):
        self.logger = Logger(log_file)
        self.random_state = random_state
        self.pipeline = None
        self.search = None
        self.best_threshold = 0.5
        self.logger.info("ImprovedModelTrainer initialized")

    def build_pipeline(self):
        """
        Builds the ExtraTrees pipeline: feature selection -> SMOTE -> ExtraTreesClassifier.
        """
        self.logger.info("Building ExtraTrees pipeline")

        self.pipeline = Pipeline([
            (
                "selector",
                SelectFromModel(
                    ExtraTreesClassifier(
                        n_estimators=100,
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    threshold="median"
                )
            ),
            (
                "sampler",
                SMOTE(random_state=self.random_state)
            ),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                )
            )
        ])

        self.logger.info("Pipeline built successfully")
        return self.pipeline

    def get_param_distributions(self):
        """
        Returns the hyperparameter search space for RandomizedSearchCV.
        """
        param_dist = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 15, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__class_weight": [None, "balanced"]
        }
        self.logger.info(f"Parameter search space defined: {param_dist}")
        return param_dist

    def run_search(self, x_train, y_train, n_iter=15, cv=3, beta=0.5, verbose=2):
        """
        Runs RandomizedSearchCV on the pipeline using an F-beta scorer (beta < 1
        weights precision more heavily than recall), since this project prioritizes
        minimizing false positives.
        """
        if self.pipeline is None:
            self.build_pipeline()

        param_dist = self.get_param_distributions()

        fbeta_scorer = make_scorer(fbeta_score, beta=beta)

        self.logger.info(
            f"Starting RandomizedSearchCV (n_iter={n_iter}, cv={cv}, scoring=fbeta[beta={beta}])"
        )
        self.logger.info(f"Training data shape: {x_train.shape}")

        self.search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=fbeta_scorer,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=verbose
        )

        self.search.fit(x_train, y_train)

        self.logger.info(f"Search finished. Best F{beta}-score: {self.search.best_score_:.4f}")
        self.logger.info(f"Best params: {self.search.best_params_}")

        return self.search

    def get_selected_features(self, x_train):
        """
        Returns the list of features selected by SelectFromModel in the best estimator.
        """
        if self.search is None:
            raise RuntimeError("run_search() must be called before get_selected_features()")

        best_selector = self.search.best_estimator_.named_steps["selector"]
        selected_mask = best_selector.get_support()
        selected_features = x_train.columns[selected_mask].tolist()

        self.logger.info(
            f"Selected {len(selected_features)} / {x_train.shape[1]} features: {selected_features}"
        )
        return selected_features

    def get_best_model(self):
        """
        Returns the best fitted pipeline found by the search.
        """
        if self.search is None:
            raise RuntimeError("run_search() must be called before get_best_model()")
        return self.search.best_estimator_

    def tune_threshold_for_precision(self, x_val, y_val, target_precision=0.85):
        """
        Finds the lowest probability threshold that achieves at least the target
        precision on the validation/test set, using the best fitted model.
        """
        if self.search is None:
            raise RuntimeError("run_search() must be called before tune_threshold_for_precision()")

        self.logger.info(f"Tuning threshold for target precision >= {target_precision}")

        model = self.get_best_model()
        y_proba = model.predict_proba(x_val)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # precision_recall_curve returns one more precision/recall point than thresholds
        candidates = [
            (t, p, r) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds)
            if p >= target_precision
        ]

        if not candidates:
            self.logger.info(
                f"No threshold achieves precision >= {target_precision}. "
                f"Keeping default threshold 0.5"
            )
            self.best_threshold = 0.5
        else:
            # Among thresholds meeting the precision target, pick the one with highest recall
            best = max(candidates, key=lambda c: c[2])
            self.best_threshold, best_precision, best_recall = best
            self.logger.info(
                f"Selected threshold: {self.best_threshold:.3f} "
                f"(precision={best_precision:.4f}, recall={best_recall:.4f})"
            )

        return self.best_threshold

    def evaluate(self, x_test, y_test, threshold=None):
        """
        Evaluates the best model on the test set at the given threshold
        (defaults to the tuned threshold, or 0.5 if not tuned).
        """
        if self.search is None:
            raise RuntimeError("run_search() must be called before evaluate()")

        threshold = threshold if threshold is not None else self.best_threshold

        model = self.get_best_model()
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        self.logger.info(f"Evaluation at threshold={threshold:.3f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1: {f1:.4f}")
        self.logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_pred": y_pred
        }