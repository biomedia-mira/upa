import numpy as np


class HistogramMatchEstimator:
    def fit(
        self, adaptation_set_predictions: np.ndarray, reference_predictions: np.ndarray
    ):
        """
        Fits a linear interpolator to match the cumulative distributions of the reference and adaptation
        set distributions
        """
        # Get both cumulative distribution
        (
            adaptation_set_quantiles,
            self.orig_adaptation_set_values,
        ) = self.get_cumulative_density(adaptation_set_predictions)
        ref_quantiles, ref_values = self.get_cumulative_density(reference_predictions)

        # Match both observed cdf by linear interpolation
        self.matched_adaptation_set_values = np.interp(
            adaptation_set_quantiles, ref_quantiles, ref_values
        )

    def predict(self, test_ood_prediction: np.ndarray):
        """
        Adapts new test prediction by applying the fitted linear interpolator
        """
        return np.interp(
            test_ood_prediction,
            self.orig_adaptation_set_values,
            self.matched_adaptation_set_values,
        )

    def get_cumulative_density(self, observations: np.ndarray):
        """
        Returns empirical cumulative distribution function based on array of observations.
        """
        values, counts = np.unique(observations.ravel(), return_counts=True)
        quantiles = np.cumsum(counts) / observations.size
        return quantiles, values


def align_predictions(
    reference_df, df_for_alignment, df_for_testing, sorted_preds_columns=["pred1"]
):
    if len(sorted_preds_columns) == 1:
        est = HistogramMatchEstimator()
        est.fit(df_for_alignment.pred1.values, reference_df.pred1.values)
        transformed_pred1 = est.predict(df_for_testing.pred1.values).reshape(-1, 1)
        transformed_preds = np.concatenate(
            [1 - transformed_pred1, transformed_pred1], 1
        )
    else:
        transformed_preds = np.zeros((len(df_for_testing), len(sorted_preds_columns)))
        for i, column in enumerate(sorted_preds_columns):
            est = HistogramMatchEstimator()
            est.fit(df_for_alignment[column].values, reference_df[column].values)
            transformed_preds[:, i] = est.predict(df_for_testing[column].values)
    return transformed_preds
