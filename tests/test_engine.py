import unittest
import numpy as np
from neural_playground.engine import MLP, make_dataset


class DatasetTests(unittest.TestCase):
    def test_dataset_shapes_and_reproducibility(self):
        for kind in ("xor", "circles", "spiral", "moons", "blobs"):
            X1, y1 = make_dataset(kind, 100, 23, 0.1)
            X2, y2 = make_dataset(kind, 100, 23, 0.1)
            self.assertEqual(X1.shape, (100, 2))
            self.assertEqual(y1.shape, (100,))
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)
            self.assertTrue(set(np.unique(y1)).issubset({0, 1}))


class MLPTests(unittest.TestCase):
    def test_forward_shapes_and_probability_range(self):
        model = MLP(hidden=[6, 4], seed=11)
        self.assertEqual(model.layer_sizes, [2, 6, 4, 1])
        probabilities = model.predict_proba(np.array([[0.1, 0.2], [-0.3, 0.5]]))
        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(model.parameter_count(), 6 * 3 + 4 * 7 + 5)

    def test_all_optimizers_make_finite_updates(self):
        X, y = make_dataset("xor", 80, 7, 0.1)
        for optimizer in ("sgd", "momentum", "adam"):
            model = MLP(hidden=[8, 8], optimizer=optimizer, seed=9)
            before = [w.copy() for w in model.weights]
            metrics = model.train_epoch(X, y, batch_size=20)
            self.assertTrue(any(np.any(a != b) for a, b in zip(before, model.weights)))
            self.assertTrue(np.isfinite(metrics.loss))
            self.assertTrue(0 <= metrics.accuracy <= 1)
            self.assertTrue(all(np.isfinite(w).all() for w in model.weights))

    def test_loss_improves_on_xor(self):
        X, y = make_dataset("xor", 240, 42, 0.08)
        model = MLP(hidden=[8, 8], activation="tanh", optimizer="adam", seed=42)
        before = model.evaluate(X, y).loss
        model.fit(X, y, epochs=90, batch_size=24)
        after = model.evaluate(X, y).loss
        self.assertLess(after, before * 0.75)

    def test_gradient_matches_finite_difference(self):
        X = np.array([[0.21, -0.34]])
        y = np.array([1])
        model = MLP(hidden=[3], activation="tanh", optimizer="sgd", seed=3)
        analytic = model._batch_gradients(X, y)[0][0, 0]
        original = model.weights[0][0, 0]
        epsilon = 1e-5
        model.weights[0][0, 0] = original + epsilon
        high = model.evaluate(X, y).loss
        model.weights[0][0, 0] = original - epsilon
        low = model.evaluate(X, y).loss
        model.weights[0][0, 0] = original
        numeric = (high - low) / (2 * epsilon)
        self.assertAlmostEqual(analytic, numeric, delta=1e-5)

    def test_bad_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            MLP(hidden=[1, 8])
        with self.assertRaises(ValueError):
            make_dataset("unknown", 20)


if __name__ == "__main__":
    unittest.main()
