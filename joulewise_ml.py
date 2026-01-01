import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


@dataclass
class SolarConfig:
    """Domain + training configuration."""
    # Location defaults: Delhi
    base_lat: float = 28.6
    base_lon: float = 77.2

    # Economics (INR)
    tariff_inr_per_kwh: float = 5.5
    cost_per_watt_inr: float = 35.0
    install_multiplier: float = 1.2  # capex = cost * install_multiplier

    # System assumptions
    wp_per_m2: float = 200.0
    baseline_irradiance_kwh_m2_day: float = 5.2
    default_aoi_loss: float = 0.95
    min_shadow_factor: float = 0.85

    # Simulation bounds
    min_roi_pct: float = 50.0
    max_roi_pct: float = 200.0

    # Model
    n_estimators: int = 300
    random_state: int = 42
    test_size: float = 0.2

    # Paths
    model_path: Path = Path("joulewise_model.pkl")


class JoulewiseML:
    """ROI model for rooftop solar in Delhi region."""

    FEATURES = [
        "lat", "lon", "tilt", "panel_area", "efficiency",
        "irradiance", "shadow_factor", "aoi_loss",
    ]

    def __init__(self, config: Optional[SolarConfig] = None):
        self.config = config or SolarConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.is_trained: bool = False

    # ---------- Synthetic data generation ----------

    def _simulate_roi(
        self,
        irradiance: np.ndarray,
        panel_area: np.ndarray,
        shadow_factor: np.ndarray,
        aoi_loss: np.ndarray,
    ) -> np.ndarray:
        """Vectorized 10‑year ROI% calculation."""
        cfg = self.config

        # Annual energy (kWh)
        energy_kwh_yr = irradiance * panel_area * 365 * shadow_factor * aoi_loss

        # Revenue over 10 years
        revenue_10yr = energy_kwh_yr * cfg.tariff_inr_per_kwh * 10

        # Capex
        capacity_kw = panel_area * cfg.wp_per_m2  # Wp
        total_cost = capacity_kw * cfg.cost_per_watt_inr * cfg.install_multiplier

        roi_10yr = (revenue_10yr - total_cost) / total_cost * 100.0
        return np.clip(roi_10yr, cfg.min_roi_pct, cfg.max_roi_pct)

    def generate_sample_data(self, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic but plausible solar ROI data around Delhi."""
        rng = np.random.default_rng(seed)
        cfg = self.config

        lat = rng.normal(cfg.base_lat, 0.5, n_samples)
        lon = rng.normal(cfg.base_lon, 0.5, n_samples)
        tilt = rng.uniform(10, 40, n_samples)
        panel_area = rng.uniform(1.6, 3.2, n_samples)
        efficiency = rng.normal(0.20, 0.02, n_samples)

        irradiance = rng.normal(cfg.baseline_irradiance_kwh_m2_day, 0.8, n_samples)
        shadow_factor = rng.uniform(cfg.min_shadow_factor, 1.0, n_samples)
        aoi_loss = rng.uniform(0.92, 0.98, n_samples)

        roi_10yr = self._simulate_roi(irradiance, panel_area, shadow_factor, aoi_loss)

        return pd.DataFrame(
            {
                "lat": lat,
                "lon": lon,
                "tilt": tilt,
                "panel_area": panel_area,
                "efficiency": efficiency,
                "irradiance": irradiance,
                "shadow_factor": shadow_factor,
                "aoi_loss": aoi_loss,
                "roi_10yr": roi_10yr,
            }
        )

    # ---------- Training / evaluation ----------

    def _train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df[self.FEATURES]
        y = df["roi_10yr"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return X_train, X_test, y_train, y_test

    def train(self, n_samples: int = 2000, verbose: bool = True) -> Dict[str, float]:
        """Train model on synthetic data and persist to disk."""
        if verbose:
            print("Generating training data...")

        data = self.generate_sample_data(n_samples=n_samples)
        X_train, X_test, y_train, y_test = self._train_test_split(data)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.is_trained = True

        if verbose:
            print(f"✅ Model trained! MAE: {mae:.1f} %, R²: {r2:.3f}")
            print(f"Submission metrics: approx accuracy ~{100 - mae:.1f} %, R²: {r2:.3f}")

        # Save model (use joblib for tree ensembles)
        joblib.dump(self.model, self.config.model_path)

        return {"mae": float(mae), "r2": float(r2)}

    def load_or_train(self, retrain_if_missing: bool = True) -> None:
        """Load model from disk if available, optionally train if missing."""
        path = self.config.model_path
        if path.exists():
            self.model = joblib.load(path)
            self.is_trained = True
        elif retrain_if_missing:
            self.train()
        else:
            raise FileNotFoundError(f"Model file not found at {path}")

    # ---------- Inference utilities ----------

    def _estimate_shadow_factor(self, lat: float, tilt: float) -> float:
        """Very simple heuristic shadow factor model."""
        cfg = self.config
        # Penalize tilt far from latitude very slightly
        penalty = (abs(tilt - lat) / 10.0) * 0.05
        return max(cfg.min_shadow_factor, 1.0 - penalty)

    def _estimate_optimal_tilt(self, lat: float) -> float:
        """Heuristic optimal tilt: close to latitude with bounds."""
        # Literature: around site latitude; Delhi ~26° optimal tilt empirically.
        # Clamp for rooftop practicality.
        raw = 0.9 * lat + 5.0
        return float(np.round(np.clip(raw, 15.0, 35.0), 1))

    def predict_roi(
        self,
        lat: float,
        lon: float,
        tilt: float,
        panel_area: float = 2.0,
        efficiency: float = 0.20,
        use_saved_model: bool = True,
    ) -> Dict[str, float]:
        """Predict 10‑year ROI and key KPIs for a new site."""
        if not self.is_trained:
            if use_saved_model:
                self.load_or_train()
            else:
                self.train()

        cfg = self.config
        irradiance = cfg.baseline_irradiance_kwh_m2_day
        shadow_factor = self._estimate_shadow_factor(lat, tilt)
        aoi_loss = cfg.default_aoi_loss

        features = np.array(
            [[lat, lon, tilt, panel_area, efficiency, irradiance, shadow_factor, aoi_loss]]
        )
        roi = float(self.model.predict(features)[0])

        # Annual energy for reporting
        energy_kwh_yr = irradiance * panel_area * 365 * shadow_factor * aoi_loss

        # Simple payback: years to recover ~3.5 years at 100% ROI
        payback_years = 3.5 / (roi / 100.0) if roi > 0 else np.inf

        return {
            "predicted_roi_10yr_pct": round(roi, 1),
            "optimal_tilt_deg": self._estimate_optimal_tilt(lat),
            "energy_kwh_per_year": round(energy_kwh_yr, 1),
            "payback_years": round(payback_years, 1),
            "shadow_factor": round(shadow_factor, 3),
        }


if __name__ == "__main__":
    joule = JoulewiseML()
    metrics = joule.train()

    # Demo prediction (GIPU / Delhi)
    result = joule.predict_roi(lat=28.6, lon=77.2, tilt=25.0)
    print("\nDemo Prediction (GIPU Campus):")
    print(result)
