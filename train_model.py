"""Train and store models with optimized hyperparameters."""
from models.all_data_archetypes_simulator import main as train_archetype_simulator
from models.train_marketshare_artifacts import main as train_marketshare_artifacts

def train_marketshare_model() -> None:
    """
    Full 75k marketshare train/export (see ``models/train_marketshare_artifacts.py``).

    Already persists under ``<repo>/artifacts_75k/``:
    - joblib: ``production_lgbm_75k.pkl``, ``production_prophet_models_75k.pkl``,
      ``production_spike_engine.pkl``, and optionally ``kmeans_archetype_75k.pkl``
    - parquet: ``df_full.parquet``, ``actuals_2026.parquet``
    - JSON: ``metadata.json``, profiles, DNA, coefficients, etc.
    """
    train_marketshare_artifacts()


def train_decay_model() -> None:
    """
    Refresh archetype API artifacts (parquets + JSON under ``archetypes_artifacts/``) from
    ``models/all_releases_18_25_compressed.parquet``. Same entrypoint as ``main()`` in
    ``models/all_data_archetypes_simulator.py``.
    """
    train_archetype_simulator()


def train_model_main() -> None:
    train_marketshare_model()
    train_decay_model()


# TODO: Add a cron job to run this script every week after update_sqlite.py has been run.
if __name__ == "__main__":
    train_model_main()
