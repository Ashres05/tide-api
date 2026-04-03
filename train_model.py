# TODO: Train and store models with optimized hyperparameters

def train_marketshare_model() -> None:
    # TODO: Call train_marketshare_artifacts.py.
    # TODO: Update parquet based on latest marketshare data.
    # TODO: Train model with optimized hyperparameters.
    # train_marketshare_artifacts.py will automatically save the model locally using joblib.
    pass


def train_decay_model() -> None:
    # TODO: Call all_data_archetypes_simulator.py with set parquet.

    # TODO: Save model locally.
    save_model()
    pass


# TODO: Store model locally using joblib for future use.
def save_model() -> None:
    pass


def train_model_main() -> None:
    train_marketshare_model()
    train_decay_model()


# TODO: Add a cron job to run this script every week after update_sqlite.py has been run.
if __name__ == "__main__":
    train_model_main()
