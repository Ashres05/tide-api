from model.train_marketshare_artifacts import main as train_marketshare_artifacts

def train_model_main() -> None:
    """
    Calls the main function from train_marketshare_artifacts.py to train the marketshare artifacts.
    Will create necessary parquets and JSON files for model training.
    """
    # TODO: Update the data directory to the new data.

    # Train the model.
    train_marketshare_artifacts()


# TODO: Add a cron job to run this script every week after update_sqlite.py has been run.
if __name__ == "__main__":
    train_model_main()
