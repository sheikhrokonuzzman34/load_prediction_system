from django.core.management.base import BaseCommand

from prediction_app.utils.model_trainer import ModelTrainer


class Command(BaseCommand):
    help = "Train load forecasting models from the CSV dataset"

    def add_arguments(self, parser):
        parser.add_argument("--csv", default="data/load_data.csv", help="Path to the raw load CSV file")
        parser.add_argument(
            "--xgboost-only",
            action="store_true",
            help="Train only the production XGBoost model and skip optional LSTM",
        )

    def handle(self, *args, **options):
        trainer = ModelTrainer()
        if options["xgboost_only"]:
            _, metrics = trainer.train_xgboost_model(options["csv"])
        else:
            _, metrics = trainer.train_all_available_models(options["csv"])

        self.stdout.write(self.style.SUCCESS("Training completed successfully"))
        self.stdout.write(str(metrics))
