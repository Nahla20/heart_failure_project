from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.visualization import create_all_plots
from src.model_training import train_model
from src.evaluation import evaluate_model

df = load_data("data/heart.csv")

df = create_features(df)

# Save all plots
create_all_plots(df)

# Train model
model, X_test, y_test = train_model(df)

# Evaluate
evaluate_model(model, X_test, y_test)