from pathlib import Path


def list_available_models(models_dir="onnx_models"):
    model_dir = Path(models_dir)
    if not model_dir.exists():
        raise RuntimeError(f"Models directory not found: {model_dir}")

    models = sorted(p.stem for p in model_dir.glob("*.onnx"))
    if not models:
        raise RuntimeError(f"No .onnx models found in: {model_dir}")
    return models


def choose_model(models_dir="onnx_models"):
    models = list_available_models(models_dir)

    print("\nAvailable models:")
    for idx, model_name in enumerate(models, start=1):
        print(f"{idx}. {model_name}")

    while True:
        choice = input("Enter model number: ").strip()
        try:
            number = int(choice)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if 1 <= number <= len(models):
            model_name = models[number - 1]
            model_path = str(Path(models_dir) / f"{model_name}.onnx")
            return model_name, model_path

        print(f"Please enter a number between 1 and {len(models)}.")
