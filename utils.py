import joblib
def load_object(file_path):
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
        