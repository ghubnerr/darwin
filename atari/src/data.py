import pickle

PATH = "data.pickle"


class Data:
    def __init__(self, param):
        self.param = param

    def save(self):
        try:
            with open(PATH, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

    @staticmethod
    def load() -> "Data | None":
        try:
            with open(PATH, "rb") as f:
                pickle.load(f)
        except Exception as ex:
            print("Error during loading object (Possibly unsupported):", ex)


data = Data.load()
