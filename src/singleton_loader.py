import threading
from tokenizers import Tokenizer
import sent2vec

class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load the tokenizer and model
        tokenizer_path = "./src/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.sent2vec_model = sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model("./src/my_model.bin")

