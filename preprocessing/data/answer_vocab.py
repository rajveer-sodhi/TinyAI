class AnswerVocab:
    def __init__(self):
        # allowable output characters
        base_chars = list("0123456789-")  # integers only
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.tokens = specials + base_chars
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(self, answer, max_len):
        """
        Convert a numeric answer like '152' into:
        [<bos>, '1', '5', '2', <eos>, <pad>, ...]
        """
        text = str(answer).strip()
        chars = list(text)

        ids = [self.bos_id] + [self.stoi.get(c, self.unk_id) for c in chars] + [self.eos_id]

        # pad or trim
        ids = ids[:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        tokens = [self.itos.get(i, "") for i in ids]
        out = []
        for t in tokens:
            if t in ("<pad>", "<bos>", "<eos>", "<unk>"):
                continue
            out.append(t)
        return "".join(out)