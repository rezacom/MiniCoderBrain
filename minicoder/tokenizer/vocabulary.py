"""واژگان ثابت: هر بایت همان شناسهٔ خودش را دارد."""

BYTE_VOCAB_SIZE = 256
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259
VOCAB_SIZE = 260

SPECIAL_TOKENS = {
    "PAD": PAD_ID,
    "BOS": BOS_ID,
    "EOS": EOS_ID,
    "UNK": UNK_ID,
}
ID_TO_SPECIAL = {token_id: f"<{name}>" for name, token_id in SPECIAL_TOKENS.items()}
