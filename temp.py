from src.cryptography.crypto_models import KMACTag

tag = KMACTag(
    tag=b"\x01" * 16
)

print(len(tag.to_bits()))