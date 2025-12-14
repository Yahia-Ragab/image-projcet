import cv2
import numpy as np
from collections import Counter
import heapq
import math
from ip import IP

class Compress(IP):
    def __init__(self, path, resize_factor=0.9):
        super().__init__(path)
        h, w = self.img.shape[:2]
        new_h, new_w = max(1, int(h * resize_factor)), max(1, int(w * resize_factor))
        self.img = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.gray_img = self._to_gray_uint8()
        self.data = self.gray_img.flatten().tolist()

    def _to_gray_uint8(self):
        if self.img.ndim == 3:
            gray = np.mean(self.img, axis=2)
        else:
            gray = self.img
        return np.clip(gray, 0, 255).astype(np.uint8)

    def huffman(self):
        if not self.data:
            return "", {}
        freq = Counter(self.data)
        if len(freq) == 1:
            sym = next(iter(freq))
            return "0" * len(self.data), {sym: "0"}
        heap = [[f, [s, ""]] for s, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for p in lo[1:]:
                p[1] = "0" + p[1]
            for p in hi[1:]:
                p[1] = "1" + p[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huff_map = {s: c for s, c in heap[0][1:]}
        encoded = "".join(huff_map[v] for v in self.data)
        return encoded, huff_map

    def golomb_rice(self, M=4):
        if M <= 0 or (M & (M - 1)) != 0:
            raise ValueError("M must be power of 2")
        k = int(math.log2(M))
        encoded = []
        for x in self.data:
            q = x // M
            r = x % M
            encoded.append("1" * q + "0" + format(r, f"0{k}b"))
        return encoded

    def arithmetic(self):
        if not self.data:
            return 0.0, {}
        freq = Counter(self.data)
        total = len(self.data)
        probs = {k: v / total for k, v in freq.items()}
        cum = {}
        c = 0.0
        for s in sorted(probs):
            cum[s] = (c, c + probs[s])
            c += probs[s]
        low, high = 0.0, 1.0
        for s in self.data:
            r = high - low
            high = low + r * cum[s][1]
            low = low + r * cum[s][0]
        return (low + high) / 2, cum

    def lzw(self):
        dictionary = {bytes([i]): i for i in range(256)}
        dict_size = 256
        w = bytes()
        result = []
        for v in self.data:
            c = bytes([v])
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
        if w:
            result.append(dictionary[w])
        return result, dict_size

    def rle(self):
        if not self.data:
            return []
        encoded = []
        prev = self.data[0]
        count = 1
        for v in self.data[1:]:
            if v == prev and count < 255:
                count += 1
            else:
                encoded.append((prev, count))
                prev = v
                count = 1
        encoded.append((prev, count))
        return encoded

    def symbol_based(self):
        freq = Counter(self.data)
        symbols = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        mapping = {s: i for i, (s, _) in enumerate(symbols)}
        encoded = [mapping[v] for v in self.data]
        return encoded, mapping

    def bit_plane(self):
        planes = []
        for i in range(8):
            planes.append(((self.gray_img >> i) & 1).astype(np.uint8))
        return planes

    def dct_blocks(self, block_size=8):
        h, w = self.gray_img.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(self.gray_img, ((0, pad_h), (0, pad_w)), mode="edge")
        out = np.zeros_like(padded, dtype=np.float32)
        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = np.float32(padded[i:i+block_size, j:j+block_size])
                out[i:i+block_size, j:j+block_size] = cv2.dct(block)
        return out[:h, :w]

    def predictive(self, mode="left"):
        h, w = self.gray_img.shape
        residual = np.zeros_like(self.gray_img, dtype=np.int16)
        predicted = np.zeros_like(self.gray_img, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if mode == "left":
                    p = self.gray_img[i, j-1] if j > 0 else 0
                elif mode == "top":
                    p = self.gray_img[i-1, j] if i > 0 else 0
                else:
                    l = self.gray_img[i, j-1] if j > 0 else 0
                    t = self.gray_img[i-1, j] if i > 0 else 0
                    p = (int(l) + int(t)) // 2
                predicted[i, j] = p
                residual[i, j] = int(self.gray_img[i, j]) - int(p)
        return residual, predicted

    def wavelet(self, level=1):
        img = self.gray_img.astype(np.float32)
        for _ in range(level):
            h, w = img.shape
            h2, w2 = h // 2, w // 2
            out = np.zeros_like(img)
            for i in range(h2):
                for j in range(w):
                    out[i, j] = (img[2*i, j] + img[2*i+1, j]) / 2
                    out[h2+i, j] = (img[2*i, j] - img[2*i+1, j]) / 2
            tmp = out.copy()
            for i in range(h):
                for j in range(w2):
                    out[i, j] = (tmp[i, 2*j] + tmp[i, 2*j+1]) / 2
                    out[i, w2+j] = (tmp[i, 2*j] - tmp[i, 2*j+1]) / 2
            img = out
        return img

    def get_compression_stats(self, encoded_data, original_bits=None):
        if original_bits is None:
            original_bits = len(self.data) * 8
        if isinstance(encoded_data, str):
            compressed_bits = len(encoded_data)
        elif isinstance(encoded_data, list):
            if encoded_data and isinstance(encoded_data[0], str):
                compressed_bits = sum(len(x) for x in encoded_data)
            elif encoded_data and isinstance(encoded_data[0], tuple):
                compressed_bits = len(encoded_data) * 16
            elif encoded_data and isinstance(encoded_data[0], int):
                bits = max(9, max(encoded_data).bit_length())
                compressed_bits = len(encoded_data) * bits
            else:
                compressed_bits = original_bits
        elif isinstance(encoded_data, np.ndarray):
            compressed_bits = encoded_data.size * 32
        else:
            compressed_bits = original_bits
        ratio = original_bits / compressed_bits if compressed_bits else 0
        saving = ((original_bits - compressed_bits) / original_bits * 100) if original_bits else 0
        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": round(ratio, 2),
            "space_saving": round(saving, 2),
        }

