"""
BBPE (Byte-Level Byte Pair Encoding) 简易算法实现

核心原理：
考虑到标准 BPE 在处理包含大量未知字符（尤其是多语言或特殊符号）时可能导致词表极其庞大（因为要包含所有的基本 Unicode 字符），
BBPE 的核心思想是：将输入文本先编码为字节 (bytes, utf-8下共256个可能值)，然后直接在“字节序列”上应用 BPE 算法。
由于基础词汇表只有 256 个字节片段，它可以天然地表示任何可能出现的字符串，完全消除了 OOV (未登录词) 问题。

GPT-2 等模型使用了一种技巧：由于有很多字节并不是完全可打印的控制字符，直接处理会有很多问题。
因此首先将 256 个 byte 映射到了 256 个特定的可见 Unicode 字符上。
这样在保留字节级处理优势的同时，就可以完全复用传统的基于字符串的 BPE 实现。
"""

from collections import defaultdict


def bytes_to_unicode():
    """
    返回一个字典，将 byte (0-255) 映射为某个可见的 Unicode 字符。
    这是因为部分 byte 并不是有效的可见字符（例如控制字符）。
    GPT-2/RoBERTa 的做法是将原本就是可见字符的 byte 直接映射为它本身，其余的映射到额外的 Unicode 区间。
    """
    # 常用可见字符的 ASCII 范围
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 补充 256 个 ASCII 码里面不可见的字符映射
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # bs 包含了 0-255 的所有 byte 值，cs 是它们对应的可见 unicode 码点
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}


def get_stats(vocab):
    """
    计算当前词表中所有相邻符号对的出现频率。
    """
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pairs[pair] += freq
    return pairs


def merge_vocab(pair, v_in):
    """
    将词表中最频繁的符号对合并为一个新的符号。
    """
    v_out = {}
    for word_tuple, freq in v_in.items():
        new_word_tuple = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                new_word_tuple.append("".join(pair))
                i += 2
            else:
                new_word_tuple.append(word_tuple[i])
                i += 1
        v_out[tuple(new_word_tuple)] = freq
    return v_out


def build_bbpe_vocab(words, num_merges):
    """
    基于提供的语料库构建 BBPE 词表。
    """
    vocab = defaultdict(int)

    # 1. 初始化词表：与传统 BPE 不同，这里我们将词首先转化为字节，再映射为可见字符
    # 比如一般的中文1个字会被转化为3个UTF-8字节，也就是3个初始可见符
    print("【初始字节级切分】(片段展示):")
    for word in words:
        word = word.strip()
        if not word:
            continue
        # 编码为 utf-8 bytes, 然后把每个 byte 映射成唯一的可见字符
        byte_chars = [byte_encoder[b] for b in word.encode("utf-8")]

        # 加上 </w> 标志结尾 （在GPT-2等真实实现中其实通常基于前缀空格拆分，这里为了简单对比沿用加入结尾符）
        word_tuple = tuple(byte_chars + ["</w>"])
        vocab[word_tuple] += 1
        
    # 为避免打印太多，这里只打印词汇表的一部分
    count = 0
    for w in set(words):
        if count >= 3:
            break
        word_tuple = tuple([byte_encoder[b] for b in w.encode("utf-8")] + ["</w>"])
        print(f"  '{w}' -> UTF-8 bytes: {list(w.encode('utf-8'))} \n       -> BBPE 初始状态: {word_tuple}")
        count += 1
    print("-" * 50)

    bpe_codes = {}
    # 2. 迭代合并最频繁的对
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        bpe_codes[best_pair] = i

        print(
            f"第 {i+1:2d} 次合并: 组合 '{best_pair[0]}' + '{best_pair[1]}' -> '{''.join(best_pair)}' (频次: {pairs[best_pair]})"
        )
        vocab = merge_vocab(best_pair, vocab)

    print("-" * 50)
    print("【最终状态词表】(BBPE拆分合并结果):")
    for w, f in vocab.items():
        print(f"  {' '.join(w)} : {f}")

    return bpe_codes, vocab


def encode_word(word, bpe_codes):
    """
    推理阶段：利用训练好的 BBPE 合并规则对新词进行分词
    """
    if not word:
        return []

    # 转化为字节并做可见字符映射
    word_chars = [byte_encoder[b] for b in word.encode("utf-8")] + ["</w>"]

    while len(word_chars) > 1:
        pairs = [(word_chars[i], word_chars[i + 1]) for i in range(len(word_chars) - 1)]
        pair_to_merge = None
        min_rank = float("inf")

        for pair in pairs:
            if pair in bpe_codes and bpe_codes[pair] < min_rank:
                min_rank = bpe_codes[pair]
                pair_to_merge = pair

        if pair_to_merge is None:
            # 已经没有可以根据规则合并的组合对了
            break

        new_word_chars = []
        i = 0
        while i < len(word_chars):
            if (
                i < len(word_chars) - 1
                and (word_chars[i], word_chars[i + 1]) == pair_to_merge
            ):
                new_word_chars.append("".join(pair_to_merge))
                i += 2
            else:
                new_word_chars.append(word_chars[i])
                i += 1
        word_chars = new_word_chars

    return word_chars


if __name__ == "__main__":
    # ==== 测试数据 ===
    # 包含了中文和英文，体现 BBPE 对跨语言和各种字符的强大包容性（不存在 OOV）
    train_corpus = ["你好"] * 8 + ["你好啊"] * 4 + ["测试"] * 5 + ["hello"] * 6

    print("=" * 15, "BBPE (字节级BPE) 算法原理演示", "=" * 15)
    print("语料情况:", set(train_corpus))
    
    # 设定最高执行 15 次合并
    bpe_codes, final_vocab = build_bbpe_vocab(train_corpus, 15)

    print("\n" + "=" * 15, "第二阶段: 推理分词测试", "=" * 15)
    # 含有未见过的 OOV 词：你好呀, 测试版, hi, helloworld
    test_words = ["你好", "你好呀", "测试版", "hello", "helloworld", "hi"]

    for word in test_words:
        encoded_chars = encode_word(word, bpe_codes)

        # 尝试将 BBPE token 逆向解码，还原为原本的字符串（方便理解）
        # 注意：如果 BBPE token 刚好截断在 UTF-8 的中间字节，单纯把它解码时会报错
        # 这也正是 BBPE 的特点（有些 token 是无独立意义的被截断的字节片段），这里以 errors='replace' 来显示为 
        decoded_tokens = []
        for token in encoded_chars:
            token_clean = token.replace("</w>", "")
            if token_clean:
                try:
                    # 映射回真实 byte 值
                    raw_bytes = bytes([byte_decoder[c] for c in token_clean])
                    decoded_string = raw_bytes.decode("utf-8", errors="replace")
                    if not decoded_string.strip() and len(decoded_string) > 0:
                         decoded_tokens.append("<space_or_control>")
                    else:
                         decoded_tokens.append(decoded_string)
                except KeyError:
                    decoded_tokens.append("?")
            else:
                decoded_tokens.append("</w>")

        print(f"输入词: '{word:<8}'")
        print(f" -> 编码映射 Tokens : {encoded_chars}")
        print(f" -> 文本解码 Tokens : {decoded_tokens}")
        print("-" * 30)

    print("\n观察结果: ")
    print(" 1. 任何看似生僻或者不在训练语料中的字符 (如 '呀', 'hi') 都绝对不会出现 OOV 无法编码的问题，它们被拆分成了一个个基础字节进行表示。")
    print(" 2. '你好呀' 前面匹配了训练时的 '你好', 最终的 '呀' 则被拆成了它对应的3个 utf-8 byte 组成的 token。")
    print(" 3. 这种处理方式被广泛应用在 GPT-2、RoBERTa、LLaMA 等主流 LLM 的 tokenizer 里。")
    print("=" * 55)
