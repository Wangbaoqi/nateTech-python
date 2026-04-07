"""
BPE (Byte Pair Encoding) 简易算法实现

核心原理：
1. 初始状态下，将词汇表中的每个词拆分成单个字符，并在词尾添加特殊标记 `</w>` 以区分词的边界。
2. 统计当前所有符号（字符或字符级片段）相邻构成的字符对（pair）在语料库中的出现频率。
3. 找到频率最高的那一对，将它们合并成一个新的片段。
4. 重复步骤2和3，直到达到预设的合并次数（num_merges）或达到预设的词表大小。
5. 推理（分词）时，根据训练阶段记录的合并规则和合并顺序，将新单词按照同样的规则进行合并，得到BPE token。这样可以有效处理未登录词（OOV）。
"""

from collections import defaultdict


def get_stats(vocab):
    """
    计算当前词表中所有相邻符号对的出现频率。
    """
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        # word_tuple 形式如 ('l', 'o', 'w', '</w>')
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
            # 如果匹配到了目标符号对，则合并它们
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                new_word_tuple.append("".join(pair))
                i += 2
            else:
                # 否则保留原有符号
                new_word_tuple.append(word_tuple[i])
                i += 1
        v_out[tuple(new_word_tuple)] = freq
    return v_out


def build_bpe_vocab(corpus, num_merges):
    """
    基于提供的语料库构建BPE词表。
    """
    vocab = defaultdict(int)
    # 1. 初始化词表：拆分为单个字符序列并加上结束符 </w>
    for word in corpus:
        word = word.strip()
        if not word:
            continue
        word_tuple = tuple(list(word) + ["</w>"])
        vocab[word_tuple] += 1

    print("【初始状态词表】(单词及频次):")
    for w, f in vocab.items():
        print(f"  {w} : {f}")
    print("-" * 50)

    # 记录合并的规则和它们被学习到的相对顺序 (用来在推理时决定优先合并谁)
    bpe_codes = {}

    # 2. 迭代合并最频繁的对
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            # 如果没有可以合并的字符对了（每个词都被合并成了一个整体），则提前跳出
            break

        # 找到当前迭代中频率最高的对
        best_pair = max(pairs, key=pairs.get)
        bpe_codes[best_pair] = i
        print(f"第 {i+1:2d} 次合并: best_pair: {best_pair}")
        print(f"第 {i+1:2d} 次合并: bpe_codes: {bpe_codes}")

        print(
            f"第 {i+1:2d} 次合并: 组合 '{best_pair[0]}' + '{best_pair[1]}' -> '{''.join(best_pair)}' (出现总频次: {pairs[best_pair]})"
        )

        # 在词表中执行合并
        vocab = merge_vocab(best_pair, vocab)

    print("-" * 50)
    print("【最终状态词表】(BPE拆分结果):")
    for w, f in vocab.items():
        print(f"  {w} : {f}")

    return bpe_codes, vocab


def encode_word(word, bpe_codes):
    """
    推理阶段：使用训练好的BPE规则对单个词进行编码（分词）。
    """
    if not word:
        return []

    # 初始化：拆分成单个字符，并加上结束符
    word_chars = list(word) + ["</w>"]

    while len(word_chars) > 1:
        # 获取当前所有的相邻对
        pairs = [(word_chars[i], word_chars[i + 1]) for i in range(len(word_chars) - 1)]

        # 找到当前所有的对中，在训练时最早被合并的那一对（即 rank/索引 最小的）
        pair_to_merge = None
        min_rank = float("inf")
        for pair in pairs:
            if pair in bpe_codes and bpe_codes[pair] < min_rank:
                min_rank = bpe_codes[pair]
                pair_to_merge = pair

        # 如果当前单词中没有在合并词表中的符号对，则结束合并（得到最终BPE结果）
        if pair_to_merge is None:
            break

        # 在当前单词中合并具有最高优先级的组合对
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
    # ==== 测试数据准备 ====
    # 使用由 Sennrich 等人(2016)论文中提出的经典极简例子
    corpus = ["low"] * 5 + ["lowest"] * 2 + ["newer"] * 6 + ["wider"] * 3 + ["new"] * 2

    print("=" * 15, "BPE 算法核心原理演示", "=" * 15)
    print(f"测试语料库组成:")
    # 打印测试语料方便查看
    unique_words = set(corpus)
    for w in unique_words:
        print(f" - '{w}' : {corpus.count(w)}个")
    print(f"输入语料库总词数: {len(corpus)} \n")

    print("=" * 15, "第一阶段: 训练 BPE (构建词汇表)", "=" * 15)

    # 设定最高执行 10 次合并操作
    num_merges = 10
    bpe_codes, final_vocab = build_bpe_vocab(corpus, num_merges)

    print("\n" + "=" * 15, "第二阶段: 测试新词编码 (分词)", "=" * 15)
    # 测试集中既包含在训练时见过的词，也包含没见过的OOV（如'widest', 'lower'）词
    test_words = ["low", "lowest", "newer", "new", "widest", "lower", "wilder", "nower"]
    print("使用学到的合并规则，对以下单词进行 BPE 分词:\n")

    for word in test_words:
        encoded = encode_word(word, bpe_codes)
        print(f"输入词: '{word:<6}' =>  BPE Tokens 结果: {encoded}")
    print("\n观察结果: ")
    print(" 1. 'newer' 等高频词被合并为了一个整体token ('newer</w>')。")
    print(" 2. 'lower' (未登录词) 能够被有效拆分为已知的 'low' 和 'er</w>'。")
    print(" 3. 'widest' (未登录词) 被合理地拆分为了 'wid', 'e', 's', 't</w>'。")
    print("=" * 55)
