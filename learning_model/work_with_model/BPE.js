/**
 * BPE (Byte Pair Encoding) 简易算法实现 JavaScript 版本
 *
 * 核心原理：
 * 1. 初始状态下，将词汇表中的每个词拆分成单个字符，并在词尾添加特殊标记 `</w>`。
 * 2. 统计当前所有符号对的频率。
 * 3. 找到频率最高对合并。
 * 4. 循环直到达到 num_merges。
 * 5. 推理分词使用之前保存的规则优先级（按训练合并的顺序先后）。
 */

const END_WORD = '</w>';
const PAIR_SEP = ',';

function pairKey(a, b) {
    return `${a}${PAIR_SEP}${b}`;
}

function splitPairKey(key) {
    return key.split(PAIR_SEP);
}

function getStats(vocab) {
    const pairs = new Map();
    for (const item of vocab) {
        const tokens = item.tokens;
        const count = item.count;
        for (let i = 0; i < tokens.length - 1; i++) {
            const key = pairKey(tokens[i], tokens[i + 1]);
            pairs.set(key, (pairs.get(key) || 0) + count);
        }
    }
    return pairs;
}

function mergeVocab(pairKey, vocab) {
    const [p0, p1] = splitPairKey(pairKey);
    const newVocab = [];
    for (const item of vocab) {
        const tokens = item.tokens;
        const count = item.count;
        const newTokens = [];
        let i = 0;
        while (i < tokens.length) {
            if (i < tokens.length - 1 && tokens[i] === p0 && tokens[i + 1] === p1) {
                newTokens.push(p0 + p1);
                i += 2;
            } else {
                newTokens.push(tokens[i]);
                i += 1;
            }
        }
        newVocab.push({ tokens: newTokens, count });
    }
    return newVocab;
}

function buildBpeVocab(corpus, numMerges) {
    if (!Array.isArray(corpus)) {
        throw new TypeError('corpus 必须是字符串数组');
    }
    if (!Number.isInteger(numMerges) || numMerges < 0) {
        throw new TypeError('numMerges 必须是非负整数');
    }

    // 1. 初始化词表：拆分字符序列并加终止符
    const vocabMap = new Map();
    for (const rawWord of corpus) {
        const word = String(rawWord ?? '');
        let w = word.trim();
        if (!w) continue;
        const key = w.split('').join(PAIR_SEP) + `${PAIR_SEP}${END_WORD}`;
        vocabMap.set(key, (vocabMap.get(key) || 0) + 1);
    }

    let vocab = Array.from(vocabMap.entries()).map(([k, v]) => ({
        tokens: k.split(PAIR_SEP),
        count: v
    }));

    console.log("【初始状态词表】(单词及频次):");
    vocab.forEach(v => console.log(`  ${v.tokens.join(' ')} : ${v.count}`));
    console.log("-".repeat(50));

    const bpeCodes = new Map();

    // 2. 迭代合并最频繁的对
    for (let i = 0; i < numMerges; i++) {
        const pairs = getStats(vocab);
        if (pairs.size === 0) break;

        let bestPair = null;
        let maxCount = -1;
        for (const [pair, count] of pairs.entries()) {
            // 次数并列时使用字典序，保证结果稳定可复现
            if (count > maxCount || (count === maxCount && (bestPair === null || pair < bestPair))) {
                maxCount = count;
                bestPair = pair;
            }
        }

        bpeCodes.set(bestPair, i);
        const [left, right] = splitPairKey(bestPair);
        console.log(`第 ${i + 1} 次合并: ${left} + ${right} -> ${left}${right} (频次: ${maxCount})`);

        vocab = mergeVocab(bestPair, vocab);
    }

    console.log("-".repeat(50));
    console.log("【最终状态词表】(BPE拆分结果):");
    vocab.forEach(v => console.log(`  ${v.tokens.join(' ')} : ${v.count}`));

    return { bpeCodes, vocab };
}

function encodeWord(word, bpeCodes) {
    if (!word) return [];
    if (!(bpeCodes instanceof Map)) {
        throw new TypeError('bpeCodes 必须是 Map');
    }

    let tokens = word.split('').concat([END_WORD]);

    while (tokens.length > 1) {
        let minRank = Infinity;
        let pairToMerge = null;

        // 找所有相邻组合中 rank 最小的
        for (let i = 0; i < tokens.length - 1; i++) {
            const key = pairKey(tokens[i], tokens[i + 1]);
            if (bpeCodes.has(key)) {
                const rank = bpeCodes.get(key);
                if (rank < minRank) {
                    minRank = rank;
                    pairToMerge = key;
                }
            }
        }

        if (!pairToMerge) break;

        // 执行一次优先级最高的合并
        const [p0, p1] = splitPairKey(pairToMerge);
        const newTokens = [];
        let i = 0;
        while (i < tokens.length) {
            if (i < tokens.length - 1 && tokens[i] === p0 && tokens[i + 1] === p1) {
                newTokens.push(p0 + p1);
                i += 2;
            } else {
                newTokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = newTokens;
    }

    return tokens;
}

// ==== 测试运行 ====
if (require.main === module) {
    const corpus = [
        ...Array(5).fill("low"),
        ...Array(2).fill("lowest"),
        ...Array(6).fill("newer"),
        ...Array(3).fill("wider"),
        ...Array(2).fill("new")
    ];

    console.log("=".repeat(15), "BPE JS版本 原理演示", "=".repeat(15));

    // 训练
    const { bpeCodes, vocab } = buildBpeVocab(corpus, 10);

    console.log('vocab', vocab);
    console.log('bpeCodes', bpeCodes);

    // 推理测试
    console.log("\n" + "=".repeat(15), "推理：测试新词", "=".repeat(15));
    const testWords = ["low", "lowest", "newer", "new", "widest", "lower", "wilder", "nower"];

    for (const word of testWords) {
        const encoded = encodeWord(word, bpeCodes);
        console.log(`输入词: '${word}' => BPE Tokens: [ ${encoded.join(', ')} ]`);
    }
}
