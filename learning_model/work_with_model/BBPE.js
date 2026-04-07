/**
 * BBPE (Byte-Level Byte Pair Encoding) 简易算法实现 JavaScript 版本
 * 
 * 核心功能：
 * 将原始字符串转换为 UTF-8 字节，再利用特定的可见字符映射去表示 0-255 字节，然后进行正常的词频合并过程。
 */

function bytesToUnicode() {
    const bs = [];
    const cs = [];
    // 能够正常显示的 ASCII 范围
    for (let i = 33; i <= 126; i++) bs.push(i); // '!' to '~'
    for (let i = 161; i <= 172; i++) bs.push(i); // '¡' to '¬'
    for (let i = 174; i <= 255; i++) bs.push(i); // '®' to 'ÿ'
    
    for (let b of bs) cs.push(b);
    
    let n = 0;
    // 补足剩余不足 256 位的部分到未被占用的 unicode 外延字符
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(256 + n);
            n++;
        }
    }
    
    const encodeMap = new Map();
    const decodeMap = new Map();
    for (let i = 0; i < bs.length; i++) {
        const byteVal = bs[i];
        const charVal = String.fromCharCode(cs[i]);
        encodeMap.set(byteVal, charVal);
        decodeMap.set(charVal, byteVal);
    }
    return { byteEncoder: encodeMap, byteDecoder: decodeMap };
}

const { byteEncoder, byteDecoder } = bytesToUnicode();
const textEncoder = new TextEncoder(); // 浏览器自带或 Node 环境下自带，可将字符串转 UTF-8 Uint8Array
const textDecoder = new TextDecoder('utf-8', { fatal: false }); // fatal false 表示替代非法字节

function getStats(vocab) {
    const pairs = new Map();
    for (const item of vocab) {
        const tokens = item.tokens;
        const count = item.count;
        for (let i = 0; i < tokens.length - 1; i++) {
            const pairKey = tokens[i] + ',' + tokens[i+1];
            pairs.set(pairKey, (pairs.get(pairKey) || 0) + count);
        }
    }
    return pairs;
}

function mergeVocab(pairKey, vocab) {
    const [p0, p1] = pairKey.split(',');
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

function buildBbpeVocab(words, numMerges) {
    const vocabMap = new Map();

    console.log("【初始字节级切分】(片段展示):");
    let countOutput = 0;
    
    for (const word of words) {
        let w = word.trim();
        if (!w) continue;
        
        // str -> utf8 bytes -> mapped chars
        const bytes = textEncoder.encode(w);
        const chars = Array.from(bytes).map(b => byteEncoder.get(b));
        chars.push("</w>");
        
        const key = chars.join(',');
        vocabMap.set(key, (vocabMap.get(key) || 0) + 1);
        
        if (countOutput < 3) {
            console.log(`  '${w}' -> UTF-8 bytes: [${bytes.join(', ')}] \n       -> BBPE 初始状态: [${chars.join(', ')}]`);
            countOutput++;
        }
    }
    console.log("-".repeat(50));

    let vocab = Array.from(vocabMap.entries()).map(([k, v]) => ({
        tokens: k.split(','),
        count: v
    }));

    const bpeCodes = new Map();

    for (let i = 0; i < numMerges; i++) {
        const pairs = getStats(vocab);
        if (pairs.size === 0) break;

        let bestPair = null;
        let maxCount = -1;
        for (const [pair, count] of pairs.entries()) {
            if (count > maxCount) {
                maxCount = count;
                bestPair = pair;
            }
        }

        bpeCodes.set(bestPair, i);
        console.log(`第 ${i + 1} 次合并: ${bestPair.replace(',', ' + ')} -> ${bestPair.replace(',', '')} (频次: ${maxCount})`);

        vocab = mergeVocab(bestPair, vocab);
    }

    console.log("-".repeat(50));
    console.log("【最终状态词表】(BBPE拆分合并结果):");
    vocab.forEach(v => {
        // 为了不在终端中由于不常见字符导致错乱，简单输出即可
        console.log(`  ${v.tokens.join(' ')} : ${v.count}`);
    });

    return { bpeCodes, vocab };
}

function encodeWord(word, bpeCodes) {
    if (!word) return [];

    const bytes = textEncoder.encode(word);
    let tokens = Array.from(bytes).map(b => byteEncoder.get(b));
    tokens.push("</w>");

    while (tokens.length > 1) {
        let minRank = Infinity;
        let pairToMerge = null;

        for (let i = 0; i < tokens.length - 1; i++) {
            const pairKey = tokens[i] + ',' + tokens[i + 1];
            if (bpeCodes.has(pairKey)) {
                const rank = bpeCodes.get(pairKey);
                if (rank < minRank) {
                    minRank = rank;
                    pairToMerge = pairKey;
                }
            }
        }

        if (!pairToMerge) break;

        const [p0, p1] = pairToMerge.split(',');
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
        ...Array(8).fill("你好"),
        ...Array(4).fill("你好啊"),
        ...Array(5).fill("测试"),
        ...Array(6).fill("hello")
    ];

    console.log("=".repeat(15), "BBPE JS版本 原理演示", "=".repeat(15));
    
    const { bpeCodes, vocab } = buildBbpeVocab(corpus, 15);

    console.log("\n" + "=".repeat(15), "推理：测试新词", "=".repeat(15));
    const testWords = ["你好", "你好呀", "测试版", "hi", "helloworld"];

    for (const word of testWords) {
        const encodedChars = encodeWord(word, bpeCodes);
        
        // 尝试还原 tokens 文本
        const decodedTokens = [];
        for (const t of encodedChars) {
            const cleanToken = t.replace('</w>', '');
            if (cleanToken) {
                const bytesArray = [];
                for (let i = 0; i < cleanToken.length; i++) {
                    const char = cleanToken[i];
                    if (byteDecoder.has(char)) {
                        bytesArray.push(byteDecoder.get(char));
                    }
                }
                const uint8View = new Uint8Array(bytesArray);
                // 替换非法字节字符，模拟 Python 的 errors='replace'
                let text = textDecoder.decode(uint8View); 
                decodedTokens.push(text || '<bytes>');
            } else {
                decodedTokens.push('</w>');
            }
        }

        console.log(`输入词: '${word}'`);
        console.log(` -> 编码映射 Tokens : [ ${encodedChars.join(', ')} ]`);
        console.log(` -> 文本解码呈现    : [ ${decodedTokens.join(', ')} ]`);
        console.log("-".repeat(30));
    }
}
